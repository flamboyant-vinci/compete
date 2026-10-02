import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from pydantic import ValidationError

from teardown.cli import validate_config
from teardown.clients import Models, Search, ServiceError, canonical_url, post_json
from teardown.engine import Artifacts, run
from teardown.models import Plan, Review, Source, Verdict, check_references


PLAN = {
    "steelman": "Improve retrieval under a fixed budget.",
    "claims": [{"id": "C1", "statement": "Synthetic data improves retrieval.",
                "assumption": "Questions represent real users.", "falsifier": "No gain on a held-out human evaluation."}],
    "queries": ["synthetic retrieval benchmark", "synthetic retrieval limitations", "retrieval baseline", "retrieval study"],
}
QUOTE = "Synthetic questions did not represent real user queries."
FINDING = {
    "claim_ids": ["C1"], "title": "Evaluation mismatch", "severity": "major", "basis": "evidence",
    "objection": "Synthetic evaluation may overstate gains.",
    "citations": [{"source_id": "S1", "quote": QUOTE}],
    "strongest_defense": "The proposed domain may differ.", "cheap_test": "Label 50 real questions.",
    "rejection_threshold": "Proposed: reject the claimed gain if improvement is below 15 points.",
}
REVIEW = {"findings": [FINDING], "followup_queries": ["human retrieval study", "synthetic question distribution"]}
VERDICT = {"decision": "test_first", "rationale": "The proposed evaluation does not resolve the key assumption.",
           "steelman": PLAN["steelman"], "findings": [FINDING], "surviving_strengths": ["Narrow scope."],
           "uncertainties": ["Real user performance remains unmeasured."], "next_steps": ["Build a human test set."]}
CONFIG = {"llm": {"backend": "proxy", "base_url": "http://gateway/v1"},
          "models": {"planner": "plan", "critic": "review", "judge": "judge"},
          "search": {"initial_queries": 4, "followup_queries": 2}}


def response(content: dict | str, finish_reason="stop"):
    return {"choices": [{"message": {"content": json.dumps(content) if isinstance(content, dict) else content},
                         "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}}


def source():
    return Source(id="S1", title="Study", url="https://example.org/study", text=QUOTE,
                  kind="page_extract", retrieved_at="2026-09-22", queries=["study"])


class IntegrityTests(unittest.TestCase):
    def test_rejects_invented_quotes_and_ids(self):
        plan = Plan.model_validate(PLAN)
        review = Review.model_validate(REVIEW)
        check_references(review, [source()], plan)
        review.findings[0].citations[0].quote = "This text was never retrieved from any page."
        with self.assertRaisesRegex(ValueError, "Quote does not occur"):
            check_references(review, [source()], plan)
        review.findings[0].citations[0].quote = QUOTE
        review.findings[0].citations[0].source_id = "S99"
        with self.assertRaisesRegex(ValueError, "Unknown source"):
            check_references(review, [source()], plan)
        review.findings[0].claim_ids = ["C99"]
        with self.assertRaisesRegex(ValueError, "unknown claim"):
            check_references(review, [source()], plan)

    def test_evidence_requires_a_citation(self):
        with self.assertRaises(ValidationError):
            Review.model_validate({"findings": [{**FINDING, "citations": []}], "followup_queries": []})
        Review.model_validate({"findings": [{**FINDING, "basis": "unknown", "citations": []}], "followup_queries": []})

    def test_rejects_duplicate_claims_and_bad_budgets(self):
        with self.assertRaises(ValidationError):
            Plan.model_validate({**PLAN, "claims": PLAN["claims"] * 2})
        validate_config(CONFIG)
        for bad in (-1, 100, True):
            with self.assertRaises(ValueError):
                validate_config({**CONFIG, "search": {"initial_queries": bad}})

    def test_url_validation(self):
        self.assertEqual(canonical_url("https://EXAMPLE.org/paper#section"), "https://example.org/paper")
        for url in ("javascript:alert(1)", "file:///etc/passwd", "https://user:secret@example.org", "https://[bad"):
            self.assertIsNone(canonical_url(url))


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_length_response_is_saved_and_retried(self):
        saved = {}
        models = Models(None, {"judge_max_output_tokens": 12000}, saved.__setitem__)
        empty = {"choices": [{"finish_reason": "length", "message": {
            "content": None, "reasoning_content": "Internal reasoning is not the final answer"}}],
            "usage": {"completion_tokens": 3500}}
        models.complete = AsyncMock(side_effect=[empty, response(VERDICT)])
        result = await models.structured("judge", "model", "Judge", {}, Verdict)
        self.assertEqual(result.decision, "test_first")
        self.assertEqual(saved["judge.1"]["finish_reason"], "length")
        self.assertEqual(saved["judge.1"]["content"], "")
        self.assertTrue(saved["judge.1"]["has_reasoning"])
        self.assertNotIn("Internal reasoning", json.dumps(saved))
        self.assertEqual(models.complete.call_args.kwargs["max_output_tokens"], 12000)

    async def test_empty_stop_response_does_not_bypass_retry_limit(self):
        saved = {}
        models = Models(None, {}, saved.__setitem__)
        models.complete = AsyncMock(return_value={"choices": [{"message": {"content": None}, "finish_reason": "stop"}]})
        with self.assertRaisesRegex(ServiceError, "after one repair"):
            await models.structured("judge", "model", "Judge", {}, Verdict)
        self.assertEqual(models.complete.await_count, 2)
        self.assertIn("judge.2", saved)

    async def test_content_parts_supported_without_using_reasoning(self):
        models = Models(None, {}, lambda *_: None)
        models.complete = AsyncMock(return_value={"choices": [{"message": {"content": [
            {"type": "reasoning", "text": "not output"},
            {"type": "text", "text": json.dumps(PLAN)}]}, "finish_reason": "stop"}]})
        self.assertEqual((await models.structured("plan", "model", "Plan", {}, Plan)).steelman, PLAN["steelman"])

    async def test_resume_reuses_research_and_valid_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = Artifacts(Path(tmp) / "previous")
            previous.write("plan", PLAN)
            previous.write("sources", {"sources": [source().model_dump()], "queries": ["saved query"], "warnings": []})
            previous.write("review.novelty", REVIEW)
            previous.write("review.validity", {"broken": "must be retried"})
            previous.write("validity.2", {"content": json.dumps({**REVIEW, "findings": [
                {**FINDING, "citations": [{"source_id": "S999", "quote": QUOTE}]}]})})
            current = Artifacts(Path(tmp) / "current")
            models = Models(None, {}, current.write)
            models.structured = AsyncMock(side_effect=[Review.model_validate(REVIEW), Review.model_validate(REVIEW),
                                                       Verdict.model_validate(VERDICT)])
            search = Search(None, {}, current.write)
            search.research = AsyncMock()
            result = await run("Proposal", CONFIG, models, search, current, lambda _: None, resume=previous.directory)
            search.research.assert_not_called()
            self.assertEqual([c.args[0] for c in models.structured.call_args_list], ["validity", "feasibility", "judge"])
            self.assertIn("Unknown source ID", models.structured.call_args_list[0].args[3]["repair_errors"])
            self.assertEqual(result["search_queries"], ["saved query"])
            self.assertEqual(result["warnings"], [])
            self.assertTrue((current.directory / "review.novelty.json").exists())
            self.assertTrue((current.directory / "report.md").exists())

    async def test_full_run_with_bounded_followups_and_saved_evidence(self):
        requests = []

        def handle(request):
            body = json.loads(request.content)
            requests.append((request, body))
            if request.url.host == "api.tavily.com":
                self.assertEqual(request.headers["authorization"], "Bearer test-search")
                self.assertEqual(body["include_raw_content"], "text")
                return httpx.Response(200, json={"query": body["query"], "results": [
                    {"url": "https://example.org/study#section", "title": "Study", "content": QUOTE, "raw_content": QUOTE}]})
            self.assertEqual(str(request.url), "http://gateway/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-proxy")
            self.assertNotIn("temperature", body)
            self.assertNotIn("response_format", body)
            if body["model"] == "judge":
                data = json.loads(body["messages"][1]["content"])
                self.assertEqual(len(data["reviews"]), 3)
                self.assertEqual(len(data["sources"][0]["queries"]), 6)
            return httpx.Response(200, json=response({"plan": PLAN, "review": REVIEW, "judge": VERDICT}[body["model"]]))

        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {
            "LITELLM_BASE_URL": "http://gateway/v1", "LITELLM_API_KEY": "test-proxy", "TAVILY_API_KEY": "test-search"
        }):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                artifacts = Artifacts(Path(tmp))
                models = Models(client, CONFIG["llm"], artifacts.write)
                search = Search(client, CONFIG["search"], artifacts.write)
                result = await run("Proposal", CONFIG, models, search, artifacts, lambda _: None)
                self.assertEqual(result["model_calls"], 5)
                self.assertEqual(len(result["search_queries"]), 6)
                self.assertEqual(len(result["sources"]), 1)
                self.assertEqual(len(requests), 11)
                self.assertIn("test_first", (Path(tmp) / "report.md").read_text())
                self.assertTrue((Path(tmp) / "report.json").exists())
                self.assertNotIn("test-proxy", "".join(p.read_text() for p in Path(tmp).iterdir()))

    async def test_invalid_citation_gets_one_repair(self):
        models = Models(None, {}, lambda *_: None)
        bad = {**REVIEW, "findings": [{**FINDING, "citations": [{"source_id": "S999", "quote": QUOTE}]}]}
        models.complete = AsyncMock(side_effect=[response(bad), response(REVIEW)])
        value = await models.structured("critic", "model", "Review", {}, Review,
            lambda value: check_references(value, [source()], Plan.model_validate(PLAN)))
        self.assertEqual(value.findings[0].citations[0].source_id, "S1")
        self.assertEqual(models.complete.await_count, 2)

    async def test_invalid_output_stops_after_one_repair(self):
        models = Models(None, {}, lambda *_: None)
        models.complete = AsyncMock(return_value=response("not json"))
        with self.assertRaisesRegex(ServiceError, "after one repair"):
            await models.structured("plan", "model", "Plan", {}, Plan)
        self.assertEqual(models.complete.await_count, 2)

    async def test_truncated_valid_json_is_not_accepted(self):
        models = Models(None, {}, lambda *_: None)
        models.complete = AsyncMock(side_effect=[response(PLAN, "length"), response(PLAN)])
        await models.structured("plan", "model", "Plan", {}, Plan)
        self.assertEqual(models.complete.await_count, 2)

    async def test_search_failure_and_excerpt_fallback_are_explicit(self):
        search = Search(None, {}, lambda *_: None)
        search._query = AsyncMock(side_effect=[ServiceError("unavailable"), {"results": [
            {"url": "https://example.org", "title": "Excerpt", "content": QUOTE, "raw_content": None},
            {"url": "javascript:alert(1)", "content": "bad"}]}])
        await search.research(["failed", "works", "WORKS", "over-budget"], 2)
        self.assertEqual(search.queries, ["failed", "works"])
        self.assertEqual(len(search.warnings), 1)
        self.assertEqual(search.sources[0].kind, "search_excerpt")
        self.assertEqual(len(search.sources), 1)

    async def test_missing_critic_is_visible_in_report_and_judge_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Models(None, {}, lambda *_: None)
            models.structured = AsyncMock(side_effect=[Plan.model_validate(PLAN), Review.model_validate(REVIEW),
                ServiceError("broken"), Review.model_validate(REVIEW), Verdict.model_validate(VERDICT)])
            search = Search(None, {}, lambda *_: None)
            search.sources = [source()]
            search.research = AsyncMock()
            result = await run("Proposal", CONFIG, models, search, Artifacts(Path(tmp)), lambda _: None)
            self.assertIn("validity review failed", result["warnings"][0])
            judge_data = models.structured.call_args.args[3]
            self.assertIn("validity review failed", judge_data["review_warnings"][0])

    async def test_transient_retry_and_no_auth_retry(self):
        statuses = [429, 200, 401]
        count = 0

        def handle(_):
            nonlocal count
            count += 1
            return httpx.Response(statuses.pop(0), json={"ok": True})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with patch("teardown.clients.asyncio.sleep", new_callable=AsyncMock):
                self.assertEqual(await post_json(client, "https://example.org", headers={}, body={}), {"ok": True})
                self.assertEqual(count, 2)
                with self.assertRaisesRegex(ServiceError, "401"):
                    await post_json(client, "https://example.org", headers={}, body={})
                self.assertEqual(count, 3)


if __name__ == "__main__":
    unittest.main()
