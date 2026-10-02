import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .clients import Models, Search
from .models import Plan, Review, Source, Verdict, check_references
from .prompts import JUDGE, LENSES, PLAN, REVIEW


class Artifacts:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, value: dict):
        target = self.directory / f"{name}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)


async def run(proposal: str, config: dict, models: Models, search: Search,
              artifacts: Artifacts, progress: Callable[[str], None] = print,
              resume: Path | None = None) -> dict:
    model_cfg = config["models"]
    limits = config.get("search", {})
    initial_budget = limits.get("initial_queries", 4)
    followup_budget = limits.get("followup_queries", 2)
    artifacts.write("input", {"proposal": proposal, "config": config})
    if resume:
        progress("Resuming saved claims and evidence; no searches will be repeated...")
        plan = Plan.model_validate_json((resume / "plan.json").read_text(encoding="utf-8"))
        saved = json.loads((resume / "sources.json").read_text(encoding="utf-8"))
        search.sources = [Source.model_validate(s) for s in saved["sources"]]
        search.queries = saved["queries"]
        search.warnings = saved["warnings"]
        artifacts.write("resume", {"from": str(resume.resolve()), "accounting": "Model calls/usage cover this attempt only; searches are reused."})
    else:
        progress("Extracting claims and planning research...")
        plan = await models.structured("plan", model_cfg["planner"], PLAN,
            {"proposal": proposal, "query_budget": initial_budget,
             "date": datetime.now(timezone.utc).date().isoformat()}, Plan)
    artifacts.write("plan", plan.model_dump())
    if not resume:
        progress("Searching for prior art, counterevidence and supporting evidence...")
        await search.research(plan.queries, initial_budget)
    artifacts.write("sources.initial", {"sources": [s.model_dump() for s in search.sources], "warnings": search.warnings})
    if not search.sources:
        search.warnings.append("No web evidence retrieved; this run cannot establish empirical objections or novelty.")
    data = {"proposal": proposal, "plan": plan.model_dump(),
            "sources": [s.model_dump() for s in search.sources], "research_warnings": list(search.warnings)}
    progress("Reviewing novelty, validity and feasibility...")
    async def review(lens: str, focus: str):
        if resume and (resume / f"review.{lens}.json").exists():
            try:
                result = Review.model_validate_json((resume / f"review.{lens}.json").read_text(encoding="utf-8"))
                check_references(result, search.sources, plan)
            except ValueError:
                progress(f"Saved {lens} review has invalid structure or citations; retrying...")
            else:
                progress(f"Reusing saved {lens} review...")
                artifacts.write(f"review.{lens}", result.model_dump())
                return result
        review_data = data
        if resume:
            # Carry the failed attempt and all available errors into a resumed repair.
            for attempt in (2, 1):
                raw_path = resume / f"{lens}.{attempt}.json"
                if not raw_path.exists():
                    continue
                raw = json.loads(raw_path.read_text(encoding="utf-8")).get("content", "")
                try:
                    check_references(Review.model_validate_json(raw), search.sources, plan)
                except ValueError as exc:
                    review_data = {**data, "previous_invalid_review": raw,
                                   "repair_errors": str(exc)[:6000],
                                   "repair_instruction": "Correct these errors while preserving supported objections. "
                                       "Copy short continuous quotes exactly from source.text; do not rewrite or join passages."}
                break
        result = await models.structured(lens, model_cfg.get(lens, model_cfg["critic"]),
            REVIEW + "\nAssigned lens: " + focus, review_data, Review,
            lambda value: check_references(value, search.sources, plan))
        artifacts.write(f"review.{lens}", result.model_dump())
        return result
    # Each critic sees the same evidence but not the other critics' outputs.
    responses = await asyncio.gather(*(review(lens, focus) for lens, focus in LENSES.items()), return_exceptions=True)
    reviews = {}
    warnings = []
    for lens, result in zip(LENSES, responses):
        if isinstance(result, Exception):
            warnings.append(f"{lens} review failed ({type(result).__name__}); this lens is missing.")
            progress(f"Warning: {warnings[-1]}")
        else:
            reviews[lens] = result
    if not reviews:
        raise RuntimeError("All critics failed; inspect stage artifacts and model configuration")
    # Round-robin prevents the first lens from monopolizing the search budget.
    followups = [r.followup_queries[i] for i in range(2) for r in reviews.values() if i < len(r.followup_queries)]
    if not resume and followups and followup_budget:
        progress("Investigating the highest-priority evidence gaps...")
        await search.research(followups, followup_budget)
    artifacts.write("sources", {"sources": [s.model_dump() for s in search.sources],
                                 "queries": search.queries, "warnings": search.warnings})
    progress("Adjudicating objections and writing the report...")
    verdict = await models.structured("judge", model_cfg["judge"], JUDGE,
        {**data, "sources": [s.model_dump() for s in search.sources],
         "reviews": {k: v.model_dump() for k, v in reviews.items()},
         "research_warnings": search.warnings, "review_warnings": warnings}, Verdict,
        lambda value: check_references(value, search.sources, plan))
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "plan": plan.model_dump(),
              "verdict": verdict.model_dump(), "sources": [s.model_dump() for s in search.sources],
              "warnings": search.warnings + warnings, "usage": models.usage,
              "model_calls": models.calls, "search_queries": search.queries,
              "resumed_from": str(resume.resolve()) if resume else None}
    artifacts.write("report", report)
    (artifacts.directory / "report.md").write_text(render(report), encoding="utf-8")
    return report


def render(report: dict) -> str:
    verdict = report["verdict"]
    sources = {s["id"]: s for s in report["sources"]}
    lines = ["# Adversarial teardown", "", f"**Decision: {verdict['decision']}**", "", verdict["rationale"],
             "", "## Strongest version", "", verdict["steelman"]]
    if report["warnings"]:
        lines += ["", "## Run limitations", ""] + [f"- {w}" for w in report["warnings"]]
    lines += ["", "## Load-bearing claims", ""]
    for claim in report["plan"]["claims"]:
        lines += [f"- **{claim['id']}: {claim['statement']}**",
                  f"  Assumption: {claim['assumption']} Falsifier: {claim['falsifier']}"]
    lines += ["", "## Ranked objections", ""]
    for i, finding in enumerate(verdict["findings"], 1):
        lines += [f"### {i}. {finding['title']}", "",
                  f"**{finding['severity']} · {finding['basis']} · {', '.join(finding['claim_ids'])}**", "",
                  finding["objection"], "", f"**Strongest defense:** {finding['strongest_defense']}", "",
                  f"**Cheap test:** {finding['cheap_test']}", "",
                  f"**Rejection threshold:** {finding['rejection_threshold']}", ""]
        for citation in finding["citations"]:
            source = sources[citation["source_id"]]
            lines += [f"[{source['id']}](<{source['url']}>) ({source['kind']}): “{' '.join(citation['quote'].split())}”", ""]
    for heading, key in [("What survives", "surviving_strengths"), ("Uncertainties", "uncertainties"), ("Next actions", "next_steps")]:
        lines += ["", f"## {heading}", ""] + [f"- {item}" for item in verdict[key]]
    lines += ["", "## Evidence ledger", ""]
    for source in sources.values():
        lines += [f"- [{source['id']}](<{source['url']}>) — {source['title']} ({source['kind']}, retrieved {source['retrieved_at']})"]
    lines += ["", "## Run accounting", "", f"Model calls: {report['model_calls']}; search queries: {len(report['search_queries'])}.",
              "Token usage is saved in report.json. Dollar costs depend on your gateway and search provider.", "",
              "Citation checks confirm source IDs and quoted text, not factual truth or semantic support.", ""]
    if report.get("resumed_from"):
        lines += ["This report resumes an earlier run. Model calls and usage cover this attempt only; search queries were reused.", ""]
    return "\n".join(lines)
