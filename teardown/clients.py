import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Callable, TypeVar
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import ValidationError

from .models import Record, Source
from .prompts import COMMON

T = TypeVar("T", bound=Record)


class ServiceError(RuntimeError):
    pass


async def post_json(client: httpx.AsyncClient, url: str, *, headers: dict, body: dict) -> dict:
    """One retry for transient errors; never include credentials/response bodies in errors."""
    for attempt in range(2):
        try:
            response = await client.post(url, headers=headers, json=body)
        except httpx.TransportError:
            if attempt == 0:
                await asyncio.sleep(1)
                continue
            raise ServiceError("Network request failed after retry") from None
        if response.status_code in (408, 429, 500, 502, 503, 504) and attempt == 0:
            await asyncio.sleep(1)
            continue
        if response.is_error:
            raise ServiceError(f"Service returned HTTP {response.status_code}; check endpoint, credentials and model configuration")
        try:
            data = response.json()
        except ValueError:
            raise ServiceError("Service returned invalid JSON") from None
        if not isinstance(data, dict):
            raise ServiceError("Service returned an unexpected response shape")
        return data
    raise AssertionError("unreachable")


class Models:
    def __init__(self, client: httpx.AsyncClient, config: dict, record: Callable):
        self.client = client
        self.config = config
        self.record = record
        self.usage: list[dict] = []
        self.calls = 0
        self.semaphore = asyncio.Semaphore(config.get("concurrency", 3))

    async def complete(self, model: str, messages: list[dict], *, max_output_tokens: int | None = None) -> dict:
        cfg = self.config
        async with self.semaphore:
            self.calls += 1
            body = {"model": model, "messages": messages,
                    "max_tokens": max_output_tokens or cfg.get("max_output_tokens", 3500)}
            # Opt-in: some served models do not support JSON mode or temperature.
            if cfg.get("json_mode", False):
                body["response_format"] = {"type": "json_object"}
            if "temperature" in cfg:
                body["temperature"] = cfg["temperature"]
            if cfg.get("backend", "proxy") == "sdk":
                try:
                    from litellm import acompletion
                except ImportError:
                    raise ServiceError("SDK backend requires: pip install '.[sdk]'") from None
                try:
                    result = await acompletion(**body, timeout=cfg.get("timeout_seconds", 120), num_retries=1)
                    return result.model_dump()
                except Exception as exc:
                    raise ServiceError(f"LiteLLM SDK failed ({type(exc).__name__}); check provider configuration") from None
            base = os.environ.get("LITELLM_BASE_URL", cfg.get("base_url", "http://localhost:4000/v1"))
            key = os.environ.get("LITELLM_API_KEY", "")
            return await post_json(self.client, base.rstrip("/") + "/chat/completions",
                                   headers={"Authorization": f"Bearer {key}"} if key else {}, body=body)

    async def structured(self, stage: str, model: str, instruction: str, data: dict,
                         schema: type[T], validate: Callable[[T], None] | None = None) -> T:
        messages = [
            {"role": "system", "content": COMMON + instruction + "\nJSON Schema:\n" + json.dumps(schema.model_json_schema())},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
        ]
        # One repair also covers invalid citations. No unbounded agent loops.
        for attempt in range(2):
            token_limit = self.config.get(f"{stage}_max_output_tokens", self.config.get("max_output_tokens", 3500))
            result = await self.complete(model, messages, max_output_tokens=token_limit)
            usage = result.get("usage") or {}
            self.usage.append({"stage": stage, "model": model, "attempt": attempt + 1, "usage": usage})
            choices = result.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
            message = choice.get("message") or {}
            message = message if isinstance(message, dict) else {}
            raw_content = message.get("content")
            if isinstance(raw_content, list):
                content = "".join(part["text"] for part in raw_content
                                  if isinstance(part, dict) and part.get("type") in ("text", "output_text")
                                  and isinstance(part.get("text"), str))
            else:
                content = raw_content if isinstance(raw_content, str) else ""
            # Record even empty/truncated responses. Never substitute reasoning for the answer.
            self.record(f"{stage}.{attempt + 1}", {"model": model, "content": content, "usage": usage,
                "finish_reason": choice.get("finish_reason"), "max_output_tokens": token_limit,
                "message_fields": sorted(message), "content_type": type(raw_content).__name__,
                "has_reasoning": bool(message.get("reasoning_content") or message.get("reasoning")),
                "has_refusal": bool(message.get("refusal")), "has_tool_calls": bool(message.get("tool_calls"))})
            if message.get("refusal") or choice.get("finish_reason") == "content_filter":
                raise ServiceError(f"{stage}: provider refused or filtered the response; inspect stage metadata")
            try:
                if choice.get("finish_reason") == "length":
                    raise ValueError(f"Output token limit reached ({token_limit}); produce a shorter complete JSON object. "
                                     f"If repeated, increase llm.{stage}_max_output_tokens in the configuration")
                if not content.strip():
                    raise ValueError(f"No final text (finish_reason={choice.get('finish_reason')!r}); "
                                     "return the requested JSON in message.content, without tool calls")
                clean = content.strip()
                if clean.startswith("```") and clean.endswith("```"):
                    clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
                parsed = schema.model_validate_json(clean)
                if validate:
                    validate(parsed)
                return parsed
            except (ValueError, ValidationError) as exc:
                if attempt == 1:
                    raise ServiceError(f"{stage}: invalid or empty output/citations after one repair; "
                                       f"inspect {stage}.{attempt + 1}.json. "
                                       f"For truncated output, increase llm.{stage}_max_output_tokens") from None
                if content.strip():
                    messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Repair the JSON using the original data. "
                    "For citations, copy a SHORT continuous substring exactly, including PDF hyphenation. "
                    "Do not add ellipses, join separate passages or rewrite quotes. If no quote supports an "
                    "objection, remove that citation and use inference/unknown as appropriate. Errors:\n" + str(exc)[:6000]})
        raise AssertionError("unreachable")


def canonical_url(url: str) -> str | None:
    try:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            return None
        return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", parts.query, ""))
    except ValueError:
        return None


class Search:
    def __init__(self, client: httpx.AsyncClient, config: dict, record: Callable):
        self.client = client
        self.config = config
        self.record = record
        self.sources: list[Source] = []
        self.queries: list[str] = []
        self.warnings: list[str] = []
        self.semaphore = asyncio.Semaphore(config.get("concurrency", 3))

    async def _query(self, query: str) -> dict:
        async with self.semaphore:
            return await post_json(self.client, "https://api.tavily.com/search",
                headers={"Authorization": "Bearer " + os.environ.get("TAVILY_API_KEY", "")},
                body={"query": query, "search_depth": "advanced",
                      "max_results": self.config.get("results_per_query", 4),
                      "include_raw_content": "text", "include_answer": False})

    async def research(self, queries: list[str], budget: int) -> None:
        unique = []
        seen = {q.casefold() for q in self.queries}
        for query in queries:
            query = " ".join(query.split())[:400]
            if query and query.casefold() not in seen:
                seen.add(query.casefold())
                unique.append(query)
        unique = unique[:budget]
        self.queries.extend(unique)
        responses = await asyncio.gather(*(self._query(q) for q in unique), return_exceptions=True)
        by_url = {s.url: s for s in self.sources}
        for query, result in zip(unique, responses):
            if isinstance(result, Exception):
                self.warnings.append(f"Search failed for {query!r} ({type(result).__name__})")
                self.record(f"search.{len(self.warnings)}.error", {"query": query, "error_type": type(result).__name__})
                continue
            self.record(f"search.{self.queries.index(query) + 1}", result)
            rows = result.get("results", [])
            if not isinstance(rows, list):
                self.warnings.append(f"Malformed search results for {query!r}")
                continue
            added = 0
            for row in rows[:self.config.get("results_per_query", 4)]:
                if not isinstance(row, dict):
                    continue
                url = canonical_url(str(row.get("url", "")))
                if not url:
                    continue
                if url in by_url:
                    if query not in by_url[url].queries:
                        by_url[url].queries.append(query)
                    added += 1
                    continue
                raw = row.get("raw_content")
                excerpt = row.get("content")
                content = raw if isinstance(raw, str) and raw.strip() else excerpt
                if not isinstance(content, str) or not content.strip():
                    continue
                # Keep the ranked excerpt first when full extraction is long.
                if content == raw and isinstance(excerpt, str) and excerpt.strip():
                    content = excerpt + "\n\n" + raw
                if len(self.sources) >= self.config.get("max_sources", 20):
                    continue
                source = Source(id=f"S{len(self.sources) + 1}", title=str(row.get("title") or url),
                    url=url, text=content[:self.config.get("chars_per_source", 5000)],
                    kind="page_extract" if content != excerpt else "search_excerpt",
                    retrieved_at=datetime.now(timezone.utc).isoformat(), queries=[query])
                self.sources.append(source)
                by_url[url] = source
                added += 1
            if not added:
                self.warnings.append(f"No usable new or existing sources for {query!r}")
