import argparse
import asyncio
import json
import os
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import load_dotenv

from .clients import Models, Search, ServiceError
from .documents import read_proposal
from .engine import Artifacts, run


def validate_config(config: dict):
    for section in ("llm", "models", "search"):
        if not isinstance(config.get(section, {}), dict):
            raise ValueError(f"[{section}] must be a TOML table")
    models = config.get("models", {})
    for role in ("planner", "critic", "judge"):
        if not isinstance(models.get(role), str) or not models[role].strip():
            raise ValueError(f"Set models.{role} to your model alias")
    for role in ("novelty", "validity", "feasibility"):
        if role in models and (not isinstance(models[role], str) or not models[role].strip()):
            raise ValueError(f"models.{role} must be a nonempty model alias")
    llm = config.get("llm", {})
    if llm.get("backend", "proxy") not in ("proxy", "sdk"):
        raise ValueError("llm.backend must be proxy or sdk")
    bounds = {
        "llm": {"concurrency": (1, 8), "max_output_tokens": (512, 32000), "timeout_seconds": (1, 600),
                **{f"{stage}_max_output_tokens": (512, 32000)
                   for stage in ("plan", "novelty", "validity", "feasibility", "judge")}},
        "search": {"initial_queries": (1, 6), "followup_queries": (0, 6), "results_per_query": (1, 10),
                   "max_sources": (1, 50), "chars_per_source": (500, 15000), "concurrency": (1, 8)},
    }
    for section, fields in bounds.items():
        for key, (low, high) in fields.items():
            value = config.get(section, {}).get(key)
            if value is not None and (type(value) is not int or not low <= value <= high):
                raise ValueError(f"{section}.{key} must be an integer from {low} to {high}")
    if "json_mode" in llm and type(llm["json_mode"]) is not bool:
        raise ValueError("llm.json_mode must be true or false")
    if "temperature" in llm and (type(llm["temperature"]) not in (int, float) or not 0 <= llm["temperature"] <= 2):
        raise ValueError("llm.temperature must be between 0 and 2")


async def execute(proposal: str, config: dict, directory: Path, resume: Path | None = None):
    artifacts = Artifacts(directory)
    try:
        async with httpx.AsyncClient(timeout=config.get("llm", {}).get("timeout_seconds", 120)) as client:
            models = Models(client, config.get("llm", {}), artifacts.write)
            search = Search(client, config.get("search", {}), artifacts.write)
            await run(proposal, config, models, search, artifacts, lambda s: print(s, file=sys.stderr), resume=resume)
    except Exception as exc:
        # Store a safe summary: third-party errors may contain request credentials.
        artifacts.write("error", {"error_type": type(exc).__name__, "status": "incomplete"})
        raise


def main():
    parser = argparse.ArgumentParser(description="Evidence-grounded adversarial review using your LiteLLM models.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="Word .docx or UTF-8 text/Markdown proposal")
    source.add_argument("--text", help="Inline proposal")
    source.add_argument("--stdin", action="store_true", help="Read proposal from stdin")
    source.add_argument("--resume", type=Path, help="Reuse a run's saved evidence/reviews and retry missing critics plus the judge")
    parser.add_argument("--config", type=Path, default=Path("teardown.toml"))
    parser.add_argument("--out", type=Path, help="New run directory; must not already exist")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/input and show limits without network calls")
    args = parser.parse_args()
    try:
        # Use the config's directory, without searching unrelated parent folders.
        # Explicit shell variables take precedence over the local credentials file.
        load_dotenv(args.config.resolve().parent / ".env", override=False, interpolate=False)
        config = tomllib.loads(args.config.read_text(encoding="utf-8"))
        validate_config(config)
        if args.resume:
            for name in ("input.json", "plan.json", "sources.json"):
                if not (args.resume / name).is_file():
                    raise ValueError(f"Cannot resume: {name} is missing; resume requires a completed research stage")
            proposal = json.loads((args.resume / "input.json").read_text(encoding="utf-8"))["proposal"].strip()
        else:
            proposal = (read_proposal(args.file) if args.file else sys.stdin.read() if args.stdin else args.text).strip()
        if not proposal:
            raise ValueError("Proposal cannot be empty")
        if len(proposal) > 60000:
            raise ValueError("Proposal exceeds 60,000 characters; supply a focused summary")
        if args.dry_run:
            searches = config.get("search", {})
            print(json.dumps({"backend": config.get("llm", {}).get("backend", "proxy"), "models": config["models"],
                "proposal_characters": len(proposal), "normal_model_calls": 4 if args.resume else 5,
                "max_model_calls_with_repairs": 8 if args.resume else 10,
                "max_search_queries": 0 if args.resume else searches.get("initial_queries", 4) + searches.get("followup_queries", 2),
                "note": "Each network operation may retry once; no dollar budget is enforced locally."}, indent=2))
            return
        if not args.resume and not os.environ.get("TAVILY_API_KEY"):
            raise ValueError("Set TAVILY_API_KEY for internet research")
        directory = args.out or Path("runs") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
        directory.mkdir(parents=True, exist_ok=False)
        print(f"Run artifacts: {directory.resolve()}", file=sys.stderr)
        asyncio.run(execute(proposal, config, directory, resume=args.resume))
        print(f"Report: {(directory / 'report.md').resolve()}")
    except (OSError, ValueError, ServiceError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted; completed stage artifacts remain in the run directory.", file=sys.stderr)
        sys.exit(130)
