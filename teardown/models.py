from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Claim(Record):
    id: str = Field(pattern=r"^C[1-9][0-9]*$")
    statement: str = Field(min_length=1)
    assumption: str = Field(min_length=1)
    falsifier: str = Field(min_length=1)


class Plan(Record):
    steelman: str = Field(min_length=1)
    claims: list[Claim] = Field(min_length=1, max_length=8)
    queries: list[str] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def unique_claims(self):
        if len({c.id for c in self.claims}) != len(self.claims):
            raise ValueError("Claim IDs must be unique")
        if any(not q.strip() for q in self.queries):
            raise ValueError("Search queries cannot be blank")
        return self


class Source(Record):
    id: str
    title: str
    url: str
    text: str
    kind: Literal["page_extract", "search_excerpt"]
    retrieved_at: str
    queries: list[str]


class Citation(Record):
    source_id: str
    quote: str = Field(min_length=12, max_length=400)


class Finding(Record):
    claim_ids: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)
    severity: Literal["critical", "major", "minor"]
    basis: Literal["evidence", "inference", "unknown"]
    objection: str = Field(min_length=1)
    citations: list[Citation]
    strongest_defense: str = Field(min_length=1)
    cheap_test: str = Field(min_length=1)
    rejection_threshold: str = Field(min_length=1)

    @model_validator(mode="after")
    def evidence_needs_citations(self):
        if self.basis == "evidence" and not self.citations:
            raise ValueError("Evidence-based findings require citations")
        return self


class Review(Record):
    findings: list[Finding] = Field(max_length=5)
    followup_queries: list[str] = Field(max_length=2)


class Verdict(Record):
    decision: Literal["proceed", "revise", "test_first", "reject", "insufficient_evidence"]
    rationale: str = Field(min_length=1)
    steelman: str = Field(min_length=1)
    findings: list[Finding] = Field(max_length=8)
    surviving_strengths: list[str]
    uncertainties: list[str]
    next_steps: list[str] = Field(min_length=1, max_length=5)


def check_references(value: Review | Verdict, sources: list[Source], plan: Plan) -> None:
    """Check provenance, not semantic entailment or source reliability."""
    by_id = {s.id: s for s in sources}
    claim_ids = {c.id for c in plan.claims}
    errors = []
    for index, finding in enumerate(value.findings, 1):
        if not set(finding.claim_ids) <= claim_ids:
            errors.append(f"Finding {index} references an unknown claim ID")
        for number, citation in enumerate(finding.citations, 1):
            source = by_id.get(citation.source_id)
            if source is None:
                errors.append(f"Finding {index}, citation {number}: Unknown source ID: {citation.source_id}")
            elif " ".join(citation.quote.split()) not in " ".join(source.text.split()):
                errors.append(f"Finding {index}, citation {number}: Quote does not occur in {citation.source_id}")
    if errors:
        raise ValueError("\n".join(errors))
