"""Source-specific row adapters for support-ticket extraction datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import hashlib
import json
import re

from .schemas import (
    CustomerContext,
    IssueCategory,
    PlanTier,
    PriorityLevel,
    ProductArea,
    SentimentLabel,
    SupportTicketExtraction,
    load_support_ticket_model,
)

MAPPING_VERSION = "v1"


@dataclass(frozen=True)
class CanonicalRowDraft:
    """Canonical row draft before split assignment and final validation."""

    record_id: str
    input_text: str
    target: SupportTicketExtraction
    split_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterReject:
    """Rejected row with a stable reason for reporting."""

    adapter_name: str
    source_record_id: str
    reason: str
    raw_excerpt: str


AdapterFn = Callable[[dict[str, Any]], CanonicalRowDraft]


PRIORITY_MAP = {
    "critical": PriorityLevel.URGENT,
    "urgent": PriorityLevel.URGENT,
    "sev1": PriorityLevel.URGENT,
    "high": PriorityLevel.HIGH,
    "medium": PriorityLevel.MEDIUM,
    "normal": PriorityLevel.MEDIUM,
    "moderate": PriorityLevel.MEDIUM,
    "low": PriorityLevel.LOW,
}

PLAN_MAP = {
    "free": PlanTier.FREE,
    "starter": PlanTier.FREE,
    "pro": PlanTier.PRO,
    "business": PlanTier.BUSINESS,
    "enterprise": PlanTier.ENTERPRISE,
}


def _stable_record_id(prefix: str, raw: dict[str, Any]) -> str:
    digest = hashlib.sha1(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _raw_excerpt(raw: dict[str, Any]) -> str:
    text = json.dumps(raw, sort_keys=True)
    return text[:240]


def _keywords_match(value: str, keywords: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(keyword in lowered for keyword in keywords)


def _map_priority(*values: str) -> PriorityLevel:
    joined = " ".join(value.lower() for value in values if value).strip()
    for keyword, priority in PRIORITY_MAP.items():
        if keyword in joined:
            return priority
    if _keywords_match(joined, ("can't", "cannot", "blocked", "broken", "refund", "charged twice", "error")):
        return PriorityLevel.HIGH
    return PriorityLevel.MEDIUM


def _map_plan_tier(*values: str) -> PlanTier | None:
    joined = " ".join(value.lower() for value in values if value).strip()
    for keyword, plan_tier in PLAN_MAP.items():
        if keyword in joined:
            return plan_tier
    return None


def _infer_issue_category(*values: str) -> IssueCategory:
    joined = " ".join(value.lower() for value in values if value)
    if _keywords_match(joined, ("invoice", "billing", "charge", "refund", "payment", "loan", "bank", "credit")):
        return IssueCategory.BILLING
    if _keywords_match(joined, ("login", "password", "access", "locked", "mfa", "account portal", "sign in")):
        return IssueCategory.ACCOUNT_ACCESS
    if _keywords_match(joined, ("bug", "error", "500", "crash", "broken", "sync", "timeout", "failed", "technical")):
        return IssueCategory.TECHNICAL_BUG
    if _keywords_match(joined, ("feature", "roadmap", "enhancement", "request")):
        return IssueCategory.FEATURE_REQUEST
    if _keywords_match(joined, ("integration", "connector", "webhook", "salesforce", "api")):
        return IssueCategory.INTEGRATION
    if _keywords_match(joined, ("question", "help", "documentation", "how do i", "where can i")):
        return IssueCategory.GENERAL_QUESTION
    return IssueCategory.OTHER


def _infer_product_area(*values: str) -> ProductArea:
    joined = " ".join(value.lower() for value in values if value)
    if _keywords_match(joined, ("mobile", "ios", "android")):
        return ProductArea.MOBILE_APP
    if _keywords_match(joined, ("api", "endpoint", "webhook")):
        return ProductArea.API
    if _keywords_match(joined, ("billing", "invoice", "charge", "payment")):
        return ProductArea.BILLING_PORTAL
    if _keywords_match(joined, ("account", "mfa", "login", "password")):
        return ProductArea.ACCOUNT_PORTAL
    if _keywords_match(joined, ("salesforce", "connector", "integration", "oauth")):
        return ProductArea.INTEGRATIONS
    if _keywords_match(joined, ("app", "workspace", "dashboard", "browser")):
        return ProductArea.WEB_APP
    if _keywords_match(joined, ("unknown", "general")):
        return ProductArea.UNKNOWN
    return ProductArea.OTHER


def _infer_sentiment(*values: str) -> SentimentLabel:
    joined = " ".join(value.lower() for value in values if value)
    if _keywords_match(joined, ("thank", "appreciate", "helpful", "love")):
        return SentimentLabel.POSITIVE
    if _keywords_match(joined, ("frustrated", "angry", "broken", "can't", "cannot", "urgent", "complaint", "error")):
        return SentimentLabel.NEGATIVE
    if _keywords_match(joined, ("while", "but", "however")) and _keywords_match(joined, ("thanks", "appreciate")):
        return SentimentLabel.MIXED
    return SentimentLabel.NEUTRAL


def _extract_name_from_email(email: str | None) -> str | None:
    value = _normalize_text(email)
    if not value or "@" not in value:
        return None
    local_part = value.split("@", 1)[0].replace(".", " ").replace("_", " ").strip()
    if not local_part:
        return None
    words = [word.capitalize() for word in local_part.split() if word]
    return " ".join(words) if words else None


def _summary_from_subject(subject: str, fallback: str) -> str:
    text = _normalize_text(subject) or _normalize_text(fallback)
    if not text:
        return "Customer support request"
    text = re.sub(r"\s+", " ", text)
    return text[:140]


def _build_support_target(
    *,
    summary: str,
    issue_text: str,
    priority_text: str = "",
    category_text: str = "",
    product_text: str = "",
    customer_name: str | None = None,
    account_id: str | None = None,
    plan_tier: PlanTier | None = None,
    actions_requested: list[str] | None = None,
    requires_human_followup: bool | None = None,
) -> SupportTicketExtraction:
    issue_category = _infer_issue_category(category_text, product_text, issue_text)
    priority = _map_priority(priority_text, issue_text, category_text)
    sentiment = _infer_sentiment(issue_text, priority_text, category_text)
    product_area = _infer_product_area(product_text, category_text, issue_text)
    followup = requires_human_followup
    if followup is None:
        followup = issue_category != IssueCategory.GENERAL_QUESTION or priority != PriorityLevel.LOW
    return SupportTicketExtraction(
        summary=summary,
        issue_category=issue_category,
        priority=priority,
        product_area=product_area,
        customer=CustomerContext(
            name=customer_name,
            account_id=account_id,
            plan_tier=plan_tier,
        ),
        sentiment=sentiment,
        requires_human_followup=followup,
        actions_requested=actions_requested or [],
    )


def adapt_json_extraction_source_row(raw: dict[str, Any]) -> CanonicalRowDraft:
    """Adapt an already-canonical JSON extraction row."""

    target = load_support_ticket_model(raw["target"])
    metadata = dict(raw.get("metadata", {}))
    return CanonicalRowDraft(
        record_id=_normalize_text(raw.get("record_id")) or _stable_record_id("json-extraction", raw),
        split_hint=_normalize_text(raw.get("split")) or None,
        input_text=_normalize_text(raw.get("input_text")),
        target=target,
        metadata=metadata,
    )


def adapt_hf_it_helpdesk_ticket_v1(raw: dict[str, Any]) -> CanonicalRowDraft:
    """Map IT helpdesk ticket rows into the support-ticket extraction schema."""

    subject = _normalize_text(raw.get("subject"))
    description = _normalize_text(raw.get("description"))
    if not description:
        raise ValueError("missing_description")
    requester_email = _normalize_text(raw.get("requester_email") or raw.get("requesterEmail"))
    priority_text = _normalize_text(raw.get("priority"))
    category_text = _normalize_text(raw.get("category"))
    text = f"Subject: {subject}\n\n{description}" if subject else description
    return CanonicalRowDraft(
        record_id=_normalize_text(raw.get("ticket_id") or raw.get("id")) or _stable_record_id("it-helpdesk", raw),
        split_hint=_normalize_text(raw.get("split")) or None,
        input_text=text,
        target=_build_support_target(
            summary=_summary_from_subject(subject, description),
            issue_text=text,
            priority_text=priority_text,
            category_text=category_text,
            customer_name=_extract_name_from_email(requester_email),
            plan_tier=_map_plan_tier(_normalize_text(raw.get("plan_tier")), text),
        ),
        metadata={"requester_email": requester_email or None},
    )


def adapt_hf_customer_support_ticket_v1(raw: dict[str, Any]) -> CanonicalRowDraft:
    """Map customer-support ticket rows into the support-ticket extraction schema."""

    subject = _normalize_text(raw.get("subject"))
    body = _normalize_text(raw.get("body") or raw.get("message") or raw.get("ticket_body"))
    if not body:
        raise ValueError("missing_body")
    queue = _normalize_text(raw.get("queue"))
    ticket_type = _normalize_text(raw.get("type"))
    priority_text = _normalize_text(raw.get("priority"))
    tags = raw.get("tags") or []
    tags_text = " ".join(str(tag) for tag in tags) if isinstance(tags, list) else _normalize_text(tags)
    answer = _normalize_text(raw.get("answer"))
    plan_hint = _normalize_text(raw.get("plan_tier"))
    text = f"Subject: {subject}\n\n{body}" if subject else body
    requires_human_followup = not _keywords_match(answer.lower(), ("help center", "documentation", "kb article"))
    return CanonicalRowDraft(
        record_id=_normalize_text(raw.get("ticket_id") or raw.get("id")) or _stable_record_id("support-ticket", raw),
        split_hint=_normalize_text(raw.get("split")) or None,
        input_text=text,
        target=_build_support_target(
            summary=_summary_from_subject(subject, body),
            issue_text=text,
            priority_text=priority_text,
            category_text=" ".join(part for part in (queue, ticket_type, tags_text) if part),
            product_text=tags_text,
            customer_name=_normalize_text(raw.get("customer_name")) or None,
            account_id=_normalize_text(raw.get("account_id")) or None,
            plan_tier=_map_plan_tier(plan_hint, body),
            requires_human_followup=requires_human_followup,
        ),
        metadata={
            "queue": queue or None,
            "ticket_type": ticket_type or None,
            "tags": tags if isinstance(tags, list) else [tags] if tags else [],
        },
    )


def adapt_cfpb_complaint_csv_v1(raw: dict[str, Any]) -> CanonicalRowDraft:
    """Map CFPB complaint rows into the support-ticket extraction schema."""

    narrative = _normalize_text(raw.get("consumer_complaint_narrative"))
    if not narrative:
        raise ValueError("missing_consumer_complaint_narrative")
    product = _normalize_text(raw.get("product"))
    issue = _normalize_text(raw.get("issue"))
    sub_product = _normalize_text(raw.get("sub_product"))
    complaint_id = _normalize_text(raw.get("complaint_id"))
    summary = f"{product or 'Consumer complaint'}: {issue or 'reported issue'}".strip(": ")
    return CanonicalRowDraft(
        record_id=complaint_id or _stable_record_id("cfpb", raw),
        split_hint=_normalize_text(raw.get("split")) or None,
        input_text=narrative,
        target=_build_support_target(
            summary=summary[:140],
            issue_text=narrative,
            category_text=" ".join(part for part in (product, sub_product, issue) if part),
            product_text=product,
            requires_human_followup=True,
        ),
        metadata={
            "product": product or None,
            "sub_product": sub_product or None,
            "issue": issue or None,
            "company_response_to_consumer": _normalize_text(raw.get("company_response_to_consumer")) or None,
        },
    )


def adapt_hf_schema_discipline_json_v1(raw: dict[str, Any]) -> CanonicalRowDraft:
    """Accept only generic structured-output rows that already match this schema."""

    text = _normalize_text(raw.get("text") or raw.get("input") or raw.get("prompt"))
    output = raw.get("output") or raw.get("target") or raw.get("response")
    if not text or output is None:
        raise ValueError("missing_text_or_output")
    target = load_support_ticket_model(output)
    return CanonicalRowDraft(
        record_id=_normalize_text(raw.get("record_id") or raw.get("id")) or _stable_record_id("schema-discipline", raw),
        split_hint=_normalize_text(raw.get("split")) or None,
        input_text=text,
        target=target,
        metadata={"schema_discipline_source": True},
    )


ADAPTERS: dict[str, AdapterFn] = {
    "json_extraction": adapt_json_extraction_source_row,
    "hf_it_helpdesk_ticket_v1": adapt_hf_it_helpdesk_ticket_v1,
    "hf_customer_support_ticket_v1": adapt_hf_customer_support_ticket_v1,
    "cfpb_complaint_csv_v1": adapt_cfpb_complaint_csv_v1,
    "hf_schema_discipline_json_v1": adapt_hf_schema_discipline_json_v1,
}


def adapt_source_row(adapter_name: str, raw: dict[str, Any]) -> CanonicalRowDraft:
    """Adapt a raw row with the named source adapter."""

    adapter = ADAPTERS.get(adapter_name)
    if adapter is None:
        raise ValueError(f"Unsupported adapter_name: {adapter_name}")
    return adapter(raw)


def reject_row(adapter_name: str, raw: dict[str, Any], reason: str) -> AdapterReject:
    """Build a normalized rejected-row record."""

    source_record_id = (
        _normalize_text(raw.get("record_id"))
        or _normalize_text(raw.get("id"))
        or _normalize_text(raw.get("ticket_id"))
        or _normalize_text(raw.get("complaint_id"))
        or _stable_record_id("rejected", raw)
    )
    return AdapterReject(
        adapter_name=adapter_name,
        source_record_id=source_record_id,
        reason=reason,
        raw_excerpt=_raw_excerpt(raw),
    )
