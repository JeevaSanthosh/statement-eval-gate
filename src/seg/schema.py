"""Target schema for statement extraction, and the normalisation used when scoring.

Every field the extractor must return is defined here once. Scoring compares
normalised values, so formatting differences (case, spacing, trailing zeros)
never count as errors, but a wrong number or a wrong date always does.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from pydantic import BaseModel, Field

SCALAR_FIELDS = [
    "provider",
    "account_number",
    "statement_start",
    "statement_end",
    "opening_value",
    "closing_value",
    "contributions",
    "withdrawals",
    "fees",
]
LIST_FIELDS = ["holdings"]
ALL_FIELDS = SCALAR_FIELDS + LIST_FIELDS

AMOUNT_FIELDS = {"opening_value", "closing_value", "contributions", "withdrawals", "fees"}
DATE_FIELDS = {"statement_start", "statement_end"}


class Holding(BaseModel):
    fund_name: str = Field(description="Fund name exactly as printed, without '(continued)' markers")
    units: float | None = Field(default=None, description="Number of units held at the end of the period")
    value: float | None = Field(default=None, description="Value in GBP at the end of the period")


class Statement(BaseModel):
    provider: str | None = Field(default=None, description="Name of the company that issued the statement")
    account_number: str | None = Field(default=None, description="Account or plan number")
    statement_start: str | None = Field(default=None, description="First day of the statement period, YYYY-MM-DD")
    statement_end: str | None = Field(default=None, description="Last day of the statement period, YYYY-MM-DD")
    opening_value: float | None = Field(default=None, description="Portfolio value at the start of the period, GBP")
    closing_value: float | None = Field(default=None, description="Portfolio value at the end of the period, GBP")
    contributions: float | None = Field(default=None, description="Total paid in during the period, GBP")
    withdrawals: float | None = Field(default=None, description="Total taken out during the period, GBP, as a positive number")
    fees: float | None = Field(default=None, description="Total fees charged during the period, GBP. Null if the statement does not show fees")
    holdings: list[Holding] = Field(default_factory=list, description="One entry per fund held at the end of the period")


# --- normalisation -----------------------------------------------------------

_SPACE = re.compile(r"\s+")


def norm_text(value: object) -> str | None:
    if value is None:
        return None
    text = _SPACE.sub(" ", str(value)).strip().casefold()
    return text or None


def norm_account(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"[^0-9a-z]", "", str(value).casefold())
    return text or None


def norm_amount(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value).replace(",", "").replace("£", "")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    except (InvalidOperation, ValueError):
        return None


def norm_units(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value).replace(",", "")).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None


def norm_date(value: object) -> str | None:
    """Strict ISO dates. The schema asks for YYYY-MM-DD, so any other format counts as wrong."""
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value).strip()).isoformat()
    except ValueError:
        return None


def norm_scalar(field: str, value: object) -> object:
    if field in AMOUNT_FIELDS:
        return norm_amount(value)
    if field in DATE_FIELDS:
        return norm_date(value)
    if field == "account_number":
        return norm_account(value)
    return norm_text(value)


def norm_holding(h: dict) -> tuple:
    return (norm_text(h.get("fund_name")), norm_units(h.get("units")), norm_amount(h.get("value")))
