"""Check the model's holdings against the rows of the statement text.

A small model copies most of a row correctly but slips on the details: it drops
the end of a fund name ("Acc GBP" becomes "Acc"), adds an ending that isn't there,
or rounds a unit count to two decimal places. The right answer is in the text, so
code can put it back:

  fund name  If the model's name and exactly one row's name match except that one
             runs on for extra whole words, take the row's name.
  units      If the row's unit count, rounded to 0, 1 or 2 decimal places, equals
             the model's, take the row's unit count.

Anything ambiguous or unmatched is left as the model gave it, so grounding can
only replace an answer with text from the statement itself.
"""

from __future__ import annotations

import copy
import re

# The first number with a decimal point in a row: the unit count in every layout here.
_QUANTITY = re.compile(r"\d[\d,]*\.\d+")


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _extends(longer: str, shorter: str) -> bool:
    """True if `longer` is `shorter` plus one or more whole words."""
    return longer.startswith(shorter + " ")


def holding_rows(text: str) -> list[tuple[str, float]]:
    """(name, units) for every line that reads like a holding: words, then a number."""
    rows = []
    for line in text.splitlines():
        match = _QUANTITY.search(line)
        if not match:
            continue
        words = line[: match.start()].split()
        while words and not any(ch.isalpha() for ch in words[-1]):  # OCR debris such as "§," or "|"
            words.pop()
        if words and not words[0][0].isdigit():  # skip dated transaction lines
            rows.append((" ".join(words), float(match.group().replace(",", ""))))
    return rows


def ground_holdings(prediction: dict, text: str) -> tuple[dict, list[str]]:
    """Return the prediction with names and units checked against the text, and what changed."""
    prediction = copy.deepcopy(prediction)
    rows = holding_rows(text)
    changes = []
    for holding in prediction.get("holdings", []):
        name = _norm(holding.get("fund_name") or "")
        if not name:
            continue
        exact = [row for row in rows if _norm(row[0]) == name]
        if exact:
            row = exact[0]
        else:
            candidates = {r[0] for r in rows if _extends(_norm(r[0]), name) or _extends(name, _norm(r[0]))}
            if len(candidates) != 1:
                continue
            row = next(r for r in rows if r[0] in candidates)
            changes.append(f"fund_name: {holding['fund_name']!r} -> {row[0]!r}")
            holding["fund_name"] = row[0]
        units = holding.get("units")
        if units is not None and units != row[1] and any(round(row[1], k) == units for k in range(3)):
            changes.append(f"units of {row[0]!r}: {units} -> {row[1]}")
            holding["units"] = row[1]
    return prediction, changes
