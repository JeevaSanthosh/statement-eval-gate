import pytest

from seg.prepare import repair_unit_counts


@pytest.mark.parametrize(
    "ocr, repaired",
    [
        ("8,395,242", "8,395.242"),      # decimal point read as a comma
        ("6.721.722", "6,721.722"),      # thousands comma read as a point
        ("7.290,600", "7,290.600"),      # both swapped
        ("167,617", "167.617"),          # no thousands part
        ("1,234,567,890", "1,234,567.890"),
    ],
)
def test_swapped_separators_in_unit_counts_are_put_back(ocr, repaired):
    assert repair_unit_counts(ocr) == repaired


@pytest.mark.parametrize(
    "text",
    [
        "2,143.679",              # already right, so the repair changes nothing
        "863.800",
        "£49,681.26",             # money has two decimal places
        "(€3,400,00)",            # OCR'd money keeps its two-digit tail
        "£1,234,567",             # anything after a currency sign is left alone
        "30/06/2026",
        "KLW48347824",
        "TDW 7866 6617",
        "ASP/123456/78",
        "5.9180",                 # a four-decimal price
        "8395,242",               # a dropped separator: too ambiguous to guess
        "worth 1,234.",           # a sentence-ending point is not part of the number
    ],
)
def test_everything_else_is_left_alone(text):
    assert repair_unit_counts(text) == text


def test_repairs_numbers_inside_a_table_row():
    row = "Orwell Short-Dated Bond Acc GBP 8,395,242 £49,681.26"
    assert repair_unit_counts(row) == "Orwell Short-Dated Bond Acc GBP 8,395.242 £49,681.26"


def test_repair_is_idempotent():
    text = "Fund A 8,395,242 £49,681.26\nFund B 6.721.722 £23,532.70"
    once = repair_unit_counts(text)
    assert repair_unit_counts(once) == once
