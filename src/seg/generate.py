"""Generate synthetic investment statements with known answers.

Four fictional providers, each with its own layout and its own trap:

  A  Brackenford Investments  clean summary box; extra price column to ignore
  B  Tidewell Platform        holdings table split across pages with the last row
                              repeated after the break; payments, withdrawals and
                              charges only in a summary table at the very end
  C  Ashcombe Pensions        values given in a sentence; individual transactions
                              listed, totals only in a period summary
  D  Kestrel Lane Wealth      no fees figure at all (correct answer is null);
                              withdrawals in brackets; ordinal dates

A share of statements is also rendered as a noisy, slightly rotated scan with no
text layer, so they have to go through OCR.

Every provider, fund and person here is invented. The output is deterministic for
a given seed.
"""

from __future__ import annotations

import calendar
import json
import random
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PROVIDERS = {
    "A": ("Brackenford Investments", "BRK"),
    "B": ("Tidewell Platform", "TDW"),
    "C": ("Ashcombe Pensions", "ASP"),
    "D": ("Kestrel Lane Wealth", "KLW"),
}
FUND_BRANDS = ["Meridian", "Larchmont", "Orwell", "Saltmarsh", "Pennine", "Calder", "Wrekin", "Hartsop"]
FUND_TYPES = [
    "Global Equity Index",
    "UK Equity Income",
    "Sterling Corporate Bond",
    "Global Property Securities",
    "Emerging Markets Equity",
    "UK Gilt Tracker",
    "Multi-Asset Balanced",
    "Japan Equity",
    "US Equity Tracker",
    "Short-Dated Bond",
    "European Smaller Companies",
    "Asia Pacific ex Japan",
]
SHARE_CLASSES = ["Acc", "Inc", "Acc GBP", "C Acc"]

STYLES = getSampleStyleSheet()


@dataclass
class Truth:
    id: str
    layout: str
    scanned: bool
    provider: str
    account_number: str
    statement_start: date
    statement_end: date
    opening_value: float
    closing_value: float
    contributions: float
    withdrawals: float
    fees: float | None
    holdings: list[dict] = field(default_factory=list)
    traps: list[str] = field(default_factory=list)

    def answer(self) -> dict:
        """The ground truth in exactly the shape the extractor must return."""
        return {
            "provider": self.provider,
            "account_number": self.account_number,
            "statement_start": self.statement_start.isoformat(),
            "statement_end": self.statement_end.isoformat(),
            "opening_value": self.opening_value,
            "closing_value": self.closing_value,
            "contributions": self.contributions,
            "withdrawals": self.withdrawals,
            "fees": self.fees,
            "holdings": self.holdings,
        }


# --- formatting helpers ------------------------------------------------------

def money(x: float) -> str:
    return f"£{x:,.2f}"


def ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def fmt_date(d: date, layout: str) -> str:
    if layout == "A":
        return d.strftime("%d %b %Y")
    if layout == "B":
        return d.strftime("%d/%m/%Y")
    if layout == "C":
        return f"{d.day} {d.strftime('%B %Y')}"
    return f"{ordinal(d.day)} {d.strftime('%B %Y')}"


def table(rows: list[list[str]], widths: list[float], header: bool = True, total: bool = False) -> Table:
    t = Table(rows, colWidths=[w * mm for w in widths])
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style.append(("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9))
    if total:
        style += [("FONT", (0, -1), (-1, -1), "Helvetica-Bold", 9), ("LINEABOVE", (0, -1), (-1, -1), 0.5, colors.black)]
    t.setStyle(TableStyle(style))
    return t


def para(text: str, style: str = "Normal") -> Paragraph:
    return Paragraph(text, STYLES[style])


# --- truth -------------------------------------------------------------------

def make_truth(idx: int, layout: str, scanned: bool, rng: random.Random) -> Truth:
    name, prefix = PROVIDERS[layout]
    year = rng.choice([2024, 2025, 2026])
    quarter = rng.randint(1, 4) if year < 2026 else rng.randint(1, 2)
    start = date(year, 3 * quarter - 2, 1)
    end_month = 3 * quarter
    end = date(year, end_month, calendar.monthrange(year, end_month)[1])

    digits = "".join(rng.choice("0123456789") for _ in range(8))
    account = {
        "A": f"{prefix}-{digits}",
        "B": f"{prefix} {digits[:4]} {digits[4:]}",
        "C": f"{prefix}/{digits[:6]}/{digits[6:]}",
        "D": f"{prefix}{digits}",
    }[layout]

    n_funds = rng.randint(6, 9) if layout == "B" else rng.randint(3, 7)
    names = rng.sample([f"{b} {t} {rng.choice(SHARE_CLASSES)}" for b in FUND_BRANDS for t in FUND_TYPES], n_funds)
    holdings = []
    for fund in names:
        units = round(rng.uniform(40, 9000), 3)
        price = rng.uniform(0.8, 6.5)
        holdings.append({"fund_name": fund, "units": units, "value": round(units * price, 2)})
    closing = round(sum(h["value"] for h in holdings), 2)

    monthly = rng.choice([0, 0, 100, 150, 200, 250, 400, 500, 750, 1000])
    contributions = float(monthly * 3)
    withdrawals = 0.0 if rng.random() < 0.7 else float(rng.randrange(500, 5000, 50))
    fees = round(closing * rng.uniform(0.001, 0.004) / 4, 2)  # one quarter of a 0.1-0.4% annual charge
    growth = rng.uniform(-0.04, 0.06)
    opening = round((closing - contributions + withdrawals + fees) / (1 + growth), 2)

    traps = {
        "A": ["extra_price_column"],
        "B": ["duplicated_row_after_page_break", "totals_only_in_end_summary"],
        "C": ["values_in_prose", "totals_only_in_period_summary"],
        "D": ["fees_not_shown", "bracketed_withdrawals", "ordinal_dates"],
    }[layout] + (["scanned"] if scanned else [])

    return Truth(
        id=f"stmt_{idx:03d}",
        layout=layout,
        scanned=scanned,
        provider=name,
        account_number=account,
        statement_start=start,
        statement_end=end,
        opening_value=opening,
        closing_value=closing,
        contributions=contributions,
        withdrawals=withdrawals,
        fees=None if layout == "D" else fees,
        holdings=holdings,
        traps=traps,
    )


# --- layouts -----------------------------------------------------------------

def story_a(t: Truth) -> list:
    rows = [["Fund", "Units", "Price", "Value"]]
    for h in t.holdings:
        rows.append([h["fund_name"], f"{h['units']:,.3f}", f"{h['value'] / h['units']:.4f}", money(h["value"])])
    rows.append(["Total", "", "", money(t.closing_value)])
    return [
        para(t.provider, "Title"),
        para("Quarterly Investment Statement", "Heading2"),
        para(f"Account number: {t.account_number}"),
        para(f"Statement period: {fmt_date(t.statement_start, 'A')} to {fmt_date(t.statement_end, 'A')}"),
        Spacer(1, 6 * mm),
        para("Summary", "Heading3"),
        table(
            [
                ["Opening value", money(t.opening_value)],
                ["Payments in", money(t.contributions)],
                ["Withdrawals", money(t.withdrawals)],
                ["Charges", money(t.fees)],
                ["Closing value", money(t.closing_value)],
            ],
            [70, 40],
            header=False,
        ),
        Spacer(1, 6 * mm),
        para("Your investments", "Heading3"),
        table(rows, [85, 30, 25, 35], total=True),
    ]


def story_b(t: Truth, rng: random.Random) -> list:
    header = ["Fund", "Units", "Value"]
    body = [[h["fund_name"], f"{h['units']:,.3f}", money(h["value"])] for h in t.holdings]
    split = rng.randint(3, len(body) - 2)
    first, rest = body[:split], body[split - 1 :]  # last row of page 1 repeats on page 2
    return [
        para(t.provider, "Title"),
        para(f"Plan reference: {t.account_number}"),
        para(f"Period: {fmt_date(t.statement_start, 'B')} - {fmt_date(t.statement_end, 'B')}"),
        Spacer(1, 4 * mm),
        para(f"Value at start of period: {money(t.opening_value)}"),
        para(f"Value at end of period: {money(t.closing_value)}"),
        Spacer(1, 6 * mm),
        para("Your holdings", "Heading3"),
        table([header] + first, [95, 35, 40]),
        PageBreak(),
        para("Your holdings (continued)", "Heading3"),
        table([header] + rest, [95, 35, 40]),
        Spacer(1, 8 * mm),
        para("Activity summary", "Heading3"),
        table(
            [
                ["Contributions received", money(t.contributions)],
                ["Withdrawals paid", money(t.withdrawals)],
                ["Platform and fund charges", money(t.fees)],
            ],
            [80, 40],
            header=False,
        ),
        Spacer(1, 10 * mm),
        para("Assets are held by Fernhill Nominees Limited on behalf of Tidewell Platform clients.", "Italic"),
    ]


def story_c(t: Truth, rng: random.Random) -> list:
    entries = []
    months = [date(t.statement_start.year, t.statement_start.month + i, rng.randint(1, 25)) for i in range(3)]
    if t.contributions:
        for d in months:
            entries.append((d, "Monthly contribution", money(t.contributions / 3)))
    if t.withdrawals:
        entries.append((months[1], "Withdrawal", f"-{money(t.withdrawals)}"))
    f1 = f2 = round(t.fees / 3, 2)
    for d, amount in zip(months, [f1, f2, round(t.fees - f1 - f2, 2)]):
        entries.append((d, "Annual management charge", f"-{money(amount)}"))
    entries.sort(key=lambda e: e[0])
    tx = [["Date", "Description", "Amount"]] + [[fmt_date(d, "C"), desc, amt] for d, desc, amt in entries]

    holdings = [["Fund", "Units", "Value"]] + [
        [h["fund_name"], f"{h['units']:,.3f}", money(h["value"])] for h in t.holdings
    ]
    return [
        para(t.provider, "Title"),
        para("Your pension statement", "Heading2"),
        para(f"Member number: {t.account_number}"),
        para(f"This statement covers {fmt_date(t.statement_start, 'C')} – {fmt_date(t.statement_end, 'C')}."),
        Spacer(1, 4 * mm),
        para(
            f"On {fmt_date(t.statement_start, 'C')} your pension was worth {money(t.opening_value)}. "
            f"On {fmt_date(t.statement_end, 'C')} it was worth {money(t.closing_value)}."
        ),
        Spacer(1, 6 * mm),
        para("Transactions", "Heading3"),
        table(tx, [35, 85, 35]),
        Spacer(1, 6 * mm),
        para("Where your pension is invested", "Heading3"),
        table(holdings, [95, 30, 35]),
        Spacer(1, 6 * mm),
        para("Period summary", "Heading3"),
        table(
            [
                ["Total contributions", money(t.contributions)],
                ["Total withdrawals", money(t.withdrawals)],
                ["Total charges", money(t.fees)],
            ],
            [70, 40],
            header=False,
        ),
    ]


def story_d(t: Truth) -> list:
    withdrawn = f"({money(t.withdrawals)})" if t.withdrawals else money(0)
    holdings = [["Investment", "Holding", "Market value"]] + [
        [h["fund_name"], f"{h['units']:,.3f}", money(h["value"])] for h in t.holdings
    ]
    return [
        para(t.provider, "Title"),
        para("Portfolio valuation", "Heading2"),
        para(f"Client account {t.account_number}"),
        para(f"Reporting period {fmt_date(t.statement_start, 'D')} to {fmt_date(t.statement_end, 'D')}"),
        Spacer(1, 6 * mm),
        table(
            [
                ["Portfolio value brought forward", money(t.opening_value)],
                ["Money added", money(t.contributions)],
                ["Money withdrawn", withdrawn],
                ["Portfolio value carried forward", money(t.closing_value)],
            ],
            [80, 40],
            header=False,
        ),
        Spacer(1, 6 * mm),
        table(holdings, [95, 30, 40]),
        Spacer(1, 8 * mm),
        para(
            "Charges for this period will be shown in your annual costs and charges report. "
            "Custody services are provided by Fernhill Nominees Limited.",
            "Italic",
        ),
    ]


def render(t: Truth, path: Path, rng: random.Random) -> None:
    story = {
        "A": lambda: story_a(t),
        "B": lambda: story_b(t, rng),
        "C": lambda: story_c(t, rng),
        "D": lambda: story_d(t),
    }[t.layout]()
    doc = SimpleDocTemplate(str(path), pagesize=A4, invariant=1, title=f"{t.provider} statement")
    doc.build(story)


SCAN_DPI = 130  # with the noise below, Tesseract misreads about one amount in ten: hard, not hopeless


def scan(native: Path, out: Path, seed: int) -> None:
    """Turn a native PDF into an image-only PDF that looks like a cheap scan:
    low resolution, slight rotation, faded toner, noise, blur and JPEG artefacts."""
    import io

    from pdf2image import convert_from_path

    noise = np.random.default_rng(seed)
    pages = []
    for page in convert_from_path(str(native), dpi=SCAN_DPI, grayscale=True):
        page = page.rotate(noise.uniform(-2, 2), fillcolor=255, resample=Image.BICUBIC)
        arr = np.asarray(page, dtype=np.float32)
        arr = 255 - (255 - arr) * 0.8 + noise.normal(0, 22, arr.shape)
        page = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.9))
        buf = io.BytesIO()
        page.save(buf, "JPEG", quality=35)
        buf.seek(0)
        pages.append(Image.open(buf).convert("L"))
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=SCAN_DPI)


def generate(out_dir: Path, n: int = 60, seed: int = 7, scanned_share: float = 0.3) -> list[Truth]:
    rng = random.Random(seed)
    pdf_dir, truth_dir = out_dir / "statements", out_dir / "truth"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    truths = []
    for i in range(1, n + 1):
        layout = "ABCD"[(i - 1) % 4]
        t = make_truth(i, layout, rng.random() < scanned_share, rng)
        pdf = pdf_dir / f"{t.id}.pdf"
        render(t, pdf, rng)
        if t.scanned:
            native = pdf.with_suffix(".native.pdf")
            pdf.rename(native)
            scan(native, pdf, seed * 1000 + i)
            native.unlink()
        (truth_dir / f"{t.id}.json").write_text(json.dumps(t.answer(), indent=2) + "\n", encoding="utf-8")
        truths.append(t)

    manifest = [
        {"id": t.id, "layout": t.layout, "provider": t.provider, "scanned": t.scanned, "traps": t.traps}
        for t in truths
    ]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return truths
