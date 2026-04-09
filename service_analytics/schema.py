"""Data schema for an extracted service row.

Kept deliberately flexible — the spreadsheet columns vary by shop, so the
core fields are typed and everything else lands in ``extra`` as a dict.
Comparisons and dedupe use the typed fields when present and fall back to
``extra`` values by header name.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RowSource:
    """Provenance for a single extracted row."""

    frame_path: str = ""          # path of the frame it came from
    frame_time_s: float = 0.0     # video timestamp the frame was sampled at
    model: str = ""               # OCR model identifier
    confidence: Optional[float] = None  # optional model-reported confidence


@dataclass
class ServiceRow:
    """One service line item extracted from the spreadsheet.

    The interlock shop's report has roughly these fields visible:
      - service date
      - client name ("Last, First")
      - client id (a ~14-16 digit account number)
      - technician name
      - payment status (e.g. NoPaymentMade)
      - several dollar columns (different fee types)
      - one or more follow-up / due dates

    Fields beyond the core set are stored in ``extra`` keyed by normalized
    header names. Missing/unknown values are ``None``.
    """

    service_date: Optional[date] = None
    client_name: Optional[str] = None
    client_id: Optional[str] = None
    technician: Optional[str] = None
    payment_status: Optional[str] = None
    amount_total: Optional[float] = None
    # flexible fee breakdown keyed by column header (normalized lowercase)
    amounts: Dict[str, float] = field(default_factory=dict)
    # extra date columns keyed by header (normalized lowercase)
    other_dates: Dict[str, date] = field(default_factory=dict)
    # anything else the model surfaced
    extra: Dict[str, Any] = field(default_factory=dict)
    sources: List[RowSource] = field(default_factory=list)

    # ---- dedupe key -------------------------------------------------------

    def dedupe_key(self) -> str:
        """Return a stable key for row deduplication.

        Prefers (client_id, service_date) since client_id is the most
        unique field visible in the screenshot. Falls back to
        (client_name, service_date) when client_id is missing.
        """
        date_part = self.service_date.isoformat() if self.service_date else ""
        if self.client_id:
            return f"id:{self.client_id.strip()}|{date_part}"
        name = (self.client_name or "").strip().lower()
        return f"name:{name}|{date_part}"

    # ---- serialization ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.service_date is not None:
            d["service_date"] = self.service_date.isoformat()
        d["other_dates"] = {
            k: v.isoformat() if isinstance(v, date) else v
            for k, v in self.other_dates.items()
        }
        return d

    def to_flat_row(self) -> Dict[str, Any]:
        """Flatten for CSV export. Nested dicts become ``amount_<key>`` etc."""
        flat: Dict[str, Any] = {
            "service_date": self.service_date.isoformat() if self.service_date else "",
            "client_name": self.client_name or "",
            "client_id": self.client_id or "",
            "technician": self.technician or "",
            "payment_status": self.payment_status or "",
            "amount_total": self.amount_total if self.amount_total is not None else "",
        }
        for k, v in sorted(self.amounts.items()):
            flat[f"amount_{k}"] = v
        for k, v in sorted(self.other_dates.items()):
            flat[f"date_{k}"] = v.isoformat() if isinstance(v, date) else v
        for k, v in sorted(self.extra.items()):
            flat[f"extra_{k}"] = v
        return flat

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ServiceRow":
        row = cls()
        row.service_date = _parse_date(d.get("service_date"))
        row.client_name = _clean_str(d.get("client_name"))
        cid = _clean_str(d.get("client_id"))
        row.client_id = re.sub(r"\s+", "", cid) if cid else None
        row.technician = _clean_str(d.get("technician"))
        row.payment_status = _clean_str(d.get("payment_status"))
        row.amount_total = _parse_money(d.get("amount_total"))

        amounts = d.get("amounts") or {}
        if isinstance(amounts, dict):
            row.amounts = {
                _norm_key(k): v
                for k, v in (
                    (k, _parse_money(v)) for k, v in amounts.items()
                )
                if v is not None
            }

        other_dates = d.get("other_dates") or {}
        if isinstance(other_dates, dict):
            row.other_dates = {
                _norm_key(k): v
                for k, v in (
                    (k, _parse_date(v)) for k, v in other_dates.items()
                )
                if v is not None
            }

        extra = d.get("extra") or {}
        if isinstance(extra, dict):
            row.extra = {_norm_key(k): v for k, v in extra.items() if v not in (None, "")}

        src_raw = d.get("sources") or []
        if isinstance(src_raw, list):
            for s in src_raw:
                if isinstance(s, dict):
                    row.sources.append(
                        RowSource(
                            frame_path=s.get("frame_path", ""),
                            frame_time_s=float(s.get("frame_time_s", 0.0) or 0.0),
                            model=s.get("model", ""),
                            confidence=s.get("confidence"),
                        )
                    )
        return row


# ---------------------------------------------------------------------------
# Parsing helpers (tolerant of the messy output vision models produce)
# ---------------------------------------------------------------------------


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%m-%d-%Y",
    "%m-%d-%y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%b %d, %Y",
    "%B %d, %Y",
)


def _parse_date(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    s = str(value).strip()
    if not s or s.lower() in ("null", "none", "n/a", "na"):
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # last-ditch: look for m/d/yy anywhere in the string
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", s)
    if m:
        mo, d, y = m.groups()
        y_int = int(y)
        if y_int < 100:
            y_int += 2000 if y_int < 70 else 1900
        try:
            return date(y_int, int(mo), int(d))
        except ValueError:
            return None
    return None


_MONEY_RE = re.compile(r"-?\$?\s*([\d,]+(?:\.\d+)?)")


def _parse_money(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in ("null", "none", "n/a", "na", "-"):
        return None
    m = _MONEY_RE.search(s)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        val = float(num)
        return -val if s.lstrip().startswith("-") else val
    except ValueError:
        return None


def _clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("null", "none", "n/a", "na"):
        return None
    return s


def _norm_key(key: str) -> str:
    s = str(key).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


# ---------------------------------------------------------------------------
# JSON lines I/O
# ---------------------------------------------------------------------------


def load_jsonl(path) -> List[ServiceRow]:
    """Load ServiceRow objects from a JSON-lines file."""
    rows: List[ServiceRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(ServiceRow.from_dict(d))
    return rows


def dump_jsonl(rows: Iterable[ServiceRow], path) -> int:
    """Write ServiceRow objects as JSON lines. Returns count."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), default=str))
            f.write("\n")
            n += 1
    return n
