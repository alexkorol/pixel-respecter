"""Deduplicate service rows extracted from overlapping video frames.

Because the video is scrolling, the same row appears in several adjacent
frames. Each appearance may have slightly different OCR output: one frame
might be mid-scroll and miss a dollar amount, another might misread a
digit in the client id, etc.

Dedupe strategy
---------------
1. Group rows by a ``primary_key`` that is stable across OCR variations:
   ``(client_id, service_date)`` if both are present, else
   ``(client_name, service_date)``, else the row is considered unique.

2. Merge rows inside each group field-by-field, preferring non-null values
   and using a simple majority vote when multiple non-null values disagree.
   Amounts and other_dates dicts are union-merged with the same vote rule.

3. Sources are concatenated so each merged row retains provenance for every
   frame it was seen in.

Rows whose primary key is literally just an empty date (no id, no name,
no date) are dropped — these are almost always OCR noise.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional

from service_analytics.schema import ServiceRow

logger = logging.getLogger(__name__)


def dedupe_rows(rows: Iterable[ServiceRow]) -> List[ServiceRow]:
    """Merge rows that represent the same underlying service record.

    Returns rows sorted by ``service_date`` (ascending; nulls last) then
    ``client_name``.
    """
    groups: Dict[str, List[ServiceRow]] = defaultdict(list)
    dropped = 0
    for row in rows:
        key = row.dedupe_key()
        # Pure-empty keys come from completely unreadable rows
        if key in ("id:|", "name:|") or key.endswith("|") and ":" not in key:
            dropped += 1
            continue
        if not row.client_id and not row.client_name and not row.service_date:
            dropped += 1
            continue
        groups[key].append(row)

    if dropped:
        logger.info("Dropped %d rows with no identifying fields", dropped)

    merged: List[ServiceRow] = []
    for key, bucket in groups.items():
        merged.append(_merge_bucket(bucket))

    merged.sort(
        key=lambda r: (
            r.service_date is None,  # False < True, so non-null first
            r.service_date or _EMPTY_DATE,
            (r.client_name or "").lower(),
        )
    )
    logger.info("Deduped %d rows -> %d unique", sum(len(b) for b in groups.values()), len(merged))
    return merged


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


from datetime import date as _date_type

_EMPTY_DATE = _date_type(1900, 1, 1)


def _merge_bucket(bucket: List[ServiceRow]) -> ServiceRow:
    """Merge several observations of the same row into one."""
    if len(bucket) == 1:
        return bucket[0]

    merged = ServiceRow()
    merged.service_date = _vote([r.service_date for r in bucket])
    merged.client_name = _vote([r.client_name for r in bucket])
    merged.client_id = _vote([r.client_id for r in bucket])
    merged.technician = _vote([r.technician for r in bucket])
    merged.payment_status = _vote([r.payment_status for r in bucket])
    merged.amount_total = _vote([r.amount_total for r in bucket])

    # Union-merge the amounts dict
    amounts_by_key: Dict[str, List[float]] = defaultdict(list)
    for r in bucket:
        for k, v in r.amounts.items():
            amounts_by_key[k].append(v)
    merged.amounts = {k: _vote(v) for k, v in amounts_by_key.items() if _vote(v) is not None}

    dates_by_key: Dict[str, List[Any]] = defaultdict(list)
    for r in bucket:
        for k, v in r.other_dates.items():
            dates_by_key[k].append(v)
    merged.other_dates = {k: _vote(v) for k, v in dates_by_key.items() if _vote(v) is not None}

    extra_by_key: Dict[str, List[Any]] = defaultdict(list)
    for r in bucket:
        for k, v in r.extra.items():
            extra_by_key[k].append(v)
    merged.extra = {k: _vote(v) for k, v in extra_by_key.items() if _vote(v) is not None}

    # Concatenate sources so provenance is preserved
    for r in bucket:
        merged.sources.extend(r.sources)
    return merged


def _vote(values: List[Any]) -> Any:
    """Choose the best value from a list of OCR observations.

    Rules:
      - Drop None / empty strings.
      - If nothing left, return None.
      - Otherwise return the most common value; ties broken by first seen.
    """
    filtered = [v for v in values if v not in (None, "", {}, [])]
    if not filtered:
        return None
    try:
        counter = Counter(filtered)
    except TypeError:
        # unhashable (e.g. nested dicts) — just return the first non-null
        return filtered[0]
    most, count = counter.most_common(1)[0]
    return most
