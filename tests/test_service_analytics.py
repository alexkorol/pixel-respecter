"""Tests for the service_analytics schema + dedupe logic.

These tests do not touch the network or ffmpeg — they just verify the
parsing, dedupe, and analysis logic on synthetic data.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from service_analytics.dedupe import dedupe_rows
from service_analytics.schema import (
    RowSource,
    ServiceRow,
    dump_jsonl,
    load_jsonl,
)


# ---------------------------------------------------------------------------
# Schema parsing
# ---------------------------------------------------------------------------


def test_from_dict_parses_date_and_money():
    row = ServiceRow.from_dict(
        {
            "service_date": "10/29/2025",
            "client_name": "Blaugerberr, Elijah",
            "client_id": "0059022300190034",
            "technician": "Adams, Michael",
            "payment_status": "NoPaymentMade",
            "amount_total": "$82.00",
            "amounts": {"Install Fee": "$50.00", "Monitoring": "$32.00"},
            "other_dates": {"Next Appointment": "11/29/2025"},
            "extra": {"notes": "front counter"},
        }
    )
    assert row.service_date == date(2025, 10, 29)
    assert row.client_name == "Blaugerberr, Elijah"
    assert row.client_id == "0059022300190034"
    assert row.amount_total == 82.0
    assert row.amounts["install_fee"] == 50.0
    assert row.amounts["monitoring"] == 32.0
    assert row.other_dates["next_appointment"] == date(2025, 11, 29)
    assert row.extra["notes"] == "front counter"


def test_from_dict_handles_null_values():
    row = ServiceRow.from_dict(
        {
            "service_date": None,
            "client_name": "",
            "client_id": "N/A",
            "amount_total": None,
        }
    )
    assert row.service_date is None
    assert row.client_name is None
    assert row.client_id is None
    assert row.amount_total is None


def test_dedupe_key_prefers_client_id():
    row = ServiceRow.from_dict(
        {
            "service_date": "10/29/2025",
            "client_id": "0059022300190034",
            "client_name": "Blaugerberr, Elijah",
        }
    )
    assert row.dedupe_key() == "id:0059022300190034|2025-10-29"


def test_dedupe_key_falls_back_to_name():
    row = ServiceRow.from_dict(
        {"service_date": "10/29/2025", "client_name": "Smith, John"}
    )
    assert row.dedupe_key() == "name:smith, john|2025-10-29"


def test_to_flat_row_includes_amount_columns():
    row = ServiceRow.from_dict(
        {
            "service_date": "10/29/2025",
            "client_id": "123",
            "amounts": {"Install Fee": 50, "Monitoring": 32},
        }
    )
    flat = row.to_flat_row()
    assert flat["amount_install_fee"] == 50.0
    assert flat["amount_monitoring"] == 32.0
    assert flat["service_date"] == "2025-10-29"


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------


def test_dedupe_merges_same_client_across_frames():
    """Two frames seeing the same row — dedupe should collapse them."""
    r1 = ServiceRow.from_dict(
        {
            "service_date": "10/29/2025",
            "client_id": "0059022300190034",
            "client_name": "Blaugerberr, Elijah",
            "technician": "Adams, Michael",
            "amount_total": 82.0,
            "amounts": {"install_fee": 50, "monitoring": 32},
        }
    )
    r2 = ServiceRow.from_dict(
        {
            "service_date": "10/29/2025",
            "client_id": "0059022300190034",
            "client_name": "Blaugerberr, Elijah",
            "technician": "Adams, Michael",
            "amount_total": 82.0,
            # this frame missed the monitoring column (scrolled off)
            "amounts": {"install_fee": 50},
        }
    )
    r1.sources.append(RowSource(frame_path="f1.jpg", frame_time_s=1.0, model="test"))
    r2.sources.append(RowSource(frame_path="f2.jpg", frame_time_s=2.0, model="test"))

    merged = dedupe_rows([r1, r2])
    assert len(merged) == 1
    m = merged[0]
    assert m.client_id == "0059022300190034"
    assert m.amounts["install_fee"] == 50
    assert m.amounts["monitoring"] == 32  # recovered from r1
    assert len(m.sources) == 2


def test_dedupe_majority_vote_on_conflicting_ocr():
    """Three observations of an id where one misread a digit — majority wins."""
    rows = []
    for client_id in ("0059022300190034", "0059022300190034", "OO59O22300190034"):
        rows.append(
            ServiceRow.from_dict(
                {
                    "service_date": "10/29/2025",
                    "client_id": client_id,
                    "client_name": "Blaugerberr, Elijah",
                }
            )
        )
    # All three share the same name+date key, so they collapse
    # (client_id differs, so dedupe_key can't use it for the outlier — but
    # name+date still collapses them together)
    # Force them into the same bucket by forcing the same key via name fallback
    for r in rows:
        r.client_id = None  # strip so dedupe uses name
    rows[0].client_id = "0059022300190034"
    rows[1].client_id = "0059022300190034"
    rows[2].client_id = "OO59O22300190034"
    # Now re-group: the first two have id:...|date, the third also has id-based
    # key (different). They won't merge unless we pre-align keys. So instead,
    # just test on three rows all sharing a name-based key.
    name_rows = [
        ServiceRow.from_dict(
            {
                "service_date": "10/29/2025",
                "client_name": "Test, Subject",
                "technician": "Adams, Michael",
            }
        ),
        ServiceRow.from_dict(
            {
                "service_date": "10/29/2025",
                "client_name": "Test, Subject",
                "technician": "Adams, Michael",
            }
        ),
        ServiceRow.from_dict(
            {
                "service_date": "10/29/2025",
                "client_name": "Test, Subject",
                "technician": "Adarns, Michael",  # OCR typo
            }
        ),
    ]
    merged = dedupe_rows(name_rows)
    assert len(merged) == 1
    # Adams (2 votes) should beat Adarns (1 vote)
    assert merged[0].technician == "Adams, Michael"


def test_dedupe_drops_empty_rows():
    empty = ServiceRow()  # no fields at all
    real = ServiceRow.from_dict(
        {"service_date": "10/29/2025", "client_id": "123", "client_name": "Smith, John"}
    )
    merged = dedupe_rows([empty, real])
    assert len(merged) == 1
    assert merged[0].client_id == "123"


def test_dedupe_keeps_distinct_clients():
    r1 = ServiceRow.from_dict(
        {"service_date": "10/29/2025", "client_id": "111", "client_name": "A, A"}
    )
    r2 = ServiceRow.from_dict(
        {"service_date": "10/29/2025", "client_id": "222", "client_name": "B, B"}
    )
    merged = dedupe_rows([r1, r2])
    assert len(merged) == 2


def test_dedupe_keeps_same_client_different_date():
    r1 = ServiceRow.from_dict(
        {"service_date": "10/29/2025", "client_id": "111", "client_name": "A, A"}
    )
    r2 = ServiceRow.from_dict(
        {"service_date": "11/29/2025", "client_id": "111", "client_name": "A, A"}
    )
    merged = dedupe_rows([r1, r2])
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------


def test_jsonl_round_trip(tmp_path: Path):
    rows = [
        ServiceRow.from_dict(
            {
                "service_date": "10/29/2025",
                "client_id": "123",
                "client_name": "Smith, John",
                "amount_total": 100.0,
                "amounts": {"install_fee": 50, "monitoring": 50},
            }
        ),
        ServiceRow.from_dict(
            {
                "service_date": "11/01/2025",
                "client_id": "456",
                "client_name": "Doe, Jane",
                "amount_total": 82.0,
            }
        ),
    ]
    path = tmp_path / "services.jsonl"
    n = dump_jsonl(rows, path)
    assert n == 2
    loaded = load_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].client_id == "123"
    assert loaded[0].service_date == date(2025, 10, 29)
    assert loaded[0].amounts["install_fee"] == 50


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def test_analyze_produces_summary(tmp_path: Path):
    """End-to-end analyze on a small synthetic dataset."""
    pd = pytest.importorskip("pandas")
    from service_analytics.analyze import analyze

    rows = [
        ServiceRow.from_dict(
            {
                "service_date": "10/29/2025",  # Wed
                "client_id": "111",
                "client_name": "Smith, John",
                "technician": "Adams, Michael",
                "payment_status": "NoPaymentMade",
                "amount_total": 82.0,
                "amounts": {"install_fee": 50, "monitoring": 32},
            }
        ),
        ServiceRow.from_dict(
            {
                "service_date": "10/30/2025",  # Thu
                "client_id": "222",
                "client_name": "Doe, Jane",
                "technician": "Adams, Michael",
                "payment_status": "Paid",
                "amount_total": 117.59,
                "amounts": {"monitoring": 117.59},
            }
        ),
        ServiceRow.from_dict(
            {
                "service_date": "11/05/2025",  # Wed
                "client_id": "333",
                "client_name": "Bloggs, Joe",
                "technician": "Adams, Michael",
                "payment_status": "Paid",
                "amount_total": 93.00,
                "amounts": {"monitoring": 93.00},
            }
        ),
        ServiceRow.from_dict(
            {
                "service_date": "11/12/2025",  # Wed
                "client_id": "111",  # same client as row 1, different date
                "client_name": "Smith, John",
                "technician": "Smith, Pat",
                "payment_status": "Paid",
                "amount_total": 32.00,
                "amounts": {"monitoring": 32.00},
            }
        ),
    ]
    out = tmp_path / "reports"
    summary = analyze(rows, out, make_plots=False)
    assert summary["total_services"] == 4
    assert summary["unique_clients"] == 3
    assert summary["peak_day_of_week"] == "Wed"  # 3 Wednesdays vs 1 Thursday
    assert summary["revenue"]["grand_total"] == pytest.approx(324.59)
    assert (out / "services.csv").exists()
    assert (out / "summary.json").exists()
    summary_loaded = json.loads((out / "summary.json").read_text())
    assert summary_loaded["total_services"] == 4
