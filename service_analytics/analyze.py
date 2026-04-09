"""Analytics over deduplicated service rows.

Produces a JSON summary plus a handful of plots answering the questions
the user actually cares about:

  - How many services total?
  - How many unique clients?
  - Revenue totals (if amount columns were parsed)?
  - Peak months and peak days of week?
  - Technician workload breakdown?
  - Payment status breakdown?
  - Service type breakdown (inferred from which amount column is populated)?

Matplotlib is optional — if it's not installed, ``analyze`` still writes
the JSON summary and CSV exports, just no PNGs.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from service_analytics.schema import ServiceRow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze(
    rows: List[ServiceRow],
    out_dir: Path,
    *,
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Run analytics and write outputs to ``out_dir``.

    Writes:
      - ``services.csv``      — flattened row-level CSV
      - ``summary.json``      — all numeric summaries
      - ``services_by_month.png``    (if make_plots and matplotlib present)
      - ``services_by_dow.png``
      - ``revenue_by_month.png``
      - ``payment_status.png``

    Returns the summary dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(rows, out_dir / "services.csv")

    df = _build_dataframe(rows)
    summary = _compute_summary(df)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote summary to %s", summary_path)

    if make_plots:
        try:
            _make_plots(df, out_dir)
        except ImportError as e:
            logger.warning("Plots skipped: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.warning("Plot generation failed: %s", e)

    return summary


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def _write_csv(rows: List[ServiceRow], path: Path) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flats = [r.to_flat_row() for r in rows]
    headers: List[str] = []
    seen = set()
    for row in flats:
        for k in row.keys():
            if k not in seen:
                headers.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in flats:
            writer.writerow(row)
    logger.info("Wrote %d rows to %s", len(rows), path)


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------


def _build_dataframe(rows: List[ServiceRow]):
    """Build a pandas DataFrame. Falls back to a dict-of-lists shim if
    pandas is not installed (so the CSV + JSON still work)."""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed; analysis will be limited")
        return _DictFrame([r.to_flat_row() for r in rows])

    records = [r.to_flat_row() for r in rows]
    df = pd.DataFrame(records)
    if "service_date" in df.columns:
        df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    # coerce numeric columns
    for col in df.columns:
        if col.startswith("amount_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


class _DictFrame:
    """Tiny pandas-free fallback so analysis still runs without pandas."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# Summary computations
# ---------------------------------------------------------------------------


def _compute_summary(df) -> Dict[str, Any]:
    if isinstance(df, _DictFrame):
        return _compute_summary_fallback(df)

    import pandas as pd

    summary: Dict[str, Any] = {
        "total_services": int(len(df)),
        "unique_clients": 0,
        "date_range": {"start": None, "end": None},
        "revenue": {},
        "services_by_month": {},
        "services_by_day_of_week": {},
        "services_by_weekday_vs_weekend": {},
        "top_technicians": {},
        "payment_status_breakdown": {},
        "service_type_breakdown": {},
        "peak_month": None,
        "peak_day_of_week": None,
    }

    if len(df) == 0:
        return summary

    # unique clients
    if "client_id" in df.columns:
        ids = df["client_id"].replace("", pd.NA).dropna()
        if len(ids) > 0:
            summary["unique_clients"] = int(ids.nunique())
    if summary["unique_clients"] == 0 and "client_name" in df.columns:
        names = df["client_name"].replace("", pd.NA).dropna()
        summary["unique_clients"] = int(names.nunique())

    # date range
    if "service_date" in df.columns:
        dates = df["service_date"].dropna()
        if len(dates) > 0:
            summary["date_range"]["start"] = dates.min().date().isoformat()
            summary["date_range"]["end"] = dates.max().date().isoformat()

    # revenue
    amount_cols = [c for c in df.columns if c.startswith("amount_")]
    total_col = "amount_total" if "amount_total" in df.columns else None
    if total_col and df[total_col].notna().any():
        summary["revenue"]["grand_total"] = float(df[total_col].sum(skipna=True))
        summary["revenue"]["mean_per_service"] = float(df[total_col].mean(skipna=True))
    elif amount_cols:
        # sum per row across all amount columns
        per_row = df[amount_cols].sum(axis=1, skipna=True)
        summary["revenue"]["grand_total"] = float(per_row.sum())
        summary["revenue"]["mean_per_service"] = float(per_row.mean())
    summary["revenue"]["by_column"] = {
        c.replace("amount_", ""): float(df[c].sum(skipna=True))
        for c in amount_cols
        if df[c].notna().any()
    }

    # services by month
    if "service_date" in df.columns:
        dated = df.dropna(subset=["service_date"])
        if len(dated) > 0:
            months = dated["service_date"].dt.strftime("%Y-%m")
            m_counts = months.value_counts().sort_index()
            summary["services_by_month"] = m_counts.to_dict()
            summary["peak_month"] = m_counts.idxmax()

            dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dow = dated["service_date"].dt.dayofweek.map(lambda i: dow_names[i])
            dow_counts = dow.value_counts().reindex(dow_names, fill_value=0)
            summary["services_by_day_of_week"] = dow_counts.to_dict()
            summary["peak_day_of_week"] = dow_counts.idxmax()

            weekend = dow.isin(["Sat", "Sun"])
            summary["services_by_weekday_vs_weekend"] = {
                "weekday": int((~weekend).sum()),
                "weekend": int(weekend.sum()),
            }

            # revenue per month
            if total_col and df[total_col].notna().any():
                rev_by_month = dated.groupby(months)[total_col].sum().to_dict()
                summary["revenue"]["by_month"] = {
                    k: float(v) for k, v in rev_by_month.items()
                }

    # top technicians
    if "technician" in df.columns:
        techs = df["technician"].replace("", pd.NA).dropna()
        if len(techs) > 0:
            summary["top_technicians"] = techs.value_counts().head(10).to_dict()

    # payment status
    if "payment_status" in df.columns:
        ps = df["payment_status"].replace("", pd.NA).dropna()
        if len(ps) > 0:
            summary["payment_status_breakdown"] = ps.value_counts().to_dict()

    # service type breakdown — count how often each amount column is populated
    if amount_cols:
        counts = {
            c.replace("amount_", ""): int(df[c].notna().sum())
            for c in amount_cols
            if c != "amount_total"
        }
        counts = {k: v for k, v in counts.items() if v > 0}
        summary["service_type_breakdown"] = dict(
            sorted(counts.items(), key=lambda x: x[1], reverse=True)
        )

    return summary


def _compute_summary_fallback(df: _DictFrame) -> Dict[str, Any]:
    """Minimal stats without pandas."""
    recs = df.records
    dates = [r.get("service_date") for r in recs if r.get("service_date")]
    client_ids = [r.get("client_id") for r in recs if r.get("client_id")]
    techs = [r.get("technician") for r in recs if r.get("technician")]
    ps = [r.get("payment_status") for r in recs if r.get("payment_status")]
    return {
        "total_services": len(recs),
        "unique_clients": len(set(client_ids)),
        "date_range": {
            "start": min(dates) if dates else None,
            "end": max(dates) if dates else None,
        },
        "top_technicians": dict(Counter(techs).most_common(10)),
        "payment_status_breakdown": dict(Counter(ps)),
        "note": "pandas not installed; install it for full analysis",
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _make_plots(df, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    if isinstance(df, _DictFrame) or len(df) == 0:
        return

    if "service_date" in df.columns:
        dated = df.dropna(subset=["service_date"])
        if len(dated) > 0:
            # services by month
            months = dated["service_date"].dt.strftime("%Y-%m")
            m_counts = months.value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(9, 4))
            m_counts.plot.bar(ax=ax, color="#3b82f6")
            ax.set_title("Services by month")
            ax.set_ylabel("Services")
            ax.set_xlabel("Month")
            fig.tight_layout()
            fig.savefig(out_dir / "services_by_month.png", dpi=130)
            plt.close(fig)

            # services by day of week
            dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dow = dated["service_date"].dt.dayofweek.map(lambda i: dow_names[i])
            dow_counts = dow.value_counts().reindex(dow_names, fill_value=0)
            fig, ax = plt.subplots(figsize=(7, 4))
            dow_counts.plot.bar(ax=ax, color="#10b981")
            ax.set_title("Services by day of week")
            ax.set_ylabel("Services")
            fig.tight_layout()
            fig.savefig(out_dir / "services_by_dow.png", dpi=130)
            plt.close(fig)

            # revenue by month (if any total)
            total_col = "amount_total" if "amount_total" in df.columns else None
            if total_col and df[total_col].notna().any():
                rev = dated.groupby(months)[total_col].sum()
                fig, ax = plt.subplots(figsize=(9, 4))
                rev.plot.bar(ax=ax, color="#f59e0b")
                ax.set_title("Revenue by month")
                ax.set_ylabel("Revenue ($)")
                ax.set_xlabel("Month")
                fig.tight_layout()
                fig.savefig(out_dir / "revenue_by_month.png", dpi=130)
                plt.close(fig)

    if "payment_status" in df.columns:
        ps = df["payment_status"].replace("", pd.NA).dropna()
        if len(ps) > 0:
            counts = ps.value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax.set_title("Payment status")
            fig.tight_layout()
            fig.savefig(out_dir / "payment_status.png", dpi=130)
            plt.close(fig)

    logger.info("Plots written to %s", out_dir)
