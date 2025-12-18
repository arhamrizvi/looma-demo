# insights_api.py
# ------------------------------------------------------------
# A small API that:
#  - Option A) pulls filtered rows from your DB (e.g., BigQuery)
#  - Option B) accepts raw rows from the frontend (if you can't query DB here)
#  - Computes metrics & MoM/DoD/WoW
#  - Calls Groq (OpenAI-compatible) to turn numbers into crisp prose
# ------------------------------------------------------------
import os
import math
import requests
import pandas as pd
import numpy as np
import random
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Query, Body, HTTPException
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
import logging

from google.cloud import bigquery
from google.oauth2 import service_account

# ====== Groq LLM config ======
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

app = FastAPI(title="Looma Insights API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://your-netlify-site.netlify.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/llm_debug")
def llm_debug():
    return {
        "has_groq_key": bool(GROQ_API_KEY),
        "groq_model": GROQ_MODEL,
    }


# ====== DB config — BigQuery ======
BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", "looma-477106")
BQ_DATASET = os.getenv("BQ_DATASET", "looma_ws")
BQ_TABLE = os.getenv("BQ_TABLE", "fact_visits")

DEMO_MODE = os.getenv("DEMO_MODE", "1")  # "1" to allow demo fallback when DB isn't set


def _make_bq_client() -> bigquery.Client:
    """
    Use local JSON key when running on your laptop.
    Use Cloud Run's service account when running in the cloud.
    """
    # Cloud Run sets K_SERVICE env var
    if os.getenv("K_SERVICE"):
        # Running on Cloud Run → use default service account (no key file)
        logging.warning("Using default Cloud Run credentials for BigQuery")
        return bigquery.Client(project=BQ_PROJECT_ID)
    else:
        # Local dev → use your JSON key file
        key_path = r"G:\My Drive\White Space\looker\web\not git\looma-477106-dffc8e700305.json"
        logging.warning(f"Using local service account key for BigQuery: {key_path}")
        credentials = service_account.Credentials.from_service_account_file(key_path)
        return bigquery.Client(project=BQ_PROJECT_ID, credentials=credentials)


def fetch_rows_from_db(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pulls daily KPIs from BigQuery between start_date and end_date (inclusive),
    based on looma_ws.visits.
    """
    if not BQ_PROJECT_ID:
        raise RuntimeError("BQ_PROJECT_ID not set.")

    client = _make_bq_client()

    sql = f"""
    SELECT
  CAST(visit_date AS DATE) AS date,
  ANY_VALUE(holiday_name)  AS holiday_name,
  ANY_VALUE(is_weekend)    AS is_weekend,
  SUM(amount_sar)          AS revenue,
  SUM(cost_sar)            AS cost,
  SUM(profit_sar)          AS profit,
  SUM(hours_used)          AS hours_used,
  COUNT(*)                 AS visitors,
  COUNT(DISTINCT user_id)  AS users
FROM `{BQ_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
WHERE CAST(visit_date AS DATE) BETWEEN @start_date AND @end_date
GROUP BY date
ORDER BY date

    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )

    job = client.query(sql, job_config=job_config)
    df = job.to_dataframe()

    for col in ["revenue", "cost", "profit", "visitors", "users", "hours_used"]:
        if col not in df.columns:
            df[col] = None

    return df


# ──────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────
class Row(BaseModel):
    month: Optional[str] = None
    date: Optional[str] = None
    week_day: Optional[str] = None
    week_day_name: Optional[str] = None
    week_number: Optional[int] = None
    revenue: float
    cost: float
    profit: float
    hours_used: Optional[float] = None
    visitors: Optional[float] = None
    users: Optional[float] = None


class RowsPayload(BaseModel):
    rows: List[Row]
    grain: Optional[str] = "M"  # 'D','W','M'
    period_label: Optional[str] = None
    notes: Optional[str] = None
    use_llm: Optional[bool] = True
    tone: Optional[str] = "executive"  # executive|friendly|analyst


def _snake(s: str) -> str:
    return s.replace(" ", "_").replace("-", "_").replace("__", "_").strip().lower()


REQUIRED_ANY_OF = {"date", "month"}
NUMERIC_COLS = ["revenue", "cost", "profit", "visitors", "users", "hours_used"]


# ──────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────
def _pct(a: float, b: float) -> Optional[float]:
    try:
        if a is None or b is None or a == 0:
            return None
        return (b - a) / a * 100.0
    except Exception:
        return None


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _infer_period_label(idx: pd.DatetimeIndex) -> str:
    if len(idx) == 0:
        return "No period"

    start, end = idx.min(), idx.max()
    delta_days = (end.date() - start.date()).days

    # Same exact date
    if start.date() == end.date():
        return f"{start:%d %b %Y}"

    # Short daily range (<=7 days) within same year → show explicit dates
    if delta_days <= 7 and start.year == end.year:
        return f"{start:%d %b} to {end:%d %b %Y}"

    # Same year → monthly format
    if start.year == end.year:
        return f"{start:%b} to {end:%b} {end:%Y}"

    # Different years → month + year
    return f"{start:%b %Y} to {end:%b %Y}"


def _money(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:,.0f}"


def _generate_demo_df(start_date: str, end_date: str) -> pd.DataFrame:
    idx = pd.date_range(start_date, end_date, freq="D")
    if len(idx) == 0:
        return pd.DataFrame(
            columns=["date", "revenue", "cost", "profit", "visitors", "users", "hours_used"]
        )

    days = np.arange(len(idx))
    weekly_pattern = np.array([1.00, 1.05, 1.10, 1.08, 1.15, 0.95, 0.90])
    weekly = np.resize(weekly_pattern, len(idx))
    rng = np.random.default_rng(42)

    base_rev = 200_000 + days * 900
    noise = rng.normal(0, 1, len(idx))
    revenue = (
        base_rev
        * weekly
        * (1 + 0.03 * pd.Series(noise).rolling(3, min_periods=1).mean().to_numpy())
    )
    cost = revenue * (0.60 + 0.01 * rng.normal(0, 1, len(idx)))
    profit = revenue - cost

    visitors = 500 + days * 2 + 50 * np.sin(days / 3) + rng.normal(0, 10, len(idx))
    users = visitors * (0.36 + 0.02 * np.sin(days / 5)) + rng.normal(0, 5, len(idx))
    hours = 6 + 0.3 * np.sin(days / 2) + rng.normal(0, 0.2, len(idx))

    df = pd.DataFrame(
        {
            "date": idx,
            "revenue": np.clip(revenue, 1000, None).astype(float),
            "cost": np.clip(cost, 500, None).astype(float),
            "profit": np.clip(profit, 100, None).astype(float),
            "visitors": np.clip(visitors, 10, None).astype(float),
            "users": np.clip(users, 5, None).astype(float),
            "hours_used": np.clip(hours, 1, None).astype(float),
        }
    )
    return df


# ──────────────────────────────────────────────────────────────
# Aggregators (monthly / weekly / daily)
# ──────────────────────────────────────────────────────────────
def _monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "revenue": "sum",
        "cost": "sum",
        "profit": "sum",
        "visitors": "sum",
        "users": "sum",
        "hours_used": "mean",
    }
    g = df.resample("MS").agg(agg)
    g.index = pd.to_datetime(g.index)
    g.index.name = "month"
    out = g.reset_index()
    out["margin_pct"] = (
        (out["profit"] / out["revenue"]).replace([pd.NA, pd.NaT], 0).fillna(0) * 100
    )
    out["conv_pct"] = (
        (out["users"] / out["visitors"]).replace([pd.NA, pd.NaT], 0).fillna(0) * 100
    )
    return out


def _weekly_agg(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "revenue": "sum",
        "cost": "sum",
        "profit": "sum",
        "visitors": "sum",
        "users": "sum",
        "hours_used": "mean",
    }
    g = df.resample("W-MON").agg(agg)
    g.index = pd.to_datetime(g.index)
    g.index.name = "week"
    out = g.reset_index()
    out["margin_pct"] = (
        (out["profit"] / out["revenue"]).replace([pd.NA, pd.NaT], 0).fillna(0) * 100
    )
    out["conv_pct"] = (
        (out["users"] / out["visitors"]).replace([pd.NA, pd.NaT], 0).fillna(0) * 100
    )
    return out


def _daily_agg(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "revenue": "sum",
        "cost": "sum",
        "profit": "sum",
        "visitors": "sum",
        "users": "sum",
        "hours_used": "mean",
    }
    g = df.resample("D").agg(agg)
    g.index = pd.to_datetime(g.index)
    g.index.name = "date"
    out = g.reset_index()
    out["margin_pct"] = (
        (out["profit"] / out["revenue"]).replace([pd.NA, pd.NaT], 0).fillna(0) * 100
    )
    out["conv_pct"] = (
        (out["users"] / out["visitors"]).replace([pd.NA, pd.NaT], 0).fillna(0) * 100
    )
    return out


# ──────────────────────────────────────────────────────────────
# Summarization core (math only)
# ──────────────────────────────────────────────────────────────
def _compute_summary_table(df: pd.DataFrame, grain: str) -> Dict[str, Any]:
    df = df.copy()

    # normalize input -> index by date
    if "date" in df.columns and df["date"].notna().any():
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "month" in df.columns and df["month"].notna().any():
        df["date"] = pd.to_datetime(df["month"].astype(str), errors="coerce")
    else:
        raise ValueError("Provide either 'date' or 'month' in rows.")

    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    for col in ["revenue", "cost", "profit", "visitors", "users", "hours_used"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_float)

    # aggregate by grain
    if grain == "W":
        out = _weekly_agg(df)
        label_idx = pd.DatetimeIndex(out["week"])
    elif grain == "D":
        out = _daily_agg(df)
        label_idx = pd.DatetimeIndex(out["date"])
    else:
        out = _monthly_agg(df)
        label_idx = pd.DatetimeIndex(out["month"])

    if out.empty:
        raise ValueError(f"No data after {grain} aggregation. Check date range or rows.")

    label_idx = pd.to_datetime(label_idx)

    first, last = out.iloc[0], out.iloc[-1]
    overall = {
        "revenue": {
            "start": first["revenue"],
            "end": last["revenue"],
            "change_pct": _pct(first["revenue"], last["revenue"]),
        },
        "cost": {
            "start": first["cost"],
            "end": last["cost"],
            "change_pct": _pct(first["cost"], last["cost"]),
        },
        "profit": {
            "start": first["profit"],
            "end": last["profit"],
            "change_pct": _pct(first["profit"], last["profit"]),
        },
        "visitors": {
            "start": first["visitors"],
            "end": last["visitors"],
            "change_pct": _pct(first["visitors"], last["visitors"]),
        },
        "users": {
            "start": first["users"],
            "end": last["users"],
            "change_pct": _pct(first["users"], last["users"]),
        },
    }
    overall["hours_used"] = {
        "avg": float(out["hours_used"].mean() if "hours_used" in out.columns else 0.0)
    }
    overall["margin_pct"] = {
        "start": first.get("margin_pct"),
        "end": last.get("margin_pct"),
    }
    overall["conv_pct"] = {
        "start": first.get("conv_pct"),
        "end": last.get("conv_pct"),
    }

    # MoM / WoW / DoD lines
    mom_lines = []
    for i in range(1, len(out)):
        p, c = out.iloc[i - 1], out.iloc[i]
        if grain == "M":
            prev_lbl = label_idx[i - 1].strftime("%b %Y")
            cur_lbl = label_idx[i].strftime("%b %Y")
        elif grain == "W":
            prev_lbl = "Week of " + (label_idx[i - 1] - pd.Timedelta(days=6)).strftime(
                "%d %b %Y"
            )
            cur_lbl = "Week of " + (label_idx[i] - pd.Timedelta(days=6)).strftime(
                "%d %b %Y"
            )
        else:
            prev_lbl = label_idx[i - 1].strftime("%d %b %Y")
            cur_lbl = label_idx[i].strftime("%d %b %Y")

        mom_lines.append(
            f"{prev_lbl} → {cur_lbl}: "
            f"revenue {(_pct(p['revenue'], c['revenue']) or 0):+.1f}%, "
            f"cost {(_pct(p['cost'], c['cost']) or 0):+.1f}%, "
            f"profit {(_pct(p['profit'], c['profit']) or 0):+.1f}%, "
            f"margin {((c.get('margin_pct', 0) - p.get('margin_pct', 0)) or 0):+.1f} pp"
        )

    period_label = _infer_period_label(label_idx)
    return {
        "period_label": period_label,
        "overall": overall,
        "mom_lines": mom_lines,
        "grain": grain,  # 👈 used by the narrator to switch Day-on-day / Week-on-week / MoM
    }


# ──────────────────────────────────────────────────────────────
# Narration (LLM + few-shot) with fallback
# ──────────────────────────────────────────────────────────────
def _narrate_with_ollama(
    summary,
    notes: str | None = None,
    use_llm: bool = True,
    facts: dict | None = None,
    tone: str = "executive",
):
    """
    Uses Groq (OpenAI-compatible chat API) to turn KPI deltas into an
    executive-style narrative. Falls back to a deterministic text
    summary if LLM is disabled, not configured, or fails.

    Inputs:
      summary: dict from _compute_summary_table(...)
      notes:   optional free-text context (events, promos, one-offs)
      use_llm: True → call Groq, False → deterministic fallback only
      facts:   dict from _mk_facts(summary)
      tone:    "executive" | "analyst" | "friendly"
    """

    # --------------------------
    # Fallback (deterministic)
    # --------------------------
    def fallback_text() -> str:
        o = summary["overall"]
        p = summary["period_label"]

        rev = o["revenue"]["change_pct"]
        cost = o["cost"]["change_pct"]
        prof = o["profit"]["change_pct"]
        mp0 = summary["overall"]["margin_pct"].get("start")
        mp1 = summary["overall"]["margin_pct"].get("end")
        dpp = (mp1 or 0) - (mp0 or 0)

        core = (
            f"From {p}, revenue {('+' if (rev or 0) > 0 else '')}{(rev or 0):.1f}% "
            f"and cost {('+' if (cost or 0) > 0 else '')}{(cost or 0):.1f}%, "
            f"with profit {('+' if (prof or 0) > 0 else '')}{(prof or 0):.1f}%. "
            f"Margin {('+' if dpp > 0 else '')}{dpp:.1f} pp."
        )
        mom = (
            " Month-on-Month: " + " | ".join(summary["mom_lines"])
            if summary.get("mom_lines")
            else ""
        )
        return core + mom

    # --------------------------
    # Short-circuit: no LLM
    # --------------------------
    if not use_llm:
        logging.warning("[_narrate_with_ollama] LLM disabled → fallback_text()")
        return fallback_text(), False

    if not GROQ_API_KEY:
        logging.error("[_narrate_with_ollama] GROQ_API_KEY not set → fallback_text()")
        return fallback_text(), False

    facts = facts or {}
    notes = (notes or "").strip()

    # --------------------------
    # Tone presets
    # --------------------------
    if tone == "executive":
        style = (
            "professional, board-ready, analytical, concise, focused on trends, risks, and next steps."
        )
    elif tone == "analyst":
        style = (
            "technical, metric-heavy, precise, focused on explaining movements and variance drivers."
        )
    elif tone == "friendly":
        style = "clear, conversational, but still data-driven and to-the-point."
    else:
        style = "professional, concise, data-driven."

    # --------------------------
    # Grain hint + comparison label
    # --------------------------
    grain = None
    if isinstance(summary, dict):
        grain = summary.get("grain")

    if grain == "D":
        grain_hint = "Focus on day-on-day movement between consecutive dates."
        comparison_label = "Day-on-day"
    elif grain == "W":
        grain_hint = "Focus on week-on-week movement between consecutive weeks."
        comparison_label = "Week-on-week"
    elif grain == "M":
        grain_hint = "Focus on overall period and month-on-month trends."
        comparison_label = "Month-on-Month"
    else:
        grain_hint = "Focus on overall trends and the provided period comparisons."
        comparison_label = "Month-on-Month"

    # --------------------------
    # Few-shot example for style
    # --------------------------
    few_shot = """
Example Input:
period: Aug–Oct 2025
revenue: rose strongly (+22%)
cost: rose in line (+22%)
profit: increased (+22%)
margin: expanded slightly (+0.2 pp)
acceleration: accelerated
mom_lines:
- Aug → Sep: revenue +7.9%, cost +7.9%, profit +7.8%, margin +0.0 pp
NOTES (optional external context from user, may be empty):
- "Early-September marketing push in Riyadh"
- "One-off corporate booking in late October"

Example Output (good style):
Across Aug–Oct 2025, topline trends were positive: revenue grew 22% with costs tracking in line, keeping profit up 22% and margins slightly higher (+0.2 pp). Growth momentum accelerated into September, supported by early-September marketing activity and a one-off corporate booking that lifted October run-rate. Cost growth has remained controlled relative to revenue, so profitability is improving rather than diluting. Month-on-Month: Aug→Sep revenue +7.9% with profit moving in tandem and a stable margin profile.
""".strip()

    # --------------------------
    # System message
    # --------------------------
    system_msg = f"""
You are an automated KPI insights generator for leadership dashboards.

You write like a senior business analyst presenting to C-level stakeholders:
- {style}
- Always anchor statements in the actual numbers and period provided.
- Always mention the overall period (dates or labels), and highlight direction and magnitude of change.
- Use clear business language (no hype, no emojis, no chatty filler).
- You NEVER ask questions or say things like "How can I help you".
- You NEVER mention that you are an AI model or that you were given data.
- You NEVER invent events or campaigns that are not explicitly mentioned in NOTES.
- When the period covers only a small range (e.g. two consecutive days), phrase it explicitly with both dates,
  such as "Across 1 Sep 2025 to 2 Sep 2025" instead of "Across Sep to Sep 2025".
- Use the appropriate comparison label:
  - "Day-on-day:" for daily grain
  - "Week-on-week:" for weekly grain
  - "Month-on-Month:" for monthly grain.
- If a significant performance drop (>10%) coincides with a named holiday
(e.g. Saudi National Day), explicitly mention it as a likely contributing factor,
but phrase causality carefully (e.g. “likely driven by”, “consistent with”).

If NOTES are provided, you may reference them as potential context or drivers
(e.g. "supported by X"), but do not state them as facts if the causal impact is uncertain.
""".strip()

    # --------------------------
    # Pull structured facts
    # --------------------------
    period = facts.get("period")
    rev = facts.get("revenue", {})
    cost = facts.get("cost", {})
    prof = facts.get("profit", {})
    marg = facts.get("margin", {})
    accel = facts.get("acceleration")
    mom_lines = facts.get("mom_lines", []) or []

    notes_block = "None provided."
    if notes:
        notes_block = notes

    # --------------------------
    # User message
    # --------------------------
    user_msg = f"""
{grain_hint}

Write a short executive-style narrative (4–7 sentences) using the FACTS and NOTES below.

Structure:
1) One sentence summarising the overall performance for the period (direction and intensity of change).
2) One–two sentences on revenue, cost and profit, explicitly referencing approximate percentage changes and margin movement.
3) One sentence on momentum or trend (e.g. accelerating, stabilising, slowing), using the acceleration signal.
4) One sentence that starts with "{comparison_label}:" and summarises the comparison based on the provided lines
   (for daily use the day-on-day change, for weekly week-on-week, for monthly month-on-month).
5) Optionally, one sentence that references relevant NOTES as potential drivers or context (only if NOTES are provided).

Constraints:
- Do NOT ask the reader any questions.
- Do NOT explain what you are going to do.
- Do NOT repeat the word "FACTS" or list bullet points.
- Do NOT invent new events; you may only reference items that appear in NOTES.
- Keep the tone suitable for a CEO/ExCo review deck.

FACTS:
- holidays (date → name): {facts.get('holidays', [])}
- period: {period}
- revenue: direction={rev.get('direction')}, change={rev.get('pretty')}
- cost:    direction={cost.get('direction')}, change={cost.get('pretty')}
- profit:  direction={prof.get('direction')}, change={prof.get('pretty')}
- margin:  direction={marg.get('direction')}, delta={marg.get('pretty_delta')}
- acceleration signal: {accel}
- Comparison summary lines:
{chr(10).join('  • ' + m for m in mom_lines)}

NOTES (optional context provided by the user; may highlight campaigns, calendar effects, or one-off events):
{notes_block}

REFERENCE EXAMPLE (for style and depth, not for exact wording):
{few_shot}
""".strip()

    # --------------------------
    # Call Groq Chat Completions
    # --------------------------
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.35,  # keep it tight & analytical
            "max_tokens": 400,
        }

        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        logging.warning(f"[Groq] POST status={r.status_code} model={GROQ_MODEL}")

        if r.status_code != 200:
            logging.error(f"[Groq] error body: {r.text[:400]}")
            return fallback_text(), False

        data = r.json()
        out = (data["choices"][0]["message"]["content"] or "").strip()
        if not out:
            logging.error("[Groq] returned empty response → fallback.")
            return fallback_text(), False

        return out, True

    except Exception as e:
        logging.exception(f"[Groq] call failed → fallback. Reason: {e}")
        return fallback_text(), False


# ──────────────────────────────────────────────────────────────
# Feature pass → richer facts for the writer
# ──────────────────────────────────────────────────────────────
def _mk_facts(summary: dict, df: pd.DataFrame | None = None) -> dict:

    o = summary["overall"]
    mom = summary["mom_lines"]

    holidays = []
    if df is not None and "holiday_name" in df.columns:
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        holidays = (
            tmp[["date", "holiday_name"]]
            .dropna(subset=["date", "holiday_name"])
            .drop_duplicates()
            .sort_values("date")
            .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
            .to_dict("records")
        )


    def _dir(p):
        if p is None:
            return "flat"
        return "rose" if p > 0 else ("fell" if p < 0 else "was flat")

    def _pp(x):
        return ("+" if (x or 0) > 0 else "") + f"{(x or 0):.1f}%"

    revchg = o["revenue"]["change_pct"]
    costchg = o["cost"]["change_pct"]
    profchg = o["profit"]["change_pct"]

    m0 = o["margin_pct"].get("start")
    m1 = o["margin_pct"].get("end")
    dpp = (m1 or 0) - (m0 or 0)

    accel = None
    if len(mom) >= 2:
        import re

        def ext(line):
            m = re.search(r"revenue ([+\-]?\d+\.\d)%", line)
            return float(m.group(1)) if m else 0.0

        a, b = ext(mom[-2]), ext(mom[-1])
        accel = (
            "accelerated"
            if abs(b) > abs(a)
            else ("decelerated" if abs(b) < abs(a) else "held steady")
        )

    return {
        "period": summary["period_label"],
        "revenue": {
            "change_pct": revchg,
            "direction": _dir(revchg),
            "pretty": _pp(revchg),
        },
        "cost": {
            "change_pct": costchg,
            "direction": _dir(costchg),
            "pretty": _pp(costchg),
        },
        "profit": {
            "change_pct": profchg,
            "direction": _dir(profchg),
            "pretty": _pp(profchg),
        },
        "margin": {
            "start": m0,
            "end": m1,
            "delta_pp": dpp,
            "direction": "expanded"
            if dpp > 0
            else ("compressed" if dpp < 0 else "held"),
            "pretty_delta": ("+" if dpp > 0 else "") + f"{dpp:.1f} pp",
        },
        "mom_lines": mom,
        "acceleration": accel,
        "holidays": holidays,
    }


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────
@app.get("/summarize_from_db")
def summarize_from_db(
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    grain: str = Query("M", description="D, W, or M"),
    notes: Optional[str] = Query(None),
    llm: Optional[str] = Query("true", description="true/false/1/0/yes/no"),
    tone: str = Query("executive"),
    use_demo: bool = Query(False),
):
    # ----- Normalize LLM flag ourselves -----
    llm_raw = "true" if llm is None else str(llm)
    llm_flag = llm_raw.strip().lower() in ("1", "true", "yes", "y", "t")
    logging.warning(f"[summarize_from_db] llm_raw={llm_raw!r} -> llm_flag={llm_flag}")

    # ----- Load data -----
    try:
        if use_demo:
            df = _generate_demo_df(start_date, end_date)
        else:
            df = fetch_rows_from_db(start_date, end_date)
    except Exception as e:
        logging.exception("[summarize_from_db] Data load failed")
        raise HTTPException(status_code=500, detail=f"Data load failed: {e}")

    if df is None or df.empty:
        return {
            "text": "No data for selected range.",
            "summary": None,
            "facts": None,
            "used_llm": False,
            "llm_flag": llm_flag,
            "debug": {"stage": "no_data"},
        }

    # ----- Summarise + narrate -----
    summary = _compute_summary_table(df, grain=grain.upper())
    facts = _mk_facts(summary, df=df)

    text, used_llm = _narrate_with_ollama(
        summary,
        notes=notes,
        use_llm=llm_flag,
        facts=facts,
        tone=tone,
    )

    return {
        "text": text,
        "summary": summary,
        "facts": facts,
        "used_llm": used_llm,
        "llm_flag": llm_flag,
        "debug": {
            "has_groq_key": bool(GROQ_API_KEY),
            "groq_model": GROQ_MODEL,
        },
    }


@app.get("/groq_smoketest")
def groq_smoketest():
    """Minimal Groq test from Cloud Run."""
    if not GROQ_API_KEY:
        return {
            "ok": False,
            "reason": "GROQ_API_KEY not set in env",
        }

    system_msg = "You are a very short KPI assistant."
    user_msg = "Summarise: revenue up 10%, cost up 8%, profit up 15%, margin up 2 pp."

    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": 80,
        }

        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20,
        )
        logging.warning(f"[groq_smoketest] status={r.status_code}")

        if r.status_code != 200:
            return {
                "ok": False,
                "status": r.status_code,
                "body": r.text[:400],
            }

        data = r.json()
        content = (data["choices"][0]["message"]["content"] or "").strip()

        return {
            "ok": True,
            "status": r.status_code,
            "model": GROQ_MODEL,
            "sample": content,
        }

    except Exception as e:
        logging.exception("[groq_smoketest] Exception calling Groq")
        return {
            "ok": False,
            "exception": str(e),
        }


# ---------- Option B) Summarize from raw rows (push from FE) ----------
@app.post("/summarize_rows")
def summarize_rows(
    payload: RowsPayload = Body(...),
    llm: Optional[bool] = Query(None, description="Override LLM usage (true/false)"),
    tone: str = Query("executive"),
):
    if not payload.rows:
        raise HTTPException(status_code=400, detail="No rows received.")

    rows_norm = []
    for r in payload.rows[:20000]:
        d = {_snake(k): v for k, v in r.dict().items()}
        rows_norm.append(d)

    df = pd.DataFrame(rows_norm)
    if not any(col in df.columns for col in REQUIRED_ANY_OF):
        raise HTTPException(
            status_code=422, detail="Provide either 'date' or 'month' in each row."
        )

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns and df["date"].notna().any():
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "month" in df.columns and df["month"].notna().any():
        df["date"] = pd.to_datetime(df["month"].astype(str), errors="coerce")
    else:
        raise HTTPException(
            status_code=422,
            detail="Could not parse 'date' or 'month' into a timestamp.",
        )

    df = (
        df.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    if df.empty:
        raise HTTPException(
            status_code=422, detail="All provided rows had invalid dates."
        )

    grain = (payload.grain or "M").upper()
    if grain not in {"D", "W", "M"}:
        raise HTTPException(status_code=422, detail="Invalid grain. Use D, W, or M.")

    summary = _compute_summary_table(df, grain=grain)
    if payload.period_label:
        summary["period_label"] = payload.period_label
    use_llm_effective = bool(payload.use_llm) if llm is None else bool(llm)
    
    facts = _mk_facts(summary, df=df)

    text, used_llm = _narrate_with_ollama(
        summary,
        notes=payload.notes,
        use_llm=use_llm_effective,
        facts=facts,
        tone=payload.tone or tone,
    )
    return {"text": text, "summary": summary, "facts": facts, "used_llm": used_llm}


@app.get("/llm_config")
def llm_config():
    """
    Simple config endpoint the frontend can hit to see if LLM is available.
    """
    return {
        "provider": "groq",
        "has_key": bool(GROQ_API_KEY),
        "model": GROQ_MODEL,
    }


# ──────────────────────────────────────────────────────────────
# Log routes on startup (handy for debugging)
# ──────────────────────────────────────────────────────────────
def list_routes():
    routes = [
        f"{','.join(r.methods)} {r.path}"
        for r in app.routes
        if isinstance(r, APIRoute)
    ]
    logging.warning("ROUTES: " + " | ".join(routes))


list_routes()
