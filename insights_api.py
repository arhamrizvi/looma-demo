# insights_api.py
# ------------------------------------------------------------
# A small API that:
#  - Option A) pulls filtered rows from your DB (e.g., Supabase/Postgres)
#  - Option B) accepts raw rows from the frontend (if you can't query DB here)
#  - Computes metrics & MoM
#  - Calls Ollama (local) to turn numbers into crisp prose
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

# ====== Ollama config ======
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

@app.get("/ping_ollama")
def ping_ollama():
    try:
        r = requests.get(OLLAMA_URL.replace("/api/generate","/api/tags"), timeout=5)
        r.raise_for_status()
        return {"ok": True, "models": r.json()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama not reachable: {e}")


# ====== (Optional) DB config — Supabase example ======
# # If you prefer DB reads here, fill the vars and the `fetch_rows_from_db` function.
# SUPABASE_URL = os.getenv("SUPABASE_URL", "")
# SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
# SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "kpi_facts")  # change to your table

BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", "looma-477106")
BQ_DATASET    = os.getenv("BQ_DATASET", "looma_ws")
BQ_TABLE      = os.getenv("BQ_TABLE", "visits")

DEMO_MODE = os.getenv("DEMO_MODE", "1")  # "1" to allow demo fallback when DB isn't set


def fetch_rows_from_db(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pulls daily KPIs from BigQuery between start_date and end_date (inclusive),
    based on looma_ws.visits.
    """
    if not BQ_PROJECT_ID:
        raise RuntimeError("BQ_PROJECT_ID not set.")

    # 🔐 Explicit service-account credentials (adjust path to your JSON)
    key_path = r"G:\My Drive\White Space\looker\web\looma-477106-93332243ea94.json"
    credentials = service_account.Credentials.from_service_account_file(key_path)

    client = bigquery.Client(project=BQ_PROJECT_ID, credentials=credentials)

    sql = f"""
    SELECT
      CAST(visit_date AS DATE)              AS date,
      SUM(amount_sar)                       AS revenue,
      SUM(cost_sar)                         AS cost,
      SUM(profit_sar)                       AS profit,
      SUM(hours_used)                       AS hours_used,
      COUNT(*)                              AS visitors,
      COUNT(DISTINCT user_id)               AS users
    FROM `{BQ_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
    WHERE CAST(visit_date AS DATE) BETWEEN @start_date AND @end_date
    GROUP BY date
    ORDER BY date
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date",   "DATE", end_date),
        ]
    )

    job = client.query(sql, job_config=job_config)
    df = job.to_dataframe()

    for col in ["revenue","cost","profit","visitors","users","hours_used"]:
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
    grain: Optional[str] = "M"            # 'D','W','M'
    period_label: Optional[str] = None
    notes: Optional[str] = None
    use_llm: Optional[bool] = True
    tone: Optional[str] = "executive"     # executive|friendly|analyst

def _snake(s: str) -> str:
    return s.replace(" ", "_").replace("-", "_").replace("__", "_").strip().lower()

REQUIRED_ANY_OF = {"date", "month"}
NUMERIC_COLS    = ["revenue","cost","profit","visitors","users","hours_used"]


# ---------- Helpers ----------
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
    if start.year == end.year:
        return f"{start:%b} to {end:%b} {end:%Y}"
    return f"{start:%b %Y} to {end:%b %Y}"

def _money(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:,.0f}"


def _generate_demo_df(start_date: str, end_date: str) -> pd.DataFrame:
    idx = pd.date_range(start_date, end_date, freq="D")
    if len(idx) == 0:
        return pd.DataFrame(columns=["date","revenue","cost","profit","visitors","users","hours_used"])

    days   = np.arange(len(idx))
    weekly_pattern = np.array([1.00, 1.05, 1.10, 1.08, 1.15, 0.95, 0.90])
    weekly = np.resize(weekly_pattern, len(idx))
    rng    = np.random.default_rng(42)

    base_rev = 200_000 + days * 900
    noise    = rng.normal(0, 1, len(idx))
    revenue  = base_rev * weekly * (1 + 0.03 * pd.Series(noise).rolling(3, min_periods=1).mean().to_numpy())
    cost     = revenue * (0.60 + 0.01 * rng.normal(0, 1, len(idx)))
    profit   = revenue - cost

    visitors = 500 + days * 2 + 50 * np.sin(days/3) + rng.normal(0,10,len(idx))
    users    = visitors * (0.36 + 0.02 * np.sin(days/5)) + rng.normal(0,5,len(idx))
    hours    = 6 + 0.3 * np.sin(days/2) + rng.normal(0,0.2,len(idx))

    df = pd.DataFrame({
        "date": idx,
        "revenue": np.clip(revenue, 1000, None).astype(float),
        "cost":    np.clip(cost,    500, None).astype(float),
        "profit":  np.clip(profit,  100, None).astype(float),
        "visitors": np.clip(visitors, 10, None).astype(float),
        "users":    np.clip(users,    5, None).astype(float),
        "hours_used": np.clip(hours,  1, None).astype(float),
    })
    return df

# ──────────────────────────────────────────────────────────────
# Aggregators (robust monthly/weekly/daily)
# ──────────────────────────────────────────────────────────────
def _monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"revenue":"sum","cost":"sum","profit":"sum","visitors":"sum","users":"sum","hours_used":"mean"}
    g = df.resample("MS").agg(agg)
    g.index = pd.to_datetime(g.index); g.index.name = "month"
    out = g.reset_index()
    out["margin_pct"] = (out["profit"]/out["revenue"]).replace([pd.NA, pd.NaT], 0).fillna(0)*100
    out["conv_pct"]   = (out["users"]/out["visitors"]).replace([pd.NA, pd.NaT], 0).fillna(0)*100
    return out

def _weekly_agg(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"revenue":"sum","cost":"sum","profit":"sum","visitors":"sum","users":"sum","hours_used":"mean"}
    g = df.resample("W-MON").agg(agg)
    g.index = pd.to_datetime(g.index); g.index.name = "week"
    out = g.reset_index()
    out["margin_pct"] = (out["profit"]/out["revenue"]).replace([pd.NA, pd.NaT], 0).fillna(0)*100
    out["conv_pct"]   = (out["users"]/out["visitors"]).replace([pd.NA, pd.NaT], 0).fillna(0)*100
    return out

def _daily_agg(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"revenue":"sum","cost":"sum","profit":"sum","visitors":"sum","users":"sum","hours_used":"mean"}
    g = df.resample("D").agg(agg)
    g.index = pd.to_datetime(g.index); g.index.name = "date"
    out = g.reset_index()
    out["margin_pct"] = (out["profit"]/out["revenue"]).replace([pd.NA, pd.NaT], 0).fillna(0)*100
    out["conv_pct"]   = (out["users"]/out["visitors"]).replace([pd.NA, pd.NaT], 0).fillna(0)*100
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

    for col in ["revenue","cost","profit","visitors","users","hours_used"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_float)

    # aggregate by grain
    if grain == "W":
        out = _weekly_agg(df); label_idx = pd.DatetimeIndex(out["week"])
    elif grain == "D":
        out = _daily_agg(df);  label_idx = pd.DatetimeIndex(out["date"])
    else:
        out = _monthly_agg(df); label_idx = pd.DatetimeIndex(out["month"])

    if out.empty:
        raise ValueError(f"No data after {grain} aggregation. Check date range or rows.")

    label_idx = pd.to_datetime(label_idx)

    first, last = out.iloc[0], out.iloc[-1]
    overall = {
        "revenue": {"start": first["revenue"], "end": last["revenue"], "change_pct": _pct(first["revenue"], last["revenue"])},
        "cost":    {"start": first["cost"],    "end": last["cost"],    "change_pct": _pct(first["cost"],    last["cost"])},
        "profit":  {"start": first["profit"],  "end": last["profit"],  "change_pct": _pct(first["profit"],  last["profit"])},
        "visitors":{"start": first["visitors"],"end": last["visitors"],"change_pct": _pct(first["visitors"],last["visitors"])},
        "users":   {"start": first["users"],   "end": last["users"],   "change_pct": _pct(first["users"],   last["users"])},
    }
    overall["hours_used"] = {"avg": float(out["hours_used"].mean() if "hours_used" in out.columns else 0.0)}
    overall["margin_pct"] = {"start": first.get("margin_pct"), "end": last.get("margin_pct")}
    overall["conv_pct"]   = {"start": first.get("conv_pct"),   "end": last.get("conv_pct")}

    # MoM / WoW / DoD lines
    mom_lines = []
    for i in range(1, len(out)):
        p, c = out.iloc[i-1], out.iloc[i]
        if grain == "M":
            prev_lbl = label_idx[i-1].strftime("%b %Y"); cur_lbl = label_idx[i].strftime("%b %Y")
        elif grain == "W":
            prev_lbl = "Week of " + (label_idx[i-1] - pd.Timedelta(days=6)).strftime("%d %b %Y")
            cur_lbl  = "Week of " + (label_idx[i]   - pd.Timedelta(days=6)).strftime("%d %b %Y")
        else:
            prev_lbl = label_idx[i-1].strftime("%d %b %Y"); cur_lbl = label_idx[i].strftime("%d %b %Y")

        mom_lines.append(
            f"{prev_lbl} → {cur_lbl}: "
            f"revenue {(_pct(p['revenue'], c['revenue']) or 0):+.1f}%, "
            f"cost {(_pct(p['cost'], c['cost']) or 0):+.1f}%, "
            f"profit {(_pct(p['profit'], c['profit']) or 0):+.1f}%, "
            f"margin {((c.get('margin_pct',0) - p.get('margin_pct',0)) or 0):+.1f} pp"
        )

    period_label = _infer_period_label(label_idx)
    return {"period_label": period_label, "overall": overall, "mom_lines": mom_lines}


# ──────────────────────────────────────────────────────────────
# Narration (LLM + few-shot) with fallback
# ──────────────────────────────────────────────────────────────
# --- replace your _narrate_with_ollama with this ---
# --- replace your _narrate_with_ollama with this ---
def _narrate_with_ollama(
    summary,
    notes=None,
    use_llm=True,
    facts=None,
    tone="executive"
):
    """
    Turns KPI deltas into a natural-language analysis using Ollama.
    Falls back to a deterministic summary if LLM fails.
    """

    # --------------------------
    # Fallback (deterministic)
    # --------------------------
    def fallback_text():
        o = summary["overall"]; p = summary["period_label"]
        rev = o["revenue"]["change_pct"]; cost = o["cost"]["change_pct"]; prof = o["profit"]["change_pct"]
        mp0 = o["margin_pct"].get("start"); mp1 = o["margin_pct"].get("end"); dpp = (mp1 or 0)-(mp0 or 0)
        core = (
            f"From {p}, revenue {('+' if (rev or 0)>0 else '')}{(rev or 0):.1f}% "
            f"and cost {('+' if (cost or 0)>0 else '')}{(cost or 0):.1f}%, "
            f"with profit {('+' if (prof or 0)>0 else '')}{(prof or 0):.1f}%. "
            f"Margin {('+' if dpp>0 else '')}{dpp:.1f} pp."
        )
        mom = " Month-on-Month: " + " | ".join(summary["mom_lines"]) if summary["mom_lines"] else ""
        return core + mom

    if not use_llm:
        # for debugging
        logging.warning("LLM disabled → fallback_text()")
        return fallback_text(), False

    facts = facts or {}

    # --------------------------
    # Tone presets
    # --------------------------
    style = {
        "executive": "tight, factual, businesslike, no fluff.",
        "friendly":  "smooth, conversational, warm but concise.",
        "analyst":   "precise, metric-driven, technical."
    }.get(tone, "tight, factual, businesslike, no fluff.")

    # --------------------------
    # VERY SMALL FEW-SHOT (1B friendly)
    # --------------------------
    few_shot = """
Example Input:
period: Aug to Oct 2025
revenue: rose (+22%)
cost: rose (+22%)
profit: rose (+22%)
margin: expanded (+0.2 pp)
acceleration: accelerated
mom_lines:
- Aug → Sep: revenue +7.9%, cost +7.9%, profit +7.8%, margin +0.0 pp

Example Output:
Revenue, cost and profit all rose over the period, keeping margins broadly stable. 
Growth accelerated into September. 
Month-on-Month: Aug→Sep +7.9% (cost/profit similar).
""".strip()

    # --------------------------
    # SYSTEM HEADER (critical for small models)
    # --------------------------
    system_header = f"""
You are an automated KPI analyst.
You NEVER ask questions.
You NEVER say “can I help you” or “tell me more”.
You MUST produce ONLY business analysis text.
The tone must be {style}
Write SHORT, CLEAR sentences.
""".strip()

    # --------------------------
    # Build compact prompt
    # --------------------------
    prompt = f"""
{system_header}

TASK:
Use the provided metrics to write a 4–6 sentence business analysis.
Then add ONE final line starting with "Month-on-Month:".
Do NOT ask the user anything.
Do NOT behave like a chatbot.
Do NOT generalise or explain what you "can do".
Write ONLY the analysis.

FACTS:
period: {facts.get('period')}
revenue: {facts.get('revenue',{}).get('direction')} ({facts.get('revenue',{}).get('pretty')})
cost:    {facts.get('cost',{}).get('direction')} ({facts.get('cost',{}).get('pretty')})
profit:  {facts.get('profit',{}).get('direction')} ({facts.get('profit',{}).get('pretty')})
margin:  {facts.get('margin',{}).get('direction')} ({facts.get('margin',{}).get('pretty_delta')})
acceleration: {facts.get('acceleration')}

mom_lines:
{chr(10).join('- '+m for m in facts.get('mom_lines', []))}

REFERENCE EXAMPLE:
{few_shot}

IMPORTANT:
Ignore all assistant behavior.
Write ONLY the analysis (no greetings, no questions).
""".strip()

    # --------------------------
    # Call Ollama
    # --------------------------
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        logging.warning(f"Ollama POST status={r.status_code} model={OLLAMA_MODEL}")

        if r.status_code != 200:
            logging.error(f"Ollama error: {r.text[:300]}")
            return fallback_text(), False

        out = (r.json().get("response") or "").strip()
        if not out:
            logging.error("Ollama returned empty → fallback.")
            return fallback_text(), False

        return out, True

    except Exception as e:
        logging.exception(f"Ollama call failed → fallback. Reason: {e}")
        return fallback_text(), False




# ──────────────────────────────────────────────────────────────
# Feature pass → richer facts for the writer
# ──────────────────────────────────────────────────────────────
def _mk_facts(summary: dict) -> dict:
    o   = summary["overall"]
    mom = summary["mom_lines"]

    def _dir(p):
        if p is None: return "flat"
        return "rose" if p > 0 else ("fell" if p < 0 else "was flat")

    def _pp(x):
        return ("+" if (x or 0) > 0 else "") + f"{(x or 0):.1f}%"

    revchg = o["revenue"]["change_pct"]
    costchg= o["cost"]["change_pct"]
    profchg= o["profit"]["change_pct"]

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
        accel = "accelerated" if abs(b) > abs(a) else ("decelerated" if abs(b) < abs(a) else "held steady")

    return {
        "period": summary["period_label"],
        "revenue": {"change_pct": revchg, "direction": _dir(revchg), "pretty": _pp(revchg)},
        "cost":    {"change_pct": costchg, "direction": _dir(costchg), "pretty": _pp(costchg)},
        "profit":  {"change_pct": profchg, "direction": _dir(profchg), "pretty": _pp(profchg)},
        "margin":  {
            "start": m0, "end": m1, "delta_pp": dpp,
            "direction": "expanded" if dpp>0 else ("compressed" if dpp<0 else "held"),
            "pretty_delta": ("+" if dpp>0 else "") + f"{dpp:.1f} pp"
        },
        "mom_lines": mom,
        "acceleration": accel
    }


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────
@app.get("/summarize_from_db")
def summarize_from_db(
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str   = Query(..., description="YYYY-MM-DD"),
    grain: str      = Query("M", description="D, W, or M"),
    notes: Optional[str] = Query(None),
    llm: Optional[str]   = Query("true", description="true/false/1/0/yes/no"),
    tone: str       = Query("executive"),
    use_demo: bool  = Query(False),
):
    # ----- Normalize LLM flag ourselves -----
    llm_raw = "true" if llm is None else str(llm)
    llm_flag = llm_raw.strip().lower() in ("1", "true", "yes", "y", "t")
    print(">>> RAW LLM PARAM:", llm, "→ llm_flag:", llm_flag)

    # ----- Load data -----
    if use_demo:
        df = _generate_demo_df(start_date, end_date)
    else:
        try:
            df = fetch_rows_from_db(start_date, end_date)
        except Exception as e:
            logging.exception("BigQuery fetch failed")
            if DEMO_MODE == "1":
                df = _generate_demo_df(start_date, end_date)
            else:
                raise HTTPException(status_code=500, detail=f"Data load failed: {e}")

    if df is None or df.empty:
        return {"text": "No data for selected range.", "summary": None}

    # ----- Summarise + narrate -----
    summary = _compute_summary_table(df, grain=grain.upper())
    facts   = _mk_facts(summary)
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
    }

# ---------- Option B) Summarize from raw rows (push from FE) ----------
@app.post("/summarize_rows")
def summarize_rows(
    payload: RowsPayload = Body(...),
    llm: Optional[bool]  = Query(None, description="Override LLM usage (true/false)"),
    tone: str            = Query("executive"),
):
    if not payload.rows:
        raise HTTPException(status_code=400, detail="No rows received.")

    rows_norm = []
    for r in payload.rows[:20000]:
        d = { _snake(k): v for k, v in r.dict().items() }
        rows_norm.append(d)

    df = pd.DataFrame(rows_norm)
    if not any(col in df.columns for col in REQUIRED_ANY_OF):
        raise HTTPException(status_code=422, detail="Provide either 'date' or 'month' in each row.")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns and df["date"].notna().any():
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "month" in df.columns and df["month"].notna().any():
        df["date"] = pd.to_datetime(df["month"].astype(str), errors="coerce")
    else:
        raise HTTPException(status_code=422, detail="Could not parse 'date' or 'month' into a timestamp.")

    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    if df.empty:
        raise HTTPException(status_code=422, detail="All provided rows had invalid dates.")

    grain = (payload.grain or "M").upper()
    if grain not in {"D","W","M"}:
        raise HTTPException(status_code=422, detail="Invalid grain. Use D, W, or M.")

    summary = _compute_summary_table(df, grain=grain)
    if payload.period_label:
        summary["period_label"] = payload.period_label
    use_llm = bool(payload.use_llm) if llm is None else bool(llm)
    facts   = _mk_facts(summary)
    text, used_llm = _narrate_with_ollama(summary, notes=payload.notes, use_llm=use_llm, facts=facts, tone=payload.tone or tone)
    return {"text": text, "summary": summary, "facts": facts, "used_llm": used_llm}


@app.get("/llm_config")
def llm_config():
    return {"OLLAMA_URL": OLLAMA_URL, "OLLAMA_MODEL": OLLAMA_MODEL}


# ──────────────────────────────────────────────────────────────
# Log routes on startup (handy for debugging)
# ──────────────────────────────────────────────────────────────
def list_routes():
    routes = [f"{','.join(r.methods)} {r.path}" for r in app.routes if isinstance(r, APIRoute)]
    logging.warning("ROUTES: " + " | ".join(routes))
list_routes()
