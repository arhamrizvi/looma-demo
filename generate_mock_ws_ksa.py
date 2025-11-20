#!/usr/bin/env python3
"""
Generate White Space (KSA) mock data with trends/seasonality + user cohorts.

Tables produced (CSV by default):
- calendar_dim.csv
- users.csv
- daily_capacity.csv
- visits.csv

Two modes:
- backfill: build Aug 1, 2025 -> today (Asia/Riyadh) from scratch
- daily:    generate exactly one day (default: today) and APPEND to existing CSVs

Optional: upload CSVs into BigQuery (if credentials available).

Usage examples:
    # Full backfill to ./out
    python generate_mock_ws_ksa.py --mode backfill --out ./out

    # Append today's data only
    python generate_mock_ws_ksa.py --mode daily --out ./out

    # Append a specific date
    python generate_mock_ws_ksa.py --mode daily --run-date 2025-11-10 --out ./out

    # Backfill and write to BigQuery
    python generate_mock_ws_ksa.py --mode backfill --out ./out \
        --bq-project YOUR_GCP_PROJECT --bq-dataset looma_ws
"""

import argparse
import math
import os
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

# --------- Configuration (adjust as you like) ---------
TZ = timezone(timedelta(hours=3))  # Asia/Riyadh UTC+3 (no DST)
START_DATE_DEFAULT = date(2025, 8, 1)

LOCATIONS = [
    "Riyadh - Olaya",
    "Riyadh - Digital City",
    "Jeddah - Corniche",
    "Dammam - Waterfront",
]

# Location multipliers (Olaya biggest)
LOC_FACTOR = {
    "Riyadh - Olaya": 1.20,
    "Riyadh - Digital City": 1.10,
    "Jeddah - Corniche": 1.00,
    "Dammam - Waterfront": 0.95,
}

# Pricing per plan
PLAN_RATES = {"Flex": 15.0, "Standard": 12.0, "Pro": 10.0, "Enterprise": 8.0}
PLAN_DURATIONS = {"Flex": 1.6, "Standard": 2.9, "Pro": 3.9, "Enterprise": 4.6}

# ------------------------------------------------------


def today_ksa() -> date:
    return datetime.now(TZ).date()


def rng_for_day(d: date) -> np.random.RandomState:
    # Deterministic daily RNG (so re-runs are stable)
    return np.random.RandomState(int(d.strftime("%Y%m%d")))


def month_sin_idx(dt: pd.Timestamp) -> float:
    # Slight Oct/Nov lift, Aug dip
    return 1.0 + 0.15 * math.sin(2 * math.pi * (dt.month / 12.0) + 0.9)


def weekday_idx_ksa(dt: pd.Timestamp) -> float:
    # pandas: Monday=0 .. Sunday=6
    # KSA: Sun-Thu high, Fri lowest, Sat low-medium
    dow = dt.weekday()  # 0=Mon .. 6=Sun
    mapping = {
        6: 1.12,  # Sun
        0: 1.14,  # Mon
        1: 1.14,  # Tue
        2: 1.10,  # Wed
        3: 1.00,  # Thu
        4: 0.80,  # Fri
        5: 0.90,  # Sat
    }
    return mapping[dow]


def build_calendar(start_date: date, end_date: date) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df = pd.DataFrame({"date": dates})  # datetime64
    df["dom"] = df["date"].dt.day
    df["moy"] = df["date"].dt.month
    df["week_of_month"] = 1 + (((df["dom"] - 1) / 30.0) * 4).astype(int)

    # Seasonality + trend
    df["month_sin_idx"] = df["date"].apply(month_sin_idx)
    df["weekday_idx"] = df["date"].apply(weekday_idx_ksa)
    df["payday_idx"] = np.where((df["dom"].between(28, 31)) | (df["dom"] <= 5), 1.08, 1.00)

    # Trend: +10% from start -> Nov 30, 2025 (clip at 1.10 max)
    trend_end = date(2025, 11, 30)
    span = max((trend_end - start_date).days, 1)
    df["trend_idx"] = 1.0 + 0.10 * np.clip(
        (df["date"] - pd.Timestamp(start_date)).dt.days / span, 0, 1
    )

    # Holiday(s)
    df["holiday_name"] = None
    df.loc[df["date"] == pd.Timestamp("2025-09-23"), "holiday_name"] = "Saudi National Day"
    df["holiday_idx"] = np.where(df["holiday_name"].notna(), 0.78, 1.00)

    # Promos: first Monday + 20th
    first_mondays = []
    for _, group in df.groupby([df["date"].dt.year, df["date"].dt.month]):
        month_start = group["date"].min()
        week = pd.date_range(start=month_start, periods=7, freq="D")
        fm = [d for d in week if d.weekday() == 0]  # Monday
        if fm:
            first_mondays.append(fm[0].normalize())

    df["promo_flag"] = df["date"].isin(first_mondays) | (df["dom"] == 20)
    df["promo_idx"] = np.where(df["promo_flag"], 1.18, 1.00)

    # KSA weekend flags (Fri/Sat)  <-- FIXED (no parentheses)
    df["is_weekend"] = df["date"].dt.weekday.isin([4, 5])

    # Convert to plain date only at the very end
    df["date"] = df["date"].dt.date
    return df[
        [
            "date",
            "is_weekend",
            "dom",
            "moy",
            "week_of_month",
            "month_sin_idx",
            "weekday_idx",
            "payday_idx",
            "trend_idx",
            "holiday_name",
            "holiday_idx",
            "promo_flag",
            "promo_idx",
        ]
    ]



def build_users(n_users: int = 700, start_date: date = START_DATE_DEFAULT) -> pd.DataFrame:
    # Deterministic, but rich variation
    rs = np.random.RandomState(42)
    uids = np.arange(1, n_users + 1)
    df = pd.DataFrame({"u": uids})
    df["user_id"] = df["u"].apply(lambda x: f"U{x:06d}")
    df["full_name"] = df["u"].apply(lambda x: f"User {x}")
    df["email"] = df["u"].apply(lambda x: f"user{x}@example.com")

    # Signup within Aug–Oct window
    signup_offsets = rs.randint(0, 92, size=n_users)  # aug1 + 0..91 days
    df["signup_date"] = [start_date + timedelta(days=int(o)) for o in signup_offsets]
    df["cohort_month"] = pd.to_datetime(df["signup_date"]).dt.to_period("M").dt.to_timestamp().dt.date

    plans = np.array(["Flex", "Standard", "Pro", "Enterprise"])
    df["plan_type"] = plans[rs.randint(0, 4, size=n_users)]

    # Weighted home locations (Riyadh-heavy)
    home_choices = (
        ["Riyadh - Olaya"] * 3
        + ["Riyadh - Digital City"] * 2
        + ["Jeddah - Corniche"] * 3
        + ["Dammam - Waterfront"] * 2
    )
    df["home_location"] = rs.choice(home_choices, size=n_users)

    acq = np.array(["Organic", "Referral", "Paid", "Partnership"])
    df["acquisition_channel"] = acq[rs.randint(0, 4, size=n_users)]

    sizes = np.array(["Solo", "2-10", "11-50", "51-200", "200+"])
    roles = np.array(["Founder", "Analyst", "Engineer", "Designer", "Sales", "Other"])
    prefs = np.array(["Morning", "Midday", "Afternoon"])

    df["company_size"] = sizes[rs.randint(0, len(sizes), size=n_users)]
    df["role_bucket"] = roles[rs.randint(0, len(roles), size=n_users)]
    df["visit_pref"] = prefs[rs.randint(0, len(prefs), size=n_users)]

    # Propensities
    df["price_sensitivity"] = rs.randint(0, 90, size=n_users) / 100.0
    df["churn_risk"] = np.minimum(1.0, 0.15 + rs.randint(0, 40, size=n_users) / 100.0)
    base_activity = 0.50 + ((df["plan_type"].isin(["Pro", "Enterprise"])).astype(float) * 0.20)
    df["activity_bias"] = np.maximum(0.05, base_activity - (rs.randint(0, 20, size=n_users) / 100.0))
    df["is_active"] = True

    return df[
        [
            "user_id",
            "full_name",
            "email",
            "signup_date",
            "plan_type",
            "home_location",
            "acquisition_channel",
            "cohort_month",
            "company_size",
            "role_bucket",
            "visit_pref",
            "churn_risk",
            "price_sensitivity",
            "activity_bias",
            "is_active",
        ]
    ]


def gen_capacity_for_date(run_date: date) -> pd.DataFrame:
    rs = rng_for_day(run_date)
    rows = []
    for loc in LOCATIONS:
        base_total = 90 + (abs(hash("cap" + loc)) % 31)
        base_hours = 11 + (abs(hash("hrs" + loc)) % 3)
        desks_total = int(base_total + round(3 * rs.rand()))
        desks_open = int((base_total - 12) + round(6 * rs.rand()))
        hours_open = int(base_hours + round(1 * rs.rand()))
        rows.append(
            {
                "date": run_date,
                "location": loc,
                "desks_total": desks_total,
                "desks_open": desks_open,
                "hours_open": hours_open,
            }
        )
    return pd.DataFrame(rows)


def gen_capacity(calendar_df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []
    for d in calendar_df["date"]:
        all_rows.append(gen_capacity_for_date(d))
    return pd.concat(all_rows, ignore_index=True)


def gaussian_noise(rs: np.random.RandomState, sigma: float) -> float:
    # Box–Muller
    u1 = max(rs.rand(), 1e-9)
    u2 = rs.rand()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return z * sigma


def expected_lambda(row, loc: str) -> float:
    # base ~ 60 scaled by seasonality, trend, holiday/promos and location factor
    lam = (
        60.0
        * row["month_sin_idx"]
        * row["weekday_idx"]
        * row["payday_idx"]
        * row["trend_idx"]
        * row["holiday_idx"]
        * row["promo_idx"]
        * LOC_FACTOR[loc]
    )
    return lam


def pick_users_for_visits(
    rs: np.random.RandomState, users_df: pd.DataFrame, loc: str, run_date: date, n: int
) -> np.ndarray:
    # Only users who have signed up by run_date and active
    pool = users_df[(users_df["signup_date"] <= run_date) & (users_df["is_active"])].copy()
    if pool.empty or n <= 0:
        return np.array([], dtype=object)

    # Weight by activity_bias and home_location affinity
    weights = pool["activity_bias"].values * np.where(pool["home_location"] == loc, 1.0, 0.6)
    # Avoid zero weights
    weights = np.maximum(weights, 0.001)
    weights = weights / weights.sum()
    idx = rs.choice(pool.index.values, size=n, replace=True, p=weights)
    return pool.loc[idx, "user_id"].values


def gen_visits_for_date(
    run_date: date,
    calendar_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    users_df: pd.DataFrame,
) -> pd.DataFrame:
    # Find calendar row
    crow = calendar_df.loc[calendar_df["date"] == run_date].iloc[0]
    rs = rng_for_day(run_date)

    rows = []
    for loc in LOCATIONS:
        lam_raw = expected_lambda(crow, loc)

        cap_open = capacity_df[(capacity_df["date"] == run_date) & (capacity_df["location"] == loc)]["desks_open"]
        cap_open = int(cap_open.iloc[0]) if not cap_open.empty else 80
        lam = min(lam_raw, cap_open * 1.9)

        # Convert to int visits with small Gaussian jitter
        v = int(max(0, round(lam + gaussian_noise(rs, sigma=4.0))))

        # Sample users
        user_ids = pick_users_for_visits(rs, users_df, loc, run_date, v)

        # For each visit: time, duration, revenue/cost/profit
        for i, uid in enumerate(user_ids, start=1):
            plan = users_df.loc[users_df["user_id"] == uid, "plan_type"].iloc[0]
            pref = users_df.loc[users_df["user_id"] == uid, "visit_pref"].iloc[0]
            rate = PLAN_RATES.get(plan, 12.0)
            base_hours = PLAN_DURATIONS.get(plan, 2.6)

            # Hour influenced by preference
            if pref == "Morning":
                in_hour = 8 + (abs(hash(f"{uid}:{run_date}:m")) % 3)  # 8..10
            elif pref == "Midday":
                in_hour = 11 + (abs(hash(f"{uid}:{run_date}:d")) % 3)  # 11..13
            else:
                in_hour = 14 + (abs(hash(f"{uid}:{run_date}:a")) % 3)  # 14..16

            minute = abs(hash(f"min:{uid}:{run_date}:{i}")) % 60

            check_in = datetime(run_date.year, run_date.month, run_date.day, in_hour, minute, tzinfo=TZ).replace(tzinfo=None)
            hours_used = max(0.8, round(base_hours + gaussian_noise(rs, sigma=0.4), 2))
            check_out = check_in + timedelta(minutes=int(round(hours_used * 60)))

            amount = round(hours_used * rate, 2)
            cost = round(amount * 0.6, 2)
            profit = round(amount * 0.4, 2)

            visit_id = f"V{run_date:%Y-%m-%d}-{abs(hash(uid + loc + str(run_date))) % 1_000_000:06d}"

            rows.append(
                {
                    "visit_id": visit_id,
                    "user_id": uid,
                    "location": loc,
                    "plan_type": plan,
                    "check_in_dt": check_in,
                    "check_out_dt": check_out,
                    "hours_used": hours_used,
                    "amount_sar": amount,
                    "cost_sar": cost,
                    "profit_sar": profit,
                }
            )

    return pd.DataFrame(rows)


def gen_visits(calendar_df: pd.DataFrame, capacity_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []
    for d in calendar_df["date"]:
        all_rows.append(gen_visits_for_date(d, calendar_df, capacity_df, users_df))
    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame(
        columns=[
            "visit_id",
            "user_id",
            "location",
            "plan_type",
            "check_in_dt",
            "check_out_dt",
            "hours_used",
            "amount_sar",
            "cost_sar",
            "profit_sar",
        ]
    )


def to_bigquery_if_requested(df_map, bq_project, bq_dataset):
    """
    Optional push to BigQuery (Standard SQL). Requires:
      pip install pandas-gbq google-cloud-bigquery
      GOOGLE_APPLICATION_CREDENTIALS set (or environment auth)
    """
    if not bq_project or not bq_dataset:
        return
    try:
        from pandas_gbq import to_gbq  # type: ignore
    except Exception as e:
        print("pandas-gbq not installed; skipping BigQuery upload. Error:", e)
        return

    if bq_project and bq_dataset:
        for name, df in df_map.items():
            table_id = f"{bq_dataset}.{name}"
            print(f"Uploading {name} to BigQuery table {table_id} ...")
            if name in {"visits", "daily_capacity"}:
                # replace for full backfill; if you want append on daily, adjust caller
                if df is not None and len(df) > 0:
                    to_gbq(df, table_id=table_id, project_id=bq_project, if_exists="replace")
            else:
                to_gbq(df, table_id=table_id, project_id=bq_project, if_exists="replace")


def save_csvs(out_dir: Path, df_map, mode: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in df_map.items():
        path = out_dir / f"{name}.csv"
        if mode == "daily" and path.exists() and name in {"visits", "daily_capacity"}:
            # append
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)
        print(f"Wrote {path}")


def run_backfill(out_dir: Path, bq_project: str, bq_dataset: str):
    start = START_DATE_DEFAULT
    end = today_ksa()
    cal = build_calendar(start, end)
    users = build_users(n_users=700, start_date=start)
    capacity = gen_capacity(cal)
    visits = gen_visits(cal, capacity, users)

    df_map = {
        "calendar_dim": cal,
        "users": users,
        "daily_capacity": capacity,
        "visits": visits,
    }
    save_csvs(out_dir, df_map, mode="backfill")
    to_bigquery_if_requested(df_map, bq_project, bq_dataset)


def run_daily(run_date: date, out_dir: Path, bq_project: str, bq_dataset: str):
    # Read existing users (or build if absent)
    users_path = out_dir / "users.csv"
    if users_path.exists():
        users = pd.read_csv(users_path, parse_dates=["signup_date", "cohort_month"])
        # keep dates as date objects
        users["signup_date"] = users["signup_date"].dt.date
        users["cohort_month"] = users["cohort_month"].dt.date
    else:
        users = build_users(n_users=700, start_date=START_DATE_DEFAULT)

    # Build 1-day calendar row
    cal = build_calendar(run_date, run_date)

    # Daily capacity + visits for one day
    capacity = gen_capacity_for_date(run_date)
    visits = gen_visits_for_date(run_date, cal, capacity, users)

    # Save/append
    # calendar_dim: update/append (idempotent-ish; for simplicity append)
    cal_path = out_dir / "calendar_dim.csv"
    if cal_path.exists():
        cal.to_csv(cal_path, mode="a", header=False, index=False)
    else:
        cal.to_csv(cal_path, index=False)

    save_csvs(out_dir, {"users": users, "daily_capacity": capacity, "visits": visits}, mode="daily")

    # Optional: append to BQ (you can change to 'append' with pandas-gbq if you like)
    to_bigquery_if_requested(
        {"calendar_dim": cal, "users": users, "daily_capacity": capacity, "visits": visits},
        bq_project,
        bq_dataset,
    )


def main():
    p = argparse.ArgumentParser(description="Generate KSA mock data for White Space")
    p.add_argument("--mode", choices=["backfill", "daily"], required=True)
    p.add_argument("--run-date", type=str, help="YYYY-MM-DD (only used in --mode daily). Default=today (Riyadh)")
    p.add_argument("--out", type=str, default="./out", help="Output directory for CSVs")
    p.add_argument("--bq-project", type=str, default="", help="(Optional) GCP project id to upload tables")
    p.add_argument("--bq-dataset", type=str, default="", help="(Optional) BigQuery dataset (e.g., looma_ws)")
    args = p.parse_args()

    out_dir = Path(args.out)

    if args.mode == "backfill":
        run_backfill(out_dir, args.bq_project, args.bq_dataset)
    else:
        rd = date.fromisoformat(args.run_date) if args.run_date else today_ksa()
        run_daily(rd, out_dir, args.bq_project, args.bq_dataset)


if __name__ == "__main__":
    main()