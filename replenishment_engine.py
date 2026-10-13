import os
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text


# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "atm_demand_model.joblib"
OUTPUT_PATH = "replenishment_recommendations.csv"
SUMMARY_PATH = "replenishment_branch_summary.csv"

# Policy assumptions (make these configurable for experiments).
BUFFER_DAYS = 3
MIN_CASH_LEVEL_PERCENT = 25.0
CRITICAL_CASH_LEVEL_PERCENT = 15.0

FEATURES = [
    "atm_id",
    "day_of_week",
    "is_weekend",
    "day_of_month",
    "month",
    "is_payday_period",
    "demand_lag_1",
    "demand_lag_7",
    "avg_demand_previous_7_days",
]


def require_columns(frame, required, table_name):
    """Fail early with a useful error if the schema/data is unexpected."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{table_name} is missing required columns: {', '.join(missing)}"
        )


def main():
    # --------------------------------------------------------
    # Load model and database connection
    # --------------------------------------------------------
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is missing. Set it in your VS Code terminal first."
        )
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train_model.py first."
        )

    model = joblib.load(MODEL_PATH)
    engine = create_engine(database_url)

    atm_query = text("""
        SELECT DISTINCT ON (d.atm_id)
            d.atm_id,
            a.atm_code,
            a.location_name,
            a.capacity,
            i.current_cash,
            a.status,
            a.branch_id,
            b.branch_name,
            b.available_cash AS branch_available_cash,
            d.demand_date,
            d.day_of_week,
            d.is_weekend,
            d.day_of_month,
            d.month,
            d.is_payday_period,
            d.demand_lag_1,
            d.demand_lag_7,
            d.avg_demand_previous_7_days
        FROM atm_ml_dataset AS d
        JOIN atm AS a ON a.atm_id = d.atm_id
        JOIN atm_cash_inventory AS i ON i.atm_id = a.atm_id
        JOIN branch AS b ON b.branch_id = a.branch_id
        WHERE d.demand_date < CURRENT_DATE
        ORDER BY d.atm_id, d.demand_date DESC
    """)

    neighbor_query = text("""
        SELECT atm_id, nearby_atm_id
        FROM nearby_atm
    """)

    try:
        with engine.connect() as connection:
            df = pd.read_sql(atm_query, connection)
            links = pd.read_sql(neighbor_query, connection)
    finally:
        engine.dispose()

    if df.empty:
        raise RuntimeError("No ATM records found for replenishment planning.")

    require_columns(
        df,
        FEATURES + [
            "atm_code", "location_name", "capacity", "current_cash", "status",
            "branch_id", "branch_name", "branch_available_cash", "demand_date",
        ],
        "ATM demand query",
    )
    require_columns(links, ["atm_id", "nearby_atm_id"], "nearby_atm query")

    # --------------------------------------------------------
    # Validate and normalize data
    # --------------------------------------------------------
    numeric_columns = [
        "capacity", "current_cash", "branch_available_cash",
        "demand_lag_1", "demand_lag_7", "avg_demand_previous_7_days",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[numeric_columns].isna().any().any():
        bad_columns = df[numeric_columns].columns[
            df[numeric_columns].isna().any()
        ].tolist()
        raise RuntimeError(
            "Null/non-numeric values found in required numeric fields: "
            + ", ".join(bad_columns)
        )

    df["capacity"] = df["capacity"].clip(lower=0)
    df["current_cash"] = df["current_cash"].clip(lower=0)
    df["branch_available_cash"] = df["branch_available_cash"].clip(lower=0)
    df["status_normalized"] = df["status"].astype(str).str.strip().str.upper()

    # Inventory exceeding capacity is flagged for visibility; we don't silently
    # invent extra capacity or allocate additional cash to that ATM.
    df["inventory_over_capacity"] = df["current_cash"] > df["capacity"]
    df["cash_level_percent"] = np.where(
        df["capacity"] > 0,
        df["current_cash"] / df["capacity"] * 100,
        0.0,
    )

    # --------------------------------------------------------
    # Predict next-day demand
    # --------------------------------------------------------
    df["predicted_demand"] = np.asarray(model.predict(df[FEATURES]), dtype=float)
    df["predicted_demand"] = (
        pd.Series(df["predicted_demand"], index=df.index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0)
    )

    # We have one next-day demand forecast. Estimate the cash remaining after
    # that day's withdrawals, then target BUFFER_DAYS of forecast demand.
    df["projected_cash_after_one_day"] = (
        df["current_cash"] - df["predicted_demand"]
    ).clip(lower=0)

    df["demand_buffer_target"] = (
        df["predicted_demand"] * BUFFER_DAYS
    ).clip(upper=df["capacity"])

    # A low-cash ATM is given a minimum target of 25% capacity. This is a
    # policy trigger, not an ML prediction.
    df["minimum_cash_target"] = (
        df["capacity"] * MIN_CASH_LEVEL_PERCENT / 100.0
    )

    # Request enough to reach the higher of the demand target and the minimum
    # cash target, after projected one-day withdrawals.
    df["target_cash"] = df[
        ["demand_buffer_target", "minimum_cash_target"]
    ].max(axis=1)

    df["requested_refill"] = (
        df["target_cash"] - df["projected_cash_after_one_day"]
    ).clip(lower=0)

    # Never request more than the ATM's remaining physical capacity.
    df["capacity_remaining"] = (
        df["capacity"] - df["current_cash"]
    ).clip(lower=0)

    df["requested_refill"] = np.minimum(
        df["requested_refill"], df["capacity_remaining"]
    )

    # Inactive/maintenance ATMs must not receive a refill recommendation.
    df.loc[
        df["status_normalized"] != "ACTIVE", "requested_refill"
    ] = 0.0

    # Explain why a request is or isn't generated.
    df["recommendation_reason"] = "Demand buffer / minimum cash target"
    df.loc[
        df["status_normalized"] != "ACTIVE", "recommendation_reason"
    ] = "Not active"
    df.loc[
        (df["status_normalized"] == "ACTIVE")
        & (df["capacity_remaining"] <= 0),
        "recommendation_reason",
    ] = "At capacity"
    df.loc[
        (df["status_normalized"] == "ACTIVE")
        & (df["capacity_remaining"] > 0)
        & (df["requested_refill"] <= 0),
        "recommendation_reason",
    ] = "Projected cash meets target"

    # --------------------------------------------------------
    # Calculate coverage at this ATM and nearby ATMs
    # --------------------------------------------------------
    df["own_days_cover"] = np.where(
        df["predicted_demand"] > 0,
        df["current_cash"] / df["predicted_demand"],
        999.0,
    )

    atm_lookup = df.set_index("atm_id")[
        ["current_cash", "predicted_demand"]
    ].to_dict("index")

    # Treat the graph as undirected for operational coverage: if either
    # direction exists in the table, both ATMs are considered neighbors.
    forward = links[["atm_id", "nearby_atm_id"]].rename(
        columns={"atm_id": "source", "nearby_atm_id": "neighbor"}
    )
    reverse = links[["nearby_atm_id", "atm_id"]].rename(
        columns={"nearby_atm_id": "source", "atm_id": "neighbor"}
    )
    all_links = pd.concat([forward, reverse], ignore_index=True).drop_duplicates()

    neighbor_map = all_links.groupby("source")["neighbor"].apply(list).to_dict()
    neighbor_stats = []

    for atm_id in df["atm_id"]:
        neighbor_ids = set(neighbor_map.get(atm_id, []))
        neighbor_ids.discard(atm_id)
        valid_neighbors = [
            atm_lookup[nid]
            for nid in neighbor_ids
            if nid in atm_lookup
        ]

        neighbor_cash = sum(x["current_cash"] for x in valid_neighbors)
        neighbor_demand = sum(x["predicted_demand"] for x in valid_neighbors)
        neighbor_days_cover = (
            neighbor_cash / neighbor_demand if neighbor_demand > 0 else 999.0
        )

        neighbor_stats.append({
            "atm_id": atm_id,
            "neighbor_count": len(valid_neighbors),
            "neighbor_cash": neighbor_cash,
            "neighbor_predicted_demand": neighbor_demand,
            "neighbor_days_cover": neighbor_days_cover,
        })

    df = df.merge(
        pd.DataFrame(neighbor_stats), on="atm_id", how="left", validate="one_to_one"
    )

    # --------------------------------------------------------
    # Priority scoring
    # --------------------------------------------------------
    # Own cash urgency is dominant. Low nearby coverage adds a secondary
    # urgency signal: an ATM surrounded by low-coverage neighbors may be in
    # a strained local network. This score changes funding order; it does
    # not erase a request. Tune this weighting with backtest evidence.
    df["priority_score"] = (
        100.0 / (df["own_days_cover"] + 0.5)
        + 20.0 / (df["neighbor_days_cover"] + 0.5)
    )

    df["priority"] = "NORMAL"
    df.loc[
        df["cash_level_percent"] <= CRITICAL_CASH_LEVEL_PERCENT, "priority"
    ] = "CRITICAL"
    df.loc[
        (df["cash_level_percent"] > CRITICAL_CASH_LEVEL_PERCENT)
        & (df["cash_level_percent"] <= MIN_CASH_LEVEL_PERCENT),
        "priority",
    ] = "HIGH"
    df.loc[
        df["status_normalized"] != "ACTIVE", "priority"
    ] = "INACTIVE"

    # --------------------------------------------------------
    # Allocate each branch's budget independently
    # --------------------------------------------------------
    df["allocated_refill"] = 0.0
    df["remaining_branch_budget"] = 0.0

    # A branch budget is taken from branch.available_cash for this planning
    # run. This is an explicit project assumption, not a verified real-bank
    # daily budget rule.
    for branch_id, group in df.groupby("branch_id", sort=False):
        budgets = group["branch_available_cash"].dropna().unique()
        if len(budgets) != 1:
            raise RuntimeError(
                f"Branch {branch_id} has inconsistent available_cash values."
            )

        branch_budget = max(0.0, float(budgets[0]))
        ordered = group.sort_values(
            ["priority_score", "requested_refill"],
            ascending=[False, False],
            kind="stable",
        )
        remaining = branch_budget

        for idx in ordered.index:
            requested = max(0.0, float(df.at[idx, "requested_refill"]))
            allocation = min(requested, remaining)
            df.at[idx, "allocated_refill"] = allocation
            remaining = max(0.0, remaining - allocation)
            df.at[idx, "remaining_branch_budget"] = remaining

    df["unfunded_refill"] = (
        df["requested_refill"] - df["allocated_refill"]
    ).clip(lower=0)

    # --------------------------------------------------------
    # Final safety checks
    # --------------------------------------------------------
    tolerance = 0.01
    if (df["allocated_refill"] < -tolerance).any():
        raise RuntimeError("Negative allocation detected.")
    if (
        df["allocated_refill"] - df["requested_refill"] > tolerance
    ).any():
        raise RuntimeError("An allocation exceeds its request.")
    if (
        df["current_cash"] + df["allocated_refill"] - df["capacity"] > tolerance
    ).any():
        raise RuntimeError("An allocation would exceed ATM capacity.")

    branch_allocations = df.groupby("branch_id")["allocated_refill"].sum()
    branch_budgets = df.groupby("branch_id")["branch_available_cash"].first()
    for branch_id, amount in branch_allocations.items():
        budget = max(0.0, float(branch_budgets.loc[branch_id]))
        if amount > budget + tolerance:
            raise RuntimeError(
                f"Branch {branch_id} allocation {amount:.2f} exceeds "
                f"budget {budget:.2f}."
            )

    # --------------------------------------------------------
    # Display and export
    # --------------------------------------------------------
    df = df.sort_values(
        ["priority_score", "requested_refill"],
        ascending=[False, False],
        kind="stable",
    )

    display_columns = [
        "atm_code",
        "location_name",
        "predicted_demand",
        "current_cash",
        "cash_level_percent",
        "own_days_cover",
        "neighbor_count",
        "neighbor_cash",
        "neighbor_days_cover",
        "target_cash",
        "requested_refill",
        "allocated_refill",
        "unfunded_refill",
        "priority",
        "recommendation_reason",
        "branch_name",
    ]

    print("\n========== NETWORK-AWARE ATM REPLENISHMENT PLAN ==========")
    print(f"ATMs evaluated: {len(df)}")
    print(f"Planning date based on latest feature rows: {df['demand_date'].max()}")
    print(f"Demand buffer: {BUFFER_DAYS} forecast-demand days")
    print(f"Minimum cash trigger: {MIN_CASH_LEVEL_PERCENT:.1f}% of capacity")
    print(f"Critical cash level: {CRITICAL_CASH_LEVEL_PERCENT:.1f}% of capacity")
    print(
        df[display_columns].round(2).to_string(index=False)
    )

    summary = df.groupby("branch_name").agg(
        branch_budget=("branch_available_cash", "first"),
        atms=("atm_code", "count"),
        atms_with_requests=("requested_refill", lambda s: int((s > 0).sum())),
        atms_funded=("allocated_refill", lambda s: int((s > 0).sum())),
        requested=("requested_refill", "sum"),
        allocated=("allocated_refill", "sum"),
        unfunded=("unfunded_refill", "sum"),
    )
    summary["budget_remaining"] = (
        summary["branch_budget"] - summary["allocated"]
    ).clip(lower=0)

    print("\n========== BRANCH BUDGET SUMMARY ==========")
    print(summary.round(2).to_string())

    df.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    print(
        "\nReminder: recommendations are planning suggestions only. "
        "They do not create or execute real replenishment orders."
    )


if __name__ == "__main__":
    main()
