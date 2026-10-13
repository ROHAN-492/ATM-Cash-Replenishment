
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

TEST_START = "2026-06-25"
TEST_END = "2026-10-10"  # Exclude the final date with incomplete next-day targets
BUFFER_DAYS = 3
THRESHOLD_PERCENT = 50


def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is missing. Activate your virtual environment "
            "and set the database connection variable first."
        )

    engine = create_engine(database_url)
    model = load("atm_demand_model.joblib")

    query = text("""
        SELECT
            atm_id,
            demand_date,
            daily_demand,
            next_day_demand,
            day_of_week,
            is_weekend,
            day_of_month,
            month,
            is_payday_period,
            demand_lag_1,
            demand_lag_7,
            avg_demand_previous_7_days
        FROM atm_ml_dataset
        WHERE demand_date >= :start_date
          AND demand_date <= :end_date
        ORDER BY demand_date, atm_id
    """)

    with engine.connect() as connection:
        data = pd.read_sql(
            query,
            connection,
            params={
                "start_date": TEST_START,
                "end_date": TEST_END,
            },
        )

        inventory = pd.read_sql(
            text("""
                SELECT
                    a.atm_id,
                    a.capacity,
                    a.status,
                    i.current_cash,
                    a.branch_id,
                    b.available_cash AS branch_budget
                FROM atm a
                JOIN atm_cash_inventory i ON i.atm_id = a.atm_id
                JOIN branch b ON b.branch_id = a.branch_id
            """),
            connection,
        )

        links = pd.read_sql(
            text("""
                SELECT atm_id, nearby_atm_id
                FROM nearby_atm
            """),
            connection,
        )

    if data.empty:
        raise RuntimeError("No test rows found for the selected date range.")

    # Remove incomplete/invalid target rows.
    data = data.dropna(subset=FEATURES + ["next_day_demand"]).copy()
    data = data[data["next_day_demand"] > 0].copy()

    data["predicted_demand"] = np.maximum(
        model.predict(data[FEATURES]), 0
    )

    mae = mean_absolute_error(
        data["next_day_demand"], data["predicted_demand"]
    )
    rmse = np.sqrt(mean_squared_error(
        data["next_day_demand"], data["predicted_demand"]
    ))
    r2 = r2_score(data["next_day_demand"], data["predicted_demand"])

    print("\n========== DEMAND MODEL EVALUATION ==========")
    print(f"Test rows: {len(data):,}")
    print(f"MAE:  ₹{mae:,.2f}")
    print(f"RMSE: ₹{rmse:,.2f}")
    print(f"R²:   {r2:.4f}")

    # Build a transparent one-day planning snapshot from latest test date.
    latest_date = data["demand_date"].max()
    snapshot = data[data["demand_date"] == latest_date].copy()
    snapshot = snapshot.merge(inventory, on="atm_id", how="inner")

    # Threshold baseline: request refill when current cash is below 25%
    # of capacity. Refill toward two days of predicted demand.
    snapshot["cash_percent"] = (
        snapshot["current_cash"] / snapshot["capacity"] * 100
    )
    
    # Target cash should cover two days of demand.
    snapshot["target_cash"] = np.minimum(
    snapshot["predicted_demand"] * BUFFER_DAYS,
    snapshot["capacity"],
    )

    # Estimate cash remaining after the next day's withdrawals.
    snapshot["projected_cash"] = np.maximum(
    snapshot["current_cash"] - snapshot["predicted_demand"],
    0,
    )

    # Refill the projected balance toward the target.
    snapshot["requested_refill"] = np.maximum(
    snapshot["target_cash"] - snapshot["projected_cash"],
    0,
    )

    # Never exceed the ATM's physical capacity.
    capacity_remaining = np.maximum(
    snapshot["capacity"] - snapshot["current_cash"],
    0,
    )

    snapshot["requested_refill"] = np.minimum(
    snapshot["requested_refill"],
    capacity_remaining,
    )
    

    # PASTE THE DEBUG CODE HERE
    print("\nCHECK REFILL FORMULA")
    print(snapshot[
    ["current_cash", "predicted_demand",
     "target_cash", "projected_cash", "requested_refill"]
    ].head(10).to_string(index=False))

    print(
    "Total requested:",
    snapshot["requested_refill"].sum()
    )

    # Existing code continues below
    snapshot.loc[
        snapshot["status"] != "ACTIVE", "requested_refill"
    ] = 0

    snapshot["threshold_request"] = np.where(
        snapshot["cash_percent"] < THRESHOLD_PERCENT,
        snapshot["requested_refill"],
        0,
    )
    snapshot["ml_only_request"] = snapshot["requested_refill"]
    print("\nDEBUG — SNAPSHOT CHECK")
    print("Snapshot rows:", len(snapshot))
    print("Current cash range:")
    print(snapshot["current_cash"].describe())
    print("\nPredicted demand range:")
    print(snapshot["predicted_demand"].describe())
    print("\nCash percentage range:")
    print(snapshot["cash_percent"].describe())
    print("\nRequested refill total:", snapshot["requested_refill"].sum())
    print("Threshold requests:", snapshot["threshold_request"].sum())
    print("Active ATMs:", (snapshot["status"] == "ACTIVE").sum())

    # Network coverage based on current cash and predicted demand.
    neighbor_cash = inventory.set_index("atm_id")["current_cash"].to_dict()
    neighbor_demand = (
        snapshot.set_index("atm_id")["predicted_demand"].to_dict()
    )

    neighbors_by_atm = links.groupby("atm_id")["nearby_atm_id"].apply(list).to_dict()

    def neighbor_cover(atm_id):
        neighbor_ids = neighbors_by_atm.get(atm_id, [])
        cash = sum(neighbor_cash.get(n, 0) for n in neighbor_ids)
        demand = sum(neighbor_demand.get(n, 0) for n in neighbor_ids)
        return cash / demand if demand > 0 else 999

    snapshot["neighbor_days_cover"] = snapshot["atm_id"].map(neighbor_cover)
    snapshot["own_days_cover"] = (
        snapshot["current_cash"] /
        snapshot["predicted_demand"].replace(0, np.nan)
    ).fillna(999)

    snapshot["network_priority_score"] = (
        100 / (snapshot["own_days_cover"] + 0.5)
        + 20 / (snapshot["neighbor_days_cover"] + 0.5)
    )

    # Fair comparison: each policy has the same branch budget and request
    # amounts; only allocation order differs for network-aware policy.
    def allocate(frame, request_col, network_aware):
        frame = frame.copy()
        frame["allocated"] = 0.0

        for branch_id, group in frame.groupby("branch_id"):
            budget_values = group["branch_budget"].dropna()
            budget = float(budget_values.iloc[0]) if len(budget_values) else 0

            if network_aware:
                group = group.sort_values(
                    ["network_priority_score", request_col],
                    ascending=[False, False],
                )
            else:
                group = group.sort_values(request_col, ascending=False)

            for idx, row in group.iterrows():
                amount = min(float(row[request_col]), budget)
                frame.loc[idx, "allocated"] = amount
                budget -= amount

        return frame

    baseline = allocate(snapshot, "threshold_request", False)
    ml_only = allocate(snapshot, "ml_only_request", False)
    network = allocate(snapshot, "ml_only_request", True)

    summary = pd.DataFrame([
        {
            "policy": "Threshold baseline",
            "requested": baseline["threshold_request"].sum(),
            "allocated": baseline["allocated"].sum(),
            "ATMs receiving refill": (baseline["allocated"] > 0).sum(),
        },
        {
            "policy": "ML-only",
            "requested": ml_only["ml_only_request"].sum(),
            "allocated": ml_only["allocated"].sum(),
            "ATMs receiving refill": (ml_only["allocated"] > 0).sum(),
        },
        {
            "policy": "Network-aware",
            "requested": network["ml_only_request"].sum(),
            "allocated": network["allocated"].sum(),
            "ATMs receiving refill": (network["allocated"] > 0).sum(),
        },
    ])

    print(f"\nPlanning snapshot date: {latest_date}")
    print("\n========== POLICY COMPARISON ==========")
    print(summary.to_string(index=False))

    network[
        [
            "atm_id",
            "branch_id",
            "current_cash",
            "predicted_demand",
            "neighbor_days_cover",
            "network_priority_score",
            "ml_only_request",
            "allocated",
        ]
    ].to_csv("policy_evaluation_details.csv", index=False)

    summary.to_csv("policy_evaluation_summary.csv", index=False)
    print("\nSaved policy_evaluation_details.csv")
    print("Saved policy_evaluation_summary.csv")
    print(
        "\nNote: this compares one planning snapshot. It does not yet "
        "measure shortage days or prove long-term policy superiority."
    )


if __name__ == "__main__":
    main()
