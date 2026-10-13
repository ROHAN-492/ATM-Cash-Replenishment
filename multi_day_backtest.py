import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Multi-day backtest for the ATM cash replenishment project.
# Uses historical demand to evaluate forecasts and simulated cash balances.
# This is a simulation, not a record of real ATM shortages.

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
TEST_END = "2026-10-10"  # End date is included; incomplete/zero targets are excluded below.
BUFFER_DAYS = 3
THRESHOLD_PERCENT = 50


def allocate_requests(frame, request_col, branch_budgets, network_aware):
    """Allocate each branch's daily budget in priority order."""
    result = frame.copy()
    result["allocated_refill"] = 0.0

    for branch_id, group in result.groupby("branch_id", sort=False):
        budget = max(float(branch_budgets.get(branch_id, 0.0)), 0.0)

        if network_aware:
            group = group.sort_values(
                ["network_priority_score", request_col],
                ascending=[False, False],
            )
        else:
            group = group.sort_values(
                [request_col, "atm_id"],
                ascending=[False, True],
            )

        for idx, row in group.iterrows():
            requested = max(float(row[request_col]), 0.0)
            amount = min(requested, budget)
            result.loc[idx, "allocated_refill"] = amount
            budget -= amount

    return result


def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is missing. Activate your virtual environment "
            "and set the database connection variable first."
        )

    model_path = "atm_demand_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find {model_path}. Run train_model.py first "
            "or place the trained model in this project folder."
        )

    engine = create_engine(database_url)
    model = load(model_path)

    dataset_query = text("""
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

    inventory_query = text("""
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
    """)

    links_query = text("""
        SELECT atm_id, nearby_atm_id
        FROM nearby_atm
    """)

    with engine.connect() as connection:
        data = pd.read_sql(
            dataset_query,
            connection,
            params={"start_date": TEST_START, "end_date": TEST_END},
        )
        inventory = pd.read_sql(inventory_query, connection)
        links = pd.read_sql(links_query, connection)

    if data.empty:
        raise RuntimeError(
            "No rows found in atm_ml_dataset for the selected test dates."
        )

    data["demand_date"] = pd.to_datetime(data["demand_date"])
    data = data.dropna(subset=FEATURES + ["next_day_demand"]).copy()

    # The final records in this synthetic dataset may have zero targets because
    # no following-day transactions were generated. Exclude those incomplete rows.
    data = data[data["next_day_demand"] > 0].copy()
    if data.empty:
        raise RuntimeError(
            "No valid positive next_day_demand targets remain. Check the "
            "dataset date range and synthetic transaction generation."
        )

    data["predicted_demand"] = np.maximum(
        model.predict(data[FEATURES]), 0
    )
    data["actual_next_day_demand"] = np.maximum(
        data["next_day_demand"].astype(float), 0
    )

    mae = mean_absolute_error(
        data["actual_next_day_demand"], data["predicted_demand"]
    )
    rmse = np.sqrt(mean_squared_error(
        data["actual_next_day_demand"], data["predicted_demand"]
    ))
    r2 = r2_score(data["actual_next_day_demand"], data["predicted_demand"])

    print("\n========== DEMAND MODEL EVALUATION ==========")
    print(f"Test rows: {len(data):,}")
    print(f"Dates evaluated: {data['demand_date'].nunique():,}")
    print(f"ATMs represented: {data['atm_id'].nunique():,}")
    print(f"MAE:  ₹{mae:,.2f}")
    print(f"RMSE: ₹{rmse:,.2f}")
    print(f"R²:   {r2:.4f}")

    inventory = inventory.copy()
    inventory["current_cash"] = pd.to_numeric(
        inventory["current_cash"], errors="coerce"
    ).fillna(0).clip(lower=0)
    inventory["capacity"] = pd.to_numeric(
        inventory["capacity"], errors="coerce"
    ).fillna(0).clip(lower=0)
    inventory["branch_budget"] = pd.to_numeric(
        inventory["branch_budget"], errors="coerce"
    ).fillna(0).clip(lower=0)

    if inventory["atm_id"].duplicated().any():
        raise RuntimeError("Inventory query returned duplicate ATM IDs.")
    if inventory.empty:
        raise RuntimeError("No ATM inventory rows were returned.")

    # Starting balances are the current inventory values, then balances are
    # simulated forward. This is a scenario simulation, not historical inventory.
    starting_cash = inventory.set_index("atm_id")["current_cash"].to_dict()
    capacities = inventory.set_index("atm_id")["capacity"].to_dict()
    statuses = inventory.set_index("atm_id")["status"].to_dict()
    branches = inventory.set_index("atm_id")["branch_id"].to_dict()

    # A fresh daily replenishment budget is available each simulated day.
    # This interprets branch.available_cash as a daily planning budget.
    branch_budgets = (
        inventory.groupby("branch_id")["branch_budget"].first().to_dict()
    )

    neighbor_map = links.groupby("atm_id")["nearby_atm_id"].apply(list).to_dict()

    # Ensure the starting state has one row for every ATM represented in the
    # demand dataset. Missing inventory is an explicit error, not silently filled.
    dataset_atms = set(data["atm_id"].unique())
    inventory_atms = set(inventory["atm_id"].unique())
    missing_inventory = dataset_atms - inventory_atms
    if missing_inventory:
        raise RuntimeError(
            "Missing inventory for ATM IDs: "
            + ", ".join(map(str, sorted(missing_inventory)))
        )

    policy_names = ["Threshold baseline", "ML-only", "Network-aware"]
    cash_state = {
        policy: {
            atm_id: min(
                starting_cash[atm_id],
                capacities.get(atm_id, starting_cash[atm_id]),
            )
            for atm_id in dataset_atms
        }
        for policy in policy_names
    }

    daily_rows = []
    detail_rows = []

    for planning_date, day in data.groupby("demand_date", sort=True):
        day = day.copy()
        day = day.drop_duplicates(subset=["atm_id"], keep="last")
        day["capacity"] = day["atm_id"].map(capacities).astype(float)
        day["branch_id"] = day["atm_id"].map(branches)
        day["status"] = day["atm_id"].map(statuses)
        day["actual_demand"] = day["actual_next_day_demand"].astype(float)
        day["predicted_demand"] = day["predicted_demand"].astype(float)

        for policy in policy_names:
            state = cash_state[policy]
            current_cash = day["atm_id"].map(state).astype(float)
            day_policy = day[
                ["atm_id", "branch_id", "capacity", "status",
                 "predicted_demand", "actual_demand"]
            ].copy()
            day_policy["current_cash"] = current_cash
            day_policy["cash_percent"] = np.where(
                day_policy["capacity"] > 0,
                day_policy["current_cash"] / day_policy["capacity"] * 100,
                0.0,
            )

            # ML-based policies refill enough to target BUFFER_DAYS of forecast
            # demand after expected next-day withdrawals.
            target_cash = np.minimum(
                day_policy["predicted_demand"] * BUFFER_DAYS,
                day_policy["capacity"],
            )
            projected_cash = np.maximum(
                day_policy["current_cash"] - day_policy["predicted_demand"],
                0.0,
            )
            ml_request = np.maximum(target_cash - projected_cash, 0.0)
            capacity_remaining = np.maximum(
                day_policy["capacity"] - day_policy["current_cash"], 0.0
            )
            ml_request = np.minimum(ml_request, capacity_remaining)
            ml_request = np.where(
                day_policy["status"].astype(str).str.upper().eq("ACTIVE"),
                ml_request,
                0.0,
            )

            # Threshold baseline triggers only below the configured cash level.
            threshold_target_cash = (
            day_policy["capacity"] * THRESHOLD_PERCENT / 100
            )

            threshold_request = np.where(
            day_policy["cash_percent"] < THRESHOLD_PERCENT,
            np.maximum(
            threshold_target_cash - day_policy["current_cash"],
            0.0,
            ),
            0.0,
            )

            threshold_request = np.minimum(
                threshold_request,
                capacity_remaining,
            )

            threshold_request = np.where(
                day_policy["status"].astype(str).str.upper().eq("ACTIVE"),
                threshold_request,
                0.0,
            )

            if policy == "Threshold baseline":
                day_policy["request"] = threshold_request
            else:
                day_policy["request"] = ml_request

            # Calculate network priority using each policy's own current cash
            # state and this date's demand forecasts.
            cash_lookup = dict(zip(day_policy["atm_id"], day_policy["current_cash"]))
            demand_lookup = dict(zip(day_policy["atm_id"], day_policy["predicted_demand"]))

            def calc_neighbor_cover(atm_id):
                neighbor_ids = neighbor_map.get(atm_id, [])
                neighbor_cash = sum(
                    cash_lookup.get(n, starting_cash.get(n, 0.0))
                    for n in neighbor_ids
                )
                neighbor_demand = sum(
                    demand_lookup.get(n, 0.0) for n in neighbor_ids
                )
                return neighbor_cash / neighbor_demand if neighbor_demand > 0 else 999.0

            day_policy["own_days_cover"] = np.where(
                day_policy["predicted_demand"] > 0,
                day_policy["current_cash"] / day_policy["predicted_demand"].replace(0, np.nan),
                999.0,
            )
            day_policy["neighbor_days_cover"] = day_policy["atm_id"].map(
                calc_neighbor_cover
            )

            # Low own cover increases urgency. Low nearby cover also increases
            # urgency because fewer nearby ATMs may be able to provide support.
            day_policy["network_priority_score"] = (
                100 / (day_policy["own_days_cover"] + 0.5)
                + 20 / (day_policy["neighbor_days_cover"] + 0.5)
            )

            allocated = allocate_requests(
                day_policy,
                "request",
                branch_budgets,
                network_aware=(policy == "Network-aware"),
            )
            # Validate allocations against each branch's daily budget.
            branch_allocations = allocated.groupby("branch_id")[
            "allocated_refill"
            ].sum()

            for branch_id, amount in branch_allocations.items():
                budget = max(float(branch_budgets.get(branch_id, 0.0)), 0.0)

            if amount > budget + 0.01:
                raise RuntimeError(
                f"Branch budget exceeded on {planning_date.date()}, "
                f"policy={policy}, branch={branch_id}: "
                f"allocated={amount:.2f}, budget={budget:.2f}"
            )

            allocated["cash_after_refill"] = np.minimum(
                allocated["current_cash"] + allocated["allocated_refill"],
                allocated["capacity"],
            )
            allocated["shortage_amount"] = np.maximum(
                allocated["actual_demand"] - allocated["cash_after_refill"],
                0.0,
            )
            allocated["cash_end_of_day"] = np.maximum(
                allocated["cash_after_refill"] - allocated["actual_demand"],
                0.0,
            )
            allocated["shortage_day"] = allocated["shortage_amount"] > 0
            allocated["planning_date"] = planning_date.date().isoformat()
            allocated["policy"] = policy

            # Update this policy's simulated cash balance for the next date.
            for _, row in allocated.iterrows():
                state[row["atm_id"]] = float(row["cash_end_of_day"])

            daily_rows.append({
                "planning_date": planning_date.date().isoformat(),
                "policy": policy,
                "atm_count": int(len(allocated)),
                "requested_refill": float(allocated["request"].sum()),
                "allocated_refill": float(allocated["allocated_refill"].sum()),
                "atms_receiving_refill": int((allocated["allocated_refill"] > 0).sum()),
                "shortage_atm_days": int(allocated["shortage_day"].sum()),
                "total_unmet_demand": float(allocated["shortage_amount"].sum()),
                "end_cash_total": float(allocated["cash_end_of_day"].sum()),
                "branch_budget_available": float(sum(branch_budgets.values())),
                "budget_utilization_percent": (
                    float(allocated["allocated_refill"].sum())
                    / float(sum(branch_budgets.values())) * 100
                    if sum(branch_budgets.values()) > 0 else 0.0
                ),
            })

            detail_columns = [
                "planning_date", "policy", "atm_id", "branch_id", "status",
                "current_cash", "capacity", "predicted_demand", "actual_demand",
                "cash_percent", "own_days_cover", "neighbor_days_cover",
                "network_priority_score", "request", "allocated_refill",
                "cash_after_refill", "shortage_amount", "shortage_day",
                "cash_end_of_day",
            ]
            detail_rows.extend(allocated[detail_columns].to_dict("records"))

    daily = pd.DataFrame(daily_rows)
    details = pd.DataFrame(detail_rows)

    if daily.empty or details.empty:
        raise RuntimeError(
            "Backtest produced no simulation results. Check dataset dates and targets."
        )

    summary = (
        daily.groupby("policy", as_index=False)
        .agg(
            simulated_days=("planning_date", "nunique"),
            total_requested_refill=("requested_refill", "sum"),
            total_allocated_refill=("allocated_refill", "sum"),
            total_refill_events=("atms_receiving_refill", "sum"),
            total_shortage_atm_days=("shortage_atm_days", "sum"),
            total_unmet_demand=("total_unmet_demand", "sum"),
            average_daily_end_cash=("end_cash_total", "mean"),
            average_daily_budget_utilization_percent=("budget_utilization_percent", "mean"),
        )
    )

    # Rates make comparisons easier to interpret across the same number of ATM-days.
    atm_days = int(details.groupby("policy").size().min())
    summary["shortage_atm_days_per_1000_atm_days"] = (
        summary["total_shortage_atm_days"] / atm_days * 1000
        if atm_days > 0 else 0.0
    )
    summary["total_unmet_demand"] = summary["total_unmet_demand"].round(2)
    summary["total_requested_refill"] = summary["total_requested_refill"].round(2)
    summary["total_allocated_refill"] = summary["total_allocated_refill"].round(2)
    summary["average_daily_end_cash"] = summary["average_daily_end_cash"].round(2)
    summary["average_daily_budget_utilization_percent"] = (
        summary["average_daily_budget_utilization_percent"].round(2)
    )

    daily.to_csv("multi_day_backtest_daily.csv", index=False)
    details.to_csv("multi_day_backtest_details.csv", index=False)
    summary.to_csv("multi_day_backtest_summary.csv", index=False)

    print("\n========== MULTI-DAY BACKTEST SUMMARY ==========")
    print(f"Planning dates: {daily['planning_date'].nunique():,}")
    print(f"ATM-days per policy: {atm_days:,}")
    print(f"Buffer target: {BUFFER_DAYS} forecast-demand days")
    print(f"Threshold trigger: cash below {THRESHOLD_PERCENT}% of capacity")
    print("\n" + summary.to_string(index=False))

    print("\nSaved files:")
    print("  multi_day_backtest_daily.csv")
    print("  multi_day_backtest_details.csv")
    print("  multi_day_backtest_summary.csv")
    print(
        "\nInterpretation note: starting cash comes from the current inventory table, "
        "then is simulated forward. Branch budgets reset daily to branch.available_cash. "
        "These are simulation assumptions, not historical records of cash availability "
        "or actual shortage events. A policy is not automatically better just because "
        "it allocates more cash; compare shortages and refill costs together."
    )


if __name__ == "__main__":
    main()
