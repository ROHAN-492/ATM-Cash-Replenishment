
import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ATM Cash Replenishment",
    page_icon="🏧",
    layout="wide",
)

st.title("🏧 Predictive ATM Cash Replenishment System")
st.caption(
    "Database-driven demand forecasting and network-aware "
    "ATM cash management"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "atm_demand_model.joblib"

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

# ---------------- DATABASE ----------------
@st.cache_resource
def get_engine(database_url):
    return create_engine(database_url, pool_pre_ping=True)


@st.cache_data(ttl=60)
def load_inventory(database_url):
    engine = get_engine(database_url)
    query = text("""
        SELECT
            a.atm_id,
            a.branch_id,
            a.capacity,
            a.status,
            i.current_cash
        FROM atm a
        JOIN atm_cash_inventory i ON i.atm_id = a.atm_id
        ORDER BY a.atm_id
    """)
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


@st.cache_data(ttl=60)
def load_branches(database_url):
    engine = get_engine(database_url)
    query = text("""
        SELECT branch_id, branch_name, available_cash
        FROM branch
        ORDER BY branch_id
    """)
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


@st.cache_data(ttl=60)
def load_latest_dataset(database_url):
    engine = get_engine(database_url)
    query = text("""
        SELECT
            atm_id, demand_date, daily_demand,
            day_of_week, is_weekend, day_of_month,
            month, is_payday_period, demand_lag_1,
            demand_lag_7, avg_demand_previous_7_days
        FROM atm_ml_dataset
        WHERE demand_lag_1 IS NOT NULL
          AND demand_lag_7 IS NOT NULL
          AND avg_demand_previous_7_days IS NOT NULL
        ORDER BY demand_date DESC, atm_id
    """)
    with engine.connect() as connection:
        data = pd.read_sql(query, connection)

    if data.empty:
        return data

    data["demand_date"] = pd.to_datetime(data["demand_date"])
    latest_date = data["demand_date"].max()
    return data[data["demand_date"] == latest_date].copy()


@st.cache_data(ttl=60)
def load_replenishment_plan():
    path = BASE_DIR / "replenishment_recommendations.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=60)
def load_backtest_summary():
    path = BASE_DIR / "multi_day_backtest_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=60)
def load_backtest_daily():
    path = BASE_DIR / "multi_day_backtest_daily.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=60)
def load_backtest_details():
    path = BASE_DIR / "multi_day_backtest_details.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# ---------------- SIDEBAR ----------------
st.sidebar.header("Dashboard Controls")
st.sidebar.caption("ATM Cash Management")
if st.sidebar.button("🔄 Refresh dashboard"):
    st.cache_data.clear()
    st.rerun()

database_url = os.getenv("DATABASE_URL")

if not database_url:
    st.error(
        "DATABASE_URL is not set. Activate your project virtual "
        "environment and configure the database connection first."
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH.name}")
    st.stop()

try:
    inventory = load_inventory(database_url)
    branches = load_branches(database_url)
    latest_data = load_latest_dataset(database_url)
    replenishment_plan = load_replenishment_plan()
    summary = load_backtest_summary()
    daily = load_backtest_daily()
    details = load_backtest_details()
except Exception as exc:
    st.error("Could not load dashboard data from PostgreSQL or CSV files.")
    st.code(str(exc))
    st.stop()

if inventory.empty:
    st.warning("No ATM inventory records were found.")
    st.stop()

# ---------------- PREPARE DATA ----------------
inventory["current_cash"] = pd.to_numeric(
    inventory["current_cash"], errors="coerce"
).fillna(0)

inventory["capacity"] = pd.to_numeric(
    inventory["capacity"], errors="coerce"
).fillna(0)

inventory["cash_percent"] = (
    inventory["current_cash"]
    .div(inventory["capacity"].where(inventory["capacity"] > 0))
    .mul(100)
    .fillna(0)
)

inventory["cash_status"] = inventory["cash_percent"].apply(
    lambda x: "Low" if x < 25 else ("Medium" if x < 50 else "Healthy")
)

model = joblib.load(MODEL_PATH)

if not latest_data.empty:
    latest_data["predicted_demand"] = model.predict(
        latest_data[FEATURES]
    ).clip(min=0)

    latest_data = latest_data.merge(
        inventory[["atm_id", "branch_id", "current_cash",
                   "capacity", "cash_percent", "status"]],
        on="atm_id",
        how="left",
    )

    
# Match the replenishment engine's refill policy
latest_data["projected_cash_after_one_day"] = (
    latest_data["current_cash"] - latest_data["predicted_demand"]
).clip(lower=0)

latest_data["demand_buffer_target"] = (
    latest_data["predicted_demand"] * 3
).clip(upper=latest_data["capacity"])

latest_data["minimum_cash_target"] = (
    latest_data["capacity"] * 0.25
)

latest_data["target_cash"] = latest_data[
    ["demand_buffer_target", "minimum_cash_target"]
].max(axis=1)

latest_data["suggested_refill"] = (
    latest_data["target_cash"]
    - latest_data["projected_cash_after_one_day"]
).clip(lower=0)

# Never exceed the ATM's remaining capacity
latest_data["capacity_remaining"] = (
    latest_data["capacity"] - latest_data["current_cash"]
).clip(lower=0)

latest_data["suggested_refill"] = latest_data[
    ["suggested_refill", "capacity_remaining"]
].min(axis=1)

# Inactive ATMs should not receive refill recommendations
latest_data["suggested_refill"] = latest_data[
    "suggested_refill"
].where(
    latest_data["status"].astype(str).str.strip().str.upper().eq("ACTIVE"),
    0,
)


# ---------------- OVERVIEW ----------------
st.header("System Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total ATMs", f"{inventory['atm_id'].nunique():,}")
c2.metric("Branches", f"{branches['branch_id'].nunique():,}")
c3.metric(
    "Total Cash in ATMs",
    f"₹{inventory['current_cash'].sum():,.0f}",
)
c4.metric(
    "Low-Cash ATMs",
    f"{int((inventory['cash_percent'] < 25).sum()):,}",
)

st.divider()

# ---------------- NAVIGATION ----------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "ATM Inventory",
        "Demand Predictions",
        "Replenishment Plan",
        "Policy Comparison",
        "Backtest Records",
    ],
)

if page == "Overview":
    left, right = st.columns(2)

    with left:
        st.subheader("ATM Cash Distribution")
        cash_counts = (
            inventory["cash_status"]
            .value_counts()
            .reindex(["Low", "Medium", "Healthy"], fill_value=0)
            .rename_axis("Cash Status")
            .reset_index(name="ATM Count")
        )
        fig = px.pie(
            cash_counts,
            names="Cash Status",
            values="ATM Count",
            hole=0.45,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Cash by Branch")
        cash_by_branch = (
            inventory.groupby("branch_id", as_index=False)
            .agg(
                ATM_Cash=("current_cash", "sum"),
                ATM_Count=("atm_id", "nunique"),
            )
        )
        fig = px.bar(
            cash_by_branch,
            x="branch_id",
            y="ATM_Cash",
            text_auto=",.0f",
            labels={
                "branch_id": "Branch",
                "ATM_Cash": "Cash currently in ATMs (₹)",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Current ATM Inventory")
    st.dataframe(
        inventory[
            ["atm_id", "branch_id", "current_cash",
             "capacity", "cash_percent", "cash_status", "status"]
        ].sort_values("cash_percent"),
        use_container_width=True,
        hide_index=True,
    )

elif page == "ATM Inventory":
    st.subheader("ATM Inventory Explorer")

    branch_options = ["All"] + sorted(
        inventory["branch_id"].dropna().astype(str).unique().tolist()
    )
    selected_branch = st.selectbox("Filter by branch", branch_options)

    view = inventory.copy()
    if selected_branch != "All":
        view = view[
            view["branch_id"].astype(str) == selected_branch
        ]

    st.dataframe(
        view[
            ["atm_id", "branch_id", "current_cash", "capacity",
             "cash_percent", "cash_status", "status"]
        ].sort_values("cash_percent"),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download inventory CSV",
        view.to_csv(index=False).encode("utf-8"),
        file_name="atm_inventory.csv",
        mime="text/csv",
    )

elif page == "Demand Predictions":
    st.subheader("ML Demand Forecasts")

    if latest_data.empty:
        st.warning("No valid feature rows are available for prediction.")
    else:
        st.caption(
            "Forecasts use the latest date available in atm_ml_dataset. "
            "These are model predictions, not confirmed future withdrawals."
        )

        st.metric(
            "Feature date",
            latest_data["demand_date"].max().strftime("%Y-%m-%d"),
        )
        st.metric(
            "Total predicted demand",
            f"₹{latest_data['predicted_demand'].sum():,.0f}",
        )

        fig = px.bar(
            latest_data.sort_values("predicted_demand", ascending=False),
            x="atm_id",
            y="predicted_demand",
            labels={
                "atm_id": "ATM",
                "predicted_demand": "Predicted next-day demand (₹)",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            latest_data[
                ["atm_id", "branch_id", "demand_date",
                 "predicted_demand", "current_cash", "capacity",
                 "cash_percent", "suggested_refill"]
            ].sort_values("suggested_refill", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download prediction CSV",
            latest_data.to_csv(index=False).encode("utf-8"),
            file_name="atm_demand_predictions.csv",
            mime="text/csv",
        )

elif page == "Replenishment Plan":
    st.subheader("Network-Aware ATM Replenishment Plan")
    st.caption(
        "This table shows the replenishment engine's ATM-wise request and "
        "budget-constrained allocation. These are recommendations, not "
        "confirmed cash deliveries."
    )

    if replenishment_plan.empty:
        st.warning(
            "Replenishment output CSV was not found or is empty. "
            "Run `python replenishment_engine.py` from your project folder, "
            "then click **Refresh dashboard**."
        )
    else:
        required = ["requested_refill", "allocated_refill", "unfunded_refill"]
        missing = [column for column in required if column not in replenishment_plan.columns]
        if missing:
            st.error(
                "The replenishment CSV is missing required columns: "
                + ", ".join(missing)
            )
        else:
            for column in required:
                replenishment_plan[column] = pd.to_numeric(
                    replenishment_plan[column], errors="coerce"
                ).fillna(0)

            total_requested = replenishment_plan["requested_refill"].sum()
            total_allocated = replenishment_plan["allocated_refill"].sum()
            total_unfunded = replenishment_plan["unfunded_refill"].sum()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ATMs in plan", f"{len(replenishment_plan):,}")
            m2.metric("Total Requested", f"₹{total_requested:,.0f}")
            m3.metric("Total Allocated", f"₹{total_allocated:,.0f}")
            m4.metric("Total Unfunded", f"₹{total_unfunded:,.0f}")

            view = replenishment_plan.copy()
            if "branch_name" in view.columns:
                branch_options = ["All"] + sorted(
                    view["branch_name"].dropna().astype(str).unique().tolist()
                )
                selected_branch = st.selectbox(
                    "Filter by branch", branch_options, key="replenishment_branch_filter"
                )
                if selected_branch != "All":
                    view = view[view["branch_name"].astype(str) == selected_branch]

            preferred_columns = [
                "atm_code", "location_name", "branch_name", "predicted_demand",
                "current_cash", "target_cash", "requested_refill",
                "allocated_refill", "unfunded_refill", "priority",
                "recommendation_reason",
            ]
            display_columns = [c for c in preferred_columns if c in view.columns]
            st.dataframe(
                view[display_columns].sort_values(
                    "requested_refill", ascending=False
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download replenishment plan CSV",
                view.to_csv(index=False).encode("utf-8"),
                file_name="replenishment_recommendations.csv",
                mime="text/csv",
            )

elif page == "Policy Comparison":
    st.subheader("Multi-Day Policy Comparison")

    if summary.empty:
        st.warning("Run multi_day_backtest.py to create the summary CSV.")
    else:
        st.caption(
            "Simulated results only. Starting cash comes from current "
            "inventory; historical cash balances are not available."
        )

        a, b = st.columns(2)
        a.metric("Policies Compared", f"{summary['policy'].nunique()}")
        if "simulated_days" in summary.columns:
            b.metric(
                "Planning Dates",
                f"{int(summary['simulated_days'].max())}",
            )

        metric_options = {
            "Cash allocated (₹)": "total_allocated_refill",
            "Shortage ATM-days": "total_shortage_atm_days",
            "Unmet demand (₹)": "total_unmet_demand",
            "Average ending cash (₹)": "average_daily_end_cash",
        }
        selected_metric = st.selectbox(
            "Compare metric", list(metric_options.keys())
        )
        metric_col = metric_options[selected_metric]

        fig = px.bar(
            summary,
            x="policy",
            y=metric_col,
            color="policy",
            labels={"policy": "Policy", metric_col: selected_metric},
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(summary, use_container_width=True, hide_index=True)

elif page == "Backtest Records":
    st.subheader("Daily Simulation Records")

    if daily.empty:
        st.warning("Daily backtest CSV not found.")
    else:
        policies = ["All"] + sorted(daily["policy"].unique().tolist())
        chosen = st.selectbox("Policy", policies)
        view = daily.copy()

        if chosen != "All":
            view = view[view["policy"] == chosen]

        st.dataframe(view, use_container_width=True, hide_index=True)

        st.download_button(
            "Download daily records CSV",
            view.to_csv(index=False).encode("utf-8"),
            file_name="atm_backtest_daily.csv",
            mime="text/csv",
        )

        if not details.empty:
            with st.expander("View ATM-level simulation details"):
                detail_view = details.copy()
                if chosen != "All":
                    detail_view = detail_view[
                        detail_view["policy"] == chosen
                    ]
                st.dataframe(
                    detail_view,
                    use_container_width=True,
                    hide_index=True,
                )

st.divider()
st.caption(
    "Academic DBMS project • PostgreSQL + Random Forest + "
    "Network-aware replenishment simulation"
)
