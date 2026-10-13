
import os
import numpy as np
import pandas as pd
import joblib

from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# 1. Connect to PostgreSQL
database_url = os.environ.get("DATABASE_URL")

if not database_url:
    raise RuntimeError(
        "DATABASE_URL is missing. Set it in your PowerShell terminal first."
    )

engine = create_engine(database_url)

try:
    # 2. Load the ML dataset from PostgreSQL
    
    query = """
        SELECT
        atm_id,
        demand_date,
        day_of_week,
        is_weekend,
        day_of_month,
        month,
        is_payday_period,
        demand_lag_1,
        demand_lag_7,
        avg_demand_previous_7_days,
        next_day_demand
        FROM atm_ml_dataset
    """


    df = pd.read_sql(query, engine)
finally:
    engine.dispose()

print("\n========== DATASET CHECK ==========")
print("Rows:", len(df))
print("ATMs:", df["atm_id"].nunique())
print("Date range:", df["demand_date"].min(),
      "to", df["demand_date"].max())
print("Missing values:", int(df.isna().sum().sum()))
print("Zero-demand targets:",
      int((df["next_day_demand"] == 0).sum()))

# 3. Clean and prepare data
df["demand_date"] = pd.to_datetime(df["demand_date"])
df = df.dropna().sort_values("demand_date").reset_index(drop=True)


features = [
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


target = "next_day_demand"

# 4. Split chronologically to avoid training on future dates
dates = np.sort(df["demand_date"].unique())

train_cutoff = dates[int(len(dates) * 0.70)]
test_cutoff = dates[int(len(dates) * 0.85)]

# The target is the following day's demand.
# Leave a one-day gap before each evaluation period.
train_df = df[df["demand_date"] < train_cutoff - pd.Timedelta(days=1)]
test_df = df[df["demand_date"] >= test_cutoff]

if train_df.empty or test_df.empty:
    raise ValueError("Training or test set is empty. Check the date range.")

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

print("\n========== DATA SPLIT ==========")
print("Training rows:", len(train_df))
print("Training dates:", train_df["demand_date"].min(),
      "to", train_df["demand_date"].max())
print("Test rows:", len(test_df))
print("Test dates:", test_df["demand_date"].min(),
      "to", test_df["demand_date"].max())

# 5. Treat ATM ID as a category, not a numerical quantity
preprocessor = ColumnTransformer(
    transformers=[
        ("atm", OneHotEncoder(handle_unknown="ignore"), ["atm_id"]),
    ],
    remainder="passthrough",
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
])

# 6. Train the model
print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# 7. Evaluate against a simple baseline
predictions = model.predict(X_test)

# Baseline: tomorrow's demand equals the previous day's demand
baseline_predictions = X_test["demand_lag_1"].to_numpy()

def evaluate(name, actual, predicted):
    print(f"\n{name}")
    print("MAE: ₹{:,.2f}".format(
        mean_absolute_error(actual, predicted)
    ))
    print("RMSE: ₹{:,.2f}".format(
        root_mean_squared_error(actual, predicted)
    ))
    print("R²: {:.4f}".format(
        r2_score(actual, predicted)
    ))

print("\n========== MODEL RESULTS ==========")
evaluate("Baseline (previous-day demand)", y_test, baseline_predictions)
evaluate("Random Forest", y_test, predictions)

# 8. Show sample predictions
results = test_df[["atm_id", "demand_date"]].copy()
results["actual_next_day_demand"] = y_test.to_numpy()
results["predicted_next_day_demand"] = predictions

# ---------- ERROR ANALYSIS ----------

# Positive signed error = overprediction
# Negative signed error = underprediction
results["signed_error"] = (
    results["predicted_next_day_demand"]
    - results["actual_next_day_demand"]
)

results["absolute_error"] = results["signed_error"].abs()

# The row's demand_date is the feature date.
# The model predicts demand for the following day.
results["prediction_for_date"] = (
    pd.to_datetime(results["demand_date"])
    + pd.Timedelta(days=1)
)

print("\n========== ERROR ANALYSIS ==========")

print("\nOverall prediction errors:")
print(
    results[
        ["absolute_error", "signed_error"]
    ].describe().round(2)
)

# Find ATMs with the highest average prediction error
atm_errors = (
    results.groupby("atm_id")
    .agg(
        MAE=("absolute_error", "mean"),
        Average_Signed_Error=("signed_error", "mean"),
        Test_Examples=("absolute_error", "count"),
    )
    .sort_values("MAE", ascending=False)
)

print("\nTop 10 ATMs by prediction error:")
print(atm_errors.head(10).round(2))

# Show the 10 least accurate individual predictions
worst_predictions = results.nlargest(
    10, "absolute_error"
)

print("\n10 Largest Prediction Errors:")
print(
    worst_predictions[
        [
            "atm_id",
            "prediction_for_date",
            "actual_next_day_demand",
            "predicted_next_day_demand",
            "signed_error",
            "absolute_error",
        ]
    ].round(2).to_string(index=False)
)

# Save the complete test predictions and diagnostics
results.to_csv("prediction_error_analysis.csv", index=False)

print("\nSaved: prediction_error_analysis.csv")

results["predicted_next_day_demand"] = (
    results["predicted_next_day_demand"].clip(lower=0)
)

print("\n========== SAMPLE PREDICTIONS ==========")
print(results.head(10).to_string(index=False))

# 9. Save the trained model
joblib.dump(model, "atm_demand_model.joblib")
print("\nModel saved as atm_demand_model.joblib")
print("Training complete!")
