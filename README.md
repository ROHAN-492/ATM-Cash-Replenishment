# Predictive ATM Cash Replenishment System

A database-driven machine learning project that predicts next-day ATM cash demand and generates replenishment recommendations while considering branch-level daily budgets and nearby-ATM network relationships.

> **Academic/demo project:** The data is synthetic and the backtest is simulated. Results are not evidence of live banking performance.

## Overview

ATM replenishment needs to balance two goals: reduce cash shortages and avoid allocating more cash than necessary. This project combines PostgreSQL, a Random Forest regression model, and a Streamlit dashboard to explore ATM demand and compare replenishment policies.

## Features

- PostgreSQL-backed branch, ATM, inventory, transaction, replenishment, and nearby-ATM data.
- Daily demand dataset with calendar features, lagged demand, rolling averages, and next-day demand.
- Random Forest model for next-day demand prediction.
- Replenishment recommendation engine with branch-level daily budget allocation.
- Network-aware policy comparison and multi-day backtesting.
- Streamlit dashboard for overview, inventory, demand predictions, policy comparison, and backtest results.

## Technology Stack

Python · PostgreSQL · SQL · pandas · scikit-learn · joblib · Streamlit

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `train_model.py` | Trains the demand-prediction model |
| `replenishment_engine.py` | Generates replenishment recommendations and branch summaries |
| `multi_day_backtest.py` | Runs multi-day policy backtesting |
| `policy_evaluation.py` | Evaluates replenishment policies |
| `inspect_schema.py` | Inspects the database schema |
| `test_connection.py` | Tests the PostgreSQL connection |
| `atm_demand_model.joblib` | Locally generated model; excluded from GitHub because it exceeds the 100 MB file limit |
| `*.csv` | Generated datasets, recommendations, evaluations, and backtest outputs |

The database schema used by the project includes `branch`, `atm`, `atm_cash_inventory`, `atm_transaction`, `replenishment`, `nearby_atm`, and the `atm_ml_dataset` view.

## Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies
- PostgreSQL
- Git (for cloning the repository)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ROHAN-492/ATM-Cash-Replenishment.git
cd ATM-Cash-Replenishment
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install pandas scikit-learn joblib streamlit sqlalchemy psycopg2-binary python-dotenv
```

If a `requirements.txt` is added to the repository later, dependencies can instead be installed with `python -m pip install -r requirements.txt`.

### 4. Configure the database

Create a `.env` file in the project root:

```dotenv
DATABASE_URL=postgresql+psycopg2://DB_USER:DB_PASSWORD@DB_HOST:5432/DB_NAME
```

Replace the placeholders with your PostgreSQL connection details. URL-encode special characters in the username or password when needed.

**Never commit `.env` or publish database credentials.** The repository's `.gitignore` should keep this file out of Git.

### 5. Prepare the database

Create or restore the schema and populate the required tables and `atm_ml_dataset` view before running the scripts. A database initialization SQL file was not included in the repository file list when this README was prepared; use the schema and data setup created during development.

### 6. Train the model

The trained model is not stored in GitHub because it exceeds GitHub's standard per-file limit. After configuring and preparing the database, run:

```powershell
python train_model.py
```

The script should generate `atm_demand_model.joblib` if configured to save the model under that filename. Check the script output to confirm.

### 7. Launch the dashboard

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Running the Main Scripts

Run these from the activated virtual environment after the database and model are ready:

```powershell
python test_connection.py
python train_model.py
python replenishment_engine.py
python policy_evaluation.py
python multi_day_backtest.py
streamlit run app.py
```

Run `test_connection.py` first. Some scripts may create or overwrite CSV outputs, so back up results you want to keep.

## Model Evaluation and Backtesting

In one development run, chronological holdout evaluation reported:

| Metric | Value |
|---|---:|
| Test rows | 10,800 |
| Mean Absolute Error (MAE) | ₹6,729.39 |
| Root Mean Squared Error (RMSE) | ₹9,179.01 |
| R² | 0.5195 |
| Lag-1 baseline MAE | ₹11,884.67 |
| Lag-1 baseline RMSE | ₹15,221.05 |

A simulated 108-day backtest compared ML-only, network-aware, and threshold-based policies. In that run, the network-aware policy reduced shortage ATM-days from 9 to 2 versus ML-only, and reduced unmet demand from ₹53,779.81 to ₹15,678.68, with approximately ₹38,101 in additional allocation.

These values describe one development run and may change with data, features, random seeds, or policy parameters. The backtest used simulated conditions and started from current inventory rather than reconstructed historical inventory. It is **not** a real-world banking result or a guarantee of future performance.

## Assumptions and Limitations

- `branch.available_cash` represents the branch's **daily replenishment budget**, not its current physical cash balance.
- Recommendations are for planning and analysis; the engine does not automatically dispatch cash or update real-world inventory.
- Backtest results are simulated, not live ATM outcomes.
- The model file must be generated locally before running components that load it.
- Database schema and data initialization must be completed separately.

## Security

- Keep `.env`, database credentials, access tokens, and private customer information out of Git.
- Use synthetic or anonymized data only.
- Rotate any credential that has accidentally been exposed.
- Review generated CSVs before publishing to ensure they contain no secrets or sensitive information.

## Future Improvements

- Add a reproducible SQL database initialization script.
- Add a `requirements.txt` with pinned dependency versions.
- Add automated tests and data-validation checks.
- Evaluate the model across multiple time windows and compare other forecasting methods.
- Reconstruct historical inventory for more realistic backtesting.
- Consider operational integration only after proper validation and authorization.

## Author

**Rohan-492**
**Arsh Shukla**


## License

No license has been specified. Unless a license is added, standard copyright restrictions apply.
