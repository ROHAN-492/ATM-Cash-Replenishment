
import os
from sqlalchemy import create_engine, text

database_url = os.environ.get("DATABASE_URL")

if not database_url:
    raise RuntimeError("DATABASE_URL is not set.")

engine = create_engine(database_url)

tables = [
    "replenishment",
    "atm_cash_inventory",
    "atm_transaction",
    "atm",
    "branch",
]

query = text("""
    SELECT
        column_name,
        data_type,
        is_nullable,
        column_default
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = :table_name
    ORDER BY ordinal_position
""")

try:
    with engine.connect() as connection:
        for table in tables:
            print(f"\n{'=' * 65}")
            print(f"TABLE: {table}")
            print("=" * 65)

            rows = connection.execute(
                query, {"table_name": table}
            ).mappings().all()

            if not rows:
                print("Table not found.")
                continue

            for row in rows:
                print(
                    f"{row['column_name']} | "
                    f"{row['data_type']} | "
                    f"nullable={row['is_nullable']} | "
                    f"default={row['column_default']}"
                )
finally:
    engine.dispose()
