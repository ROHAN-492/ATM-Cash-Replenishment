
import os
from sqlalchemy import create_engine, text

database_url = os.environ.get("DATABASE_URL")

if not database_url:
    raise RuntimeError("DATABASE_URL has not been set.")

engine = create_engine(database_url)

try:
    with engine.connect() as connection:
        print("Connected to PostgreSQL!")

        result = connection.execute(
            text("SELECT COUNT(*) FROM atm_ml_dataset")
        )

        print("ML dataset rows:", result.scalar_one())
finally:
    engine.dispose()
