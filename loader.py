"""
Data Loader and SQL Ingestion Engine for Retail Pulse Analytics.

Manages relational database connections via SQLAlchemy and ingests transactional,
customer, and product data into optimized Pandas DataFrames with type integrity checks.
"""

import os
from typing import Dict, Optional, Tuple
import pandas as pd
from sqlalchemy import create_engine, Engine, text


class DataLoader:
    """
    Connects to the Retail Pulse SQLite database and loads clean, validated
    DataFrames for analytical engines (RFM, Cohorts, and Sales Trends).
    """

    def __init__(self, db_path: str = "retail_pulse.db"):
        self.db_path = os.path.abspath(db_path)
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Database file not found at: {self.db_path}. "
                f"Please initialize the database using mock_data.py or main.py --rebuild-db."
            )
        self.engine: Engine = create_engine(f"sqlite:///{self.db_path}", echo=False)

    def get_database_summary(self) -> Dict[str, int]:
        """Return row counts for each primary table in the database."""
        summary = {}
        tables = ["customers", "products", "orders", "order_items"]
        with self.engine.connect() as conn:
            for table in tables:
                res = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                summary[table] = int(res) if res is not None else 0
        return summary

    def load_customers(self) -> pd.DataFrame:
        """Load and parse the customers table."""
        query = """
        SELECT
            customer_id,
            customer_unique_id,
            first_name,
            last_name,
            first_name || ' ' || last_name AS full_name,
            email,
            city,
            state,
            country,
            signup_date,
            acquisition_channel
        FROM customers
        """
        df = pd.read_sql(query, self.engine)
        df["signup_date"] = pd.to_datetime(df["signup_date"])
        return df

    def load_products(self) -> pd.DataFrame:
        """Load and parse the products table with profit margin calculations."""
        query = """
        SELECT
            product_id,
            sku,
            product_name,
            category,
            sub_category,
            unit_price,
            cost_price,
            (unit_price - cost_price) AS unit_margin,
            ROUND(((unit_price - cost_price) / unit_price) * 100, 2) AS margin_percentage,
            is_active
        FROM products
        """
        df = pd.read_sql(query, self.engine)
        return df

    def load_clean_transactions(
        self,
        exclude_cancelled_and_returned: bool = True
    ) -> pd.DataFrame:
        """
        Ingest orders joined with customer dimensions.
        
        Args:
            exclude_cancelled_and_returned: If True, filters out orders with status
                                            'Cancelled' or 'Returned' to reflect true revenue.
        
        Returns:
            Cleaned transaction-level DataFrame indexed by order with customer details.
        """
        status_filter = ""
        if exclude_cancelled_and_returned:
            status_filter = "WHERE o.order_status NOT IN ('Cancelled', 'Returned')"

        query = f"""
        SELECT
            o.order_id,
            o.order_number,
            o.customer_id,
            c.customer_unique_id,
            c.first_name || ' ' || c.last_name AS customer_name,
            c.city AS customer_city,
            c.state AS customer_state,
            c.signup_date,
            c.acquisition_channel,
            o.order_date,
            o.order_status,
            o.payment_method,
            o.shipping_cost,
            o.total_amount
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        {status_filter}
        ORDER BY o.order_date ASC
        """
        df = pd.read_sql(query, self.engine)
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["signup_date"] = pd.to_datetime(df["signup_date"])
        df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
        df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0.0)
        return df

    def load_order_details(self) -> pd.DataFrame:
        """
        Load complete line-item transactional dataset joining orders, items,
        products, and customers for deep basket and category analysis.
        """
        query = """
        SELECT
            oi.item_id,
            oi.order_id,
            o.order_number,
            o.order_date,
            o.order_status,
            o.customer_id,
            c.first_name || ' ' || c.last_name AS customer_name,
            c.acquisition_channel,
            p.product_id,
            p.sku,
            p.product_name,
            p.category,
            p.sub_category,
            oi.quantity,
            oi.unit_price,
            oi.discount_amount,
            oi.total_price,
            (oi.total_price - (p.cost_price * oi.quantity)) AS gross_profit
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN products p ON oi.product_id = p.product_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_status NOT IN ('Cancelled', 'Returned')
        ORDER BY o.order_date ASC
        """
        df = pd.read_sql(query, self.engine)
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
        return df

    def load_cohort_dataset(self) -> pd.DataFrame:
        """
        Prepares a minimal, highly performant dataset for cohort retention calculations.
        Contains: customer_id, order_id, order_date, total_amount.
        """
        query = """
        SELECT
            customer_id,
            order_id,
            order_date,
            total_amount
        FROM orders
        WHERE order_status NOT IN ('Cancelled', 'Returned')
        ORDER BY customer_id, order_date ASC
        """
        df = pd.read_sql(query, self.engine)
        df["order_date"] = pd.to_datetime(df["order_date"])
        return df
