"""
Mock Data Generator for Retail Pulse Analytics Engine.

Generates statistically realistic e-commerce transactional data adhering to real-world
behavioral dynamics:
- Pareto (80/20) repeat-purchase power laws (Champions vs. one-time shoppers)
- Temporal cohort distribution spanning 24 months
- Realistic product margins, categories, and item baskets
- Seasonal velocity surges (Q4 holiday spikes, summer promos)
"""

import argparse
import datetime
import math
import os
import random
import sqlite3
from typing import Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Domain Data Pools
# -----------------------------------------------------------------------------
FIRST_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Oliver", "Sophia", "Elijah",
    "Isabella", "James", "Mia", "William", "Harper", "Benjamin", "Evelyn",
    "Lucas", "Amelia", "Henry", "Abigail", "Alexander", "Emily", "Sebastian",
    "Elizabeth", "Jack", "Mila", "Daniel", "Ella", "Matthew", "Avery", "Samuel",
    "Camila", "David", "Aria", "Joseph", "Scarlett", "Carter", "Victoria",
    "Owen", "Madison", "Wyatt", "Luna", "John", "Grace", "Jack", "Chloe", "Luke"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright"
]

CITIES_STATES = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"),
    ("Houston", "TX"), ("Phoenix", "AZ"), ("Philadelphia", "PA"),
    ("San Antonio", "TX"), ("San Diego", "CA"), ("Dallas", "TX"),
    ("Austin", "TX"), ("San Jose", "CA"), ("San Francisco", "CA"),
    ("Seattle", "WA"), ("Denver", "CO"), ("Boston", "MA"),
    ("Atlanta", "GA"), ("Miami", "FL"), ("Portland", "OR"),
    ("Minneapolis", "MN"), ("Charlotte", "NC")
]

ACQUISITION_CHANNELS = [
    ("Organic Search", 0.32),
    ("Paid Social", 0.28),
    ("Email Marketing", 0.18),
    ("Direct", 0.14),
    ("Referral", 0.08),
]

PAYMENT_METHODS = ["Credit Card", "PayPal", "Apple Pay", "Debit Card", "Klarna"]

ORDER_STATUSES = [
    ("Delivered", 0.88),
    ("Shipped", 0.05),
    ("Processing", 0.03),
    ("Returned", 0.02),
    ("Cancelled", 0.02),
]

PRODUCT_CATALOG = [
    # Electronics & Gadgets
    ("ELC-001", "Pulse Pro Wireless Noise-Cancelling Headphones", "Electronics", "Audio", 199.99, 85.00),
    ("ELC-002", "Aura True Wireless Sport Earbuds", "Electronics", "Audio", 89.99, 32.00),
    ("ELC-003", "Apex Mechanical Gaming Keyboard RGB", "Electronics", "Peripherals", 129.50, 52.00),
    ("ELC-004", "Vivid 27-inch 4K UHD IPS Monitor", "Electronics", "Displays", 349.99, 175.00),
    ("ELC-005", "Orbit Smart Fitness Tracker & Watch", "Electronics", "Wearables", 149.00, 58.00),
    ("ELC-006", "HyperCharge 65W GaN Fast Charger", "Electronics", "Accessories", 39.99, 12.50),

    # Apparel & Fashion
    ("APP-001", "Merino Wool Crewneck Minimalist Sweater", "Apparel", "Knitwear", 110.00, 42.00),
    ("APP-002", "Everyday Selvedge Stretch Denim Jacket", "Apparel", "Outerwear", 135.00, 55.00),
    ("APP-003", "CloudStrider Breathable Running Shoes", "Apparel", "Footwear", 125.00, 48.00),
    ("APP-004", "Heavyweight Organic Cotton Boxy Tee", "Apparel", "Tops", 34.00, 11.00),
    ("APP-005", "All-Weather Technical Commuter Parka", "Apparel", "Outerwear", 220.00, 92.00),
    ("APP-006", "High-Rise Compressive Performance Leggings", "Apparel", "Activewear", 78.00, 26.00),

    # Home & Kitchen
    ("HMK-001", "Artisan Precision Pour-Over Kettle & Carafe", "Home & Kitchen", "Coffee", 65.00, 24.00),
    ("HMK-002", "Pre-Seasoned 12-inch Cast Iron Skillet", "Home & Kitchen", "Cookware", 48.00, 19.00),
    ("HMK-003", "Ergonomic Lumbar High-Back Mesh Desk Chair", "Home & Kitchen", "Furniture", 280.00, 120.00),
    ("HMK-004", "Aromatherapy Botanical Soy Candle Set", "Home & Kitchen", "Home Decor", 36.00, 12.00),
    ("HMK-005", "QuietAir True HEPA Room Air Purifier", "Home & Kitchen", "Appliances", 159.00, 68.00),

    # Beauty & Personal Care
    ("BTY-001", "HydraGlow 2% Hyaluronic Acid Dewy Serum", "Beauty", "Skincare", 42.00, 11.50),
    ("BTY-002", "Restorative Botanical Rosemary Scalp Oil", "Beauty", "Haircare", 28.00, 7.80),
    ("BTY-003", "Daily Mineral Broad Spectrum Sunscreen SPF 50", "Beauty", "Skincare", 32.00, 9.00),
    ("BTY-004", "Nourishing Shea Butter Lip Therapy Kit", "Beauty", "Lipcare", 18.00, 4.50),

    # Sports & Outdoors
    ("SPT-001", "EcoGrip Non-Slip Natural Rubber Yoga Mat", "Sports", "Fitness", 68.00, 25.00),
    ("SPT-002", "Summit Double-Wall Vacuum Insulated Bottle 32oz", "Sports", "Hydration", 35.00, 10.50),
    ("SPT-003", "Ultralight 35L Ripstop Trail Daypack", "Sports", "Outdoors", 95.00, 38.00),
    ("SPT-004", "Variable Resistance Latex Loop Bands 5-Pack", "Sports", "Fitness", 24.00, 6.20),
]


def weighted_choice(choices_with_weights: List[Tuple[str, float]], rng: random.Random) -> str:
    """Return a choice based on weighted probabilities."""
    items, weights = zip(*choices_with_weights)
    return rng.choices(items, weights=weights, k=1)[0]


def init_database(db_path: str, schema_path: Optional[str] = None) -> None:
    """Initialize SQLite database using schema.sql DDL."""
    if schema_path is None:
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at {schema_path}")

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)
        conn.commit()


def generate_mock_dataset(
    db_path: str,
    num_customers: int = 1000,
    target_orders: int = 4500,
    start_date: datetime.date = datetime.date(2023, 1, 1),
    end_date: datetime.date = datetime.date(2024, 12, 31),
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Populate database with synthetic e-commerce customers, products, orders, and order items.
    
    Adheres to:
    - Multi-cohort customer acquisition across 24 monthly tranches.
    - Power-law repeat frequency (long tail distribution).
    - Realistic Q4 seasonality multipliers.
    """
    rng = random.Random(seed)
    
    # Initialize fresh schema
    init_database(db_path)

    total_days = (end_date - start_date).days

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # -------------------------------------------------------------------------
    # 1. Insert Products
    # -------------------------------------------------------------------------
    if verbose:
        print(f"[*] Populating {len(PRODUCT_CATALOG)} catalog products...")

    product_records = []
    for sku, name, cat, sub_cat, price, cost in PRODUCT_CATALOG:
        product_records.append((sku, name, cat, sub_cat, price, cost, 1))

    cur.executemany(
        """
        INSERT INTO products (sku, product_name, category, sub_category, unit_price, cost_price, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        product_records
    )

    # Retrieve auto-incremented product IDs and prices
    cur.execute("SELECT product_id, unit_price FROM products")
    db_products = cur.fetchall()  # List of (product_id, unit_price)

    # -------------------------------------------------------------------------
    # 2. Insert Customers
    # -------------------------------------------------------------------------
    if verbose:
        print(f"[*] Generating {num_customers} synthetic customer profiles...")

    customer_records = []
    customer_signups = {}  # cust_id -> datetime

    # Assign customer cohorts across the simulation timeline
    # Slight upward ramp in acquisition velocity over time (e-commerce business growth)
    for cust_id in range(1, num_customers + 1):
        # Biased signup date towards later months to simulate organic business scaling
        day_offset = int(math.pow(rng.random(), 0.85) * (total_days - 30))
        signup_dt = datetime.datetime.combine(
            start_date + datetime.timedelta(days=day_offset),
            datetime.time(
                hour=rng.randint(6, 23),
                minute=rng.randint(0, 59),
                second=rng.randint(0, 59)
            )
        )
        customer_signups[cust_id] = signup_dt

        first_name = rng.choice(FIRST_NAMES)
        last_name = rng.choice(LAST_NAMES)
        domain = rng.choice(["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "proton.me"])
        email = f"{first_name.lower()}.{last_name.lower()}{rng.randint(10, 999)}@{domain}"
        city, state = rng.choice(CITIES_STATES)
        channel = weighted_choice(ACQUISITION_CHANNELS, rng)
        unique_uuid = f"CUST-{cust_id:05d}-{rng.randint(1000, 9999)}"

        customer_records.append((
            cust_id,
            unique_uuid,
            first_name,
            last_name,
            email,
            city,
            state,
            "United States",
            signup_dt.strftime("%Y-%m-%d %H:%M:%S"),
            channel
        ))

    cur.executemany(
        """
        INSERT INTO customers (
            customer_id, customer_unique_id, first_name, last_name, email,
            city, state, country, signup_date, acquisition_channel
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        customer_records
    )

    # -------------------------------------------------------------------------
    # 3. Simulate Orders with Realistic Repeat Rates & Cohort Decay
    # -------------------------------------------------------------------------
    if verbose:
        print(f"[*] Generating realistic orders with repeat-purchase dynamics...")

    # Assign each customer a behavioral tier:
    # - Champions/Power (5%): 6 to 18 orders
    # - Loyalists/Engaged (15%): 3 to 7 orders
    # - Casual Repeat (20%): 2 orders
    # - One-Time Buyers (60%): 1 order
    tier_distribution = rng.choices(
        ["champion", "loyalist", "casual", "one_time"],
        weights=[0.06, 0.16, 0.22, 0.56],
        k=num_customers
    )

    order_records = []
    order_item_records = []
    order_counter = 1

    for cust_id in range(1, num_customers + 1):
        tier = tier_distribution[cust_id - 1]
        signup_dt = customer_signups[cust_id]

        if tier == "champion":
            num_orders = rng.randint(6, 16)
        elif tier == "loyalist":
            num_orders = rng.randint(3, 6)
        elif tier == "casual":
            num_orders = 2
        else:
            num_orders = 1

        # First purchase within 0-10 days of signup
        first_order_dt = signup_dt + datetime.timedelta(
            days=rng.randint(0, 10),
            hours=rng.randint(1, 12),
            minutes=rng.randint(0, 59)
        )
        if first_order_dt.date() > end_date:
            continue

        current_order_dt = first_order_dt

        for order_idx in range(num_orders):
            if current_order_dt.date() > end_date:
                break

            # Calculate seasonal boost: November/December increase order likelihood and basket size
            month = current_order_dt.month
            is_q4 = month in [11, 12]

            status = weighted_choice(ORDER_STATUSES, rng)
            payment = rng.choice(PAYMENT_METHODS)
            order_num = f"ORD-{current_order_dt.year}-{order_counter:06d}"

            # Items in basket: 1 to 4 items (Q4 slightly larger)
            num_items = rng.choices([1, 2, 3, 4], weights=[0.55, 0.28, 0.12, 0.05] if not is_q4 else [0.40, 0.35, 0.18, 0.07])[0]
            sampled_products = rng.sample(db_products, k=min(num_items, len(db_products)))

            order_subtotal = 0.0
            order_items_temp = []

            for prod_id, unit_price in sampled_products:
                qty = rng.choices([1, 2, 3], weights=[0.82, 0.14, 0.04])[0]
                # Discount: mostly 0, occasionally 10%, 15%, 20%
                discount_pct = rng.choices([0.0, 0.10, 0.15, 0.20], weights=[0.70, 0.15, 0.10, 0.05])[0]
                item_discount = round(unit_price * qty * discount_pct, 2)
                item_total = round((unit_price * qty) - item_discount, 2)
                order_subtotal += item_total

                order_items_temp.append((
                    order_counter,
                    prod_id,
                    qty,
                    unit_price,
                    item_discount,
                    item_total
                ))

            # Free shipping over $75, otherwise $6.99
            shipping_cost = 0.0 if order_subtotal >= 75.0 else 6.99
            order_total = round(order_subtotal + shipping_cost, 2)

            order_records.append((
                order_counter,
                order_num,
                cust_id,
                current_order_dt.strftime("%Y-%m-%d %H:%M:%S"),
                status,
                payment,
                shipping_cost,
                order_total
            ))
            order_item_records.extend(order_items_temp)

            order_counter += 1

            # Next order inter-purchase interval: Gamma-like repeat cadence (20 to 90 days)
            # Champions have tighter cadence (15-45 days), others wider (35-120 days)
            if tier == "champion":
                interval_days = rng.randint(14, 48)
            elif tier == "loyalist":
                interval_days = rng.randint(25, 75)
            else:
                interval_days = rng.randint(35, 120)

            current_order_dt = current_order_dt + datetime.timedelta(
                days=interval_days,
                hours=rng.randint(-4, 4),
                minutes=rng.randint(0, 59)
            )

    cur.executemany(
        """
        INSERT INTO orders (
            order_id, order_number, customer_id, order_date,
            order_status, payment_method, shipping_cost, total_amount
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        order_records
    )

    cur.executemany(
        """
        INSERT INTO order_items (
            order_id, product_id, quantity, unit_price, discount_amount, total_price
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        order_item_records
    )

    conn.commit()
    conn.close()

    stats = {
        "customers": len(customer_records),
        "products": len(product_records),
        "orders": len(order_records),
        "order_items": len(order_item_records)
    }

    if verbose:
        print("[+] Mock database generation completed successfully!")
        print(f"    - Customers:   {stats['customers']:,}")
        print(f"    - Products:    {stats['products']:,}")
        print(f"    - Orders:      {stats['orders']:,}")
        print(f"    - Order Items: {stats['order_items']:,}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock e-commerce database for Retail Pulse Analytics.")
    parser.add_argument("--db-path", type=str, default="retail_pulse.db", help="Path to SQLite database file.")
    parser.add_argument("--customers", type=int, default=1000, help="Number of customer records.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    generate_mock_dataset(
        db_path=args.db_path,
        num_customers=args.customers,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
