-- Retail Pulse Analytics Engine: Relational Schema DDL
-- Database dialect: SQLite 3

PRAGMA foreign_keys = ON;

-- Drop existing tables to ensure clean rebuilds when needed
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

-- -----------------------------------------------------------------------------
-- 1. Customers Dimension Table
-- -----------------------------------------------------------------------------
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_unique_id TEXT NOT NULL UNIQUE,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    country TEXT NOT NULL DEFAULT 'United States',
    signup_date TIMESTAMP NOT NULL,
    acquisition_channel TEXT NOT NULL
);

-- -----------------------------------------------------------------------------
-- 2. Products Dimension Table
-- -----------------------------------------------------------------------------
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT NOT NULL UNIQUE,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    sub_category TEXT,
    unit_price REAL NOT NULL CHECK(unit_price >= 0),
    cost_price REAL NOT NULL CHECK(cost_price >= 0),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0, 1))
);

-- -----------------------------------------------------------------------------
-- 3. Orders Fact Table
-- -----------------------------------------------------------------------------
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_number TEXT NOT NULL UNIQUE,
    customer_id INTEGER NOT NULL,
    order_date TIMESTAMP NOT NULL,
    order_status TEXT NOT NULL CHECK(order_status IN ('Delivered', 'Shipped', 'Processing', 'Cancelled', 'Returned')),
    payment_method TEXT NOT NULL,
    shipping_cost REAL NOT NULL DEFAULT 0.0,
    total_amount REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- -----------------------------------------------------------------------------
-- 4. Order Items Fact Table
-- -----------------------------------------------------------------------------
CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    unit_price REAL NOT NULL CHECK(unit_price >= 0),
    discount_amount REAL NOT NULL DEFAULT 0.0 CHECK(discount_amount >= 0),
    total_price REAL NOT NULL CHECK(total_price >= 0),
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- -----------------------------------------------------------------------------
-- Performance Indexes for Analytical Queries
-- -----------------------------------------------------------------------------
CREATE INDEX idx_customers_signup ON customers(signup_date);
CREATE INDEX idx_customers_channel ON customers(acquisition_channel);
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);
CREATE INDEX idx_orders_status ON orders(order_status);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
