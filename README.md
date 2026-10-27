# Retail Pulse: E-Commerce Sales, RFM Segmentation, & Cohort Retention Analytics Engine

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Database](https://img.shields.io/badge/SQLite%203-SQLAlchemy%202.0-003B57.svg?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-K--Means%20Clustering-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-2ECC71.svg?style=for-the-badge)](LICENSE)

> **In direct-to-consumer (D2C) e-commerce, soaring customer acquisition costs (CAC) make repeat retention and customer lifetime value (LTV) the primary arbiters of profitability.**  
> **Retail Pulse** is an enterprise-grade customer intelligence and analytics engine that transforms raw transactional databases into actionable growth strategies through log-transformed **RFM Customer Segmentation**, unsupervised **K-Means Clustering**, and triangular **Monthly Cohort Retention Analysis**.

---

## Executive Summary Dashboard

![Executive Summary Dashboard](./assets/executive_summary_dashboard.png)

---

## System Architecture

Retail Pulse operates on a clean, decoupled 3-tier analytical pipeline:

```text
┌────────────────────────────────┐
│   Relational Data Layer (3NF)  │  SQLite 3 / SQLAlchemy ORM
│  • Customers, Products, Orders │  • Analytical composite indexes
│  • Line-item Order Details     │  • Statistical Pareto mock simulator
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│  High-Performance ETL Engine   │  Pandas & NumPy
│  • Skew-corrected RFM modeling │  • log1p variance stabilization
│  • Rank-based quintile scoring │  • Unsupervised K-Means clustering
│  • Triangular cohort matrices  │  • Retention, churn, & LTV curves
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│  Executive Presentation Suite  │  Matplotlib & Seaborn (300 DPI)
│  • Cohort retention heatmaps   │  • Behavioral scatter landscapes
│  • Pareto revenue disparity    │  • Multi-panel executive dashboards
└────────────────────────────────┘
```

---

## Key Technical Highlights

### 1. Relational Data Modeling & 3NF Schema
Built on a third-normal-form (3NF) relational design representing enterprise e-commerce entities:
- **`customers` Dimension**: Demographics, acquisition channels (`Organic Search`, `Paid Social`, `Email`, etc.), and signup timestamps.
- **`products` Dimension**: SKU, multi-level hierarchy, unit cost, unit price, and gross profit margin.
- **`orders` & `order_items` Facts**: Granular transactional ledger with status verification (`Delivered`, `Shipped`, `Returned`, `Cancelled`), line-level basket items, and promotional discounts.
- **Synthetic Data Generator**: Simulates realistic e-commerce mechanics including the 80/20 Pareto repeat-order distribution, seasonal Q4 velocity surges, and gamma-distributed repurchase intervals over a 24-month horizon.

---

### 2. Skew-Corrected RFM Segmentation & K-Means Clustering

Raw e-commerce transaction data suffers from extreme positive skewness: a tiny minority of VIP "Champions" purchase dozens of times and spend thousands of dollars, while the vast majority are one-time shoppers.

![RFM Log Transformations](./assets/rfm_log_distributions.png)

#### Skewness Correction via NumPy `log1p`
To eliminate outlier distortion in Euclidean space, features are normalized using natural logarithmic scaling:
$$x_{\text{transformed}} = \ln(x + 1) = \text{np.log1p}(x)$$
This stabilizes feature variance, compresses long right-tail outliers, and enables balanced cluster discovery.

#### Collision-Resistant Quintile Scoring
Because a high percentage of shoppers share identical low purchase counts ($F = 1$), traditional `pd.qcut` throws duplicate bin-edge errors. Retail Pulse employs **first-occurrence ranking** before binning:
$$R_{\text{Score}}, F_{\text{Score}}, M_{\text{Score}} \in [1, 5]$$
- **Recency ($R$)**: $5 = \text{Most recent}$, $1 = \text{Longest lapse}$.
- **Frequency ($F$)**: $5 = \text{Highest repeat orders}$, $1 = \text{Single order}$.
- **Monetary ($M$)**: $5 = \text{Top spenders}$, $1 = \text{Lowest spenders}$.

---

### 3. Customer Segments & Pareto Revenue Share

![RFM Segment Pareto Breakdown](./assets/rfm_segment_pareto.png)

Customers are categorized into 10 tactical CRM segments with strategic playbooks:

| Customer Segment | Base Share % | Revenue Share % | Average Spend | Tactical Retention Playbook |
| :--- | :---: | :---: | :---: | :--- |
| **Champions** | **19.9%** | **38.9%** | **$884.89** | VIP loyalty programs, early access to drops, dedicated concierge |
| **Loyal Customers** | **17.8%** | **19.0%** | **$483.77** | Cross-sell and upsell premium lines, solicit verified product reviews |
| **At Risk** | 17.0% | 12.4% | $329.28 | Time-sensitive win-back incentives, survey dissatisfaction causes |
| **Lost** | 3.9% | 8.9% | $1,038.07 | Single high-discount reactivation push; purge unengaged emails |
| **Hibernating** | 10.3% | 4.8% | $211.89 | Low-cost seasonal broadcasts; suppress from paid PPC retargeting |
| **Promising** | 9.7% | 4.0% | $184.64 | Curated onboarding recommendations, welcome back discounts |
| **Recent Customers** | 5.8% | 3.0% | $236.66 | 3-part post-purchase nurture series, incentive for 2nd purchase |

```
Key Finding: Champions and Loyalists account for 37.7% of customers but generate 57.9% of gross revenue.
```

---

### 4. 2D Behavioral Landscape

![Customer Behavioral Landscape](./assets/rfm_behavioral_scatter.png)

This behavioral map plots customer **Recency vs. Frequency**, with bubble size proportional to total monetary spend. It visually distinguishes between active repeat buyers (upper left), slipping high-value accounts (upper right), and churned one-time buyers (bottom right).

---

### 5. Monthly Cohort Retention Matrix

![Cohort Retention Heatmap](./assets/cohort_retention_heatmap.png)

Customers are partitioned into monthly acquisition cohorts ($M_0$). The engine tracks active return rates across lifecycle months:
$$\text{Retention Rate}(c, k) = \frac{\text{Unique Active Customers}(c, k)}{\text{Initial Cohort Size } S_c} \times 100\%$$

- **Month-1 Return Baseline**: Highlights early drop-off and onboarding effectiveness.
- **Retention Plateau**: Identifies the month where customer decay stabilizes into steady-state brand loyalty (typically Months 3–5).

---

### 6. Cumulative Customer Lifetime Value (LTV / ARPU)

![Cohort LTV Curves](./assets/cohort_ltv_curves.png)

Tracks the cumulative revenue trajectory generated per acquired customer as cohorts mature over 12–24 months:
$$\text{Cumulative ARPU}(c, k) = \frac{\sum_{t=0}^{k} \text{Gross Revenue}(c, t)}{S_c}$$

---

### 7. Monthly Sales Velocity & Growth Trends

![Monthly Sales Velocity](./assets/monthly_sales_velocity.png)

Provides macro-level business performance tracking monthly gross revenues, trailing 3-month moving averages, and transaction volume trends across the full e-commerce lifecycle.

---

## Project Directory Layout

```text
├── database/
│   ├── __init__.py           # Package exports for DB initialization
│   ├── schema.sql            # SQLite DDL: 3NF Customers, Products, Orders, Order Items, & Indexes
│   ├── mock_data.py          # Synthetic e-commerce transactional data generator
│   └── retail_pulse.db       # Relational database file
│
├── analysis/
│   ├── __init__.py           # Analytics package exports
│   ├── loader.py             # SQLAlchemy connector & Pandas data ingestion
│   ├── rfm_engine.py         # RFM feature engineering, log transforms, & K-Means clustering
│   └── cohort_engine.py      # Monthly retention matrices, churn curves, & LTV calculations
│
├── visualization/
│   ├── __init__.py           # Visualization suite exports
│   └── plots.py              # Matplotlib & Seaborn high-res plotting suite (300 DPI)
│
├── assets/                   # High-res figures displayed in documentation
├── reports/                  # Generated analytics outputs
│   ├── data_exports/         # Processed CSV matrices and customer profile tables
│   └── visualizations/       # High-resolution charts & executive summary dashboard
│
├── main.py                   # Orchestrator CLI runner
├── requirements.txt          # Production dependencies
└── README.md                 # Complete documentation
```

---

## Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/ayushi0618/Data-Analytics-Project-.git
cd Data-Analytics-Project-
```

### 2. Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Analytics Engine
```bash
python main.py
```

To regenerate the database with a custom customer volume and seed:
```bash
python main.py --rebuild-db --customers 1500 --clusters 4 --seed 42
```

---

## CLI Options

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--db-path` | `str` | `database/retail_pulse.db` | Path to SQLite database file |
| `--export-dir` | `str` | `reports` | Target directory for generated charts and CSV exports |
| `--rebuild-db` | `flag` | `False` | Drop existing database and recreate synthetic transactions |
| `--customers` | `int` | `1000` | Number of customer profiles to synthesize if rebuilding |
| `--clusters` | `int` | `4` | Number of K-Means clusters ($k$) |
| `--seed` | `int` | `42` | Random seed for deterministic reproducibility |
| `--no-plots` | `flag` | `False` | Skip chart generation for headless/data-only runs |

---

## Production Deployment & Enterprise Expansion

- **Cloud Data Warehouses**: Swap SQLite connection strings in [`analysis/loader.py`](file:///analysis/loader.py) with Google BigQuery (`google-cloud-bigquery`), Snowflake (`snowflake-sqlalchemy`), or Amazon Redshift.
- **Workflow Orchestration**: Package [`main.py`](file:///main.py) inside an **Apache Airflow** or **Prefect** DAG to trigger daily or weekly batch refreshes.
- **BI Layer Integration**: Ingest the exported CSV files in `reports/data_exports/` directly into **Tableau**, **Power BI**, or **Looker** for self-serve exploration.

---

## License
Distributed under the MIT License. See `LICENSE` for more information.
