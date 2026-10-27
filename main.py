"""
Retail Pulse Analytics Engine - Main Orchestrator.

Orchestrates the complete end-to-end data pipeline:
1. Database bootstrapping & realistic mock data generation.
2. Ingestion & relational validation via SQLAlchemy & Pandas.
3. RFM feature engineering, NumPy log transformations, quintile scoring & segmentation.
4. Unsupervised K-Means clustering & silhouette evaluation.
5. Cohort retention matrix calculation, churn tracking & LTV progression.
6. Generation and export of publication-grade visualization dashboards & analytical reports.
"""

import argparse
import os
import sys
import time
from typing import Dict

import pandas as pd

# Allow relative imports when running from within the project directory or workspace root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.mock_data import generate_mock_dataset
from analysis.loader import DataLoader
from analysis.rfm_engine import RFMEngine
from analysis.cohort_engine import CohortEngine
from visualization.plots import (
    plot_cohort_retention_heatmap,
    plot_rfm_distributions,
    plot_rfm_segments,
    plot_rfm_scatter_2d,
    plot_monthly_sales_trend,
    plot_cohort_ltv_curves,
    plot_executive_summary_dashboard,
)


def print_banner() -> None:
    banner = r"""
========================================================================================
   ____       _        _ _   ____        _             
  |  _ \ ___ | |_ __ _(_) | |  _ \ _   _| |___  ___    
  | |_) / _ \| __/ _` | | | | |_) | | | | / __|/ _ \   
  |  _ <  __/| || (_| | | | |  __/| |_| | \__ \  __/   
  |_| \_\___| \__\__,_|_|_| |_|    \__,_|_|___/\___|   
                                                        
  E-Commerce Sales, RFM Segmentation, & Cohort Retention Analytics Engine
========================================================================================
    """
    print(banner)


def format_currency(val: float) -> str:
    return f"${val:,.2f}"


def run_pipeline(
    db_path: str,
    export_dir: str,
    rebuild_db: bool = False,
    num_customers: int = 1000,
    n_clusters: int = 4,
    seed: int = 42,
    skip_plots: bool = False
) -> Dict[str, any]:
    """Execute the full Retail Pulse Analytics Pipeline."""
    start_time = time.time()
    os.makedirs(export_dir, exist_ok=True)
    csv_dir = os.path.join(export_dir, "data_exports")
    plots_dir = os.path.join(export_dir, "visualizations")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Database Initialization
    # -------------------------------------------------------------------------
    print(f"\n[STEP 1/5] Database Verification & Bootstrapping...")
    db_exists = os.path.exists(db_path)

    if not db_exists or rebuild_db:
        action = "Rebuilding" if rebuild_db else "Initializing new"
        print(f"[*] {action} SQLite database at '{db_path}'...")
        generate_mock_dataset(
            db_path=db_path,
            num_customers=num_customers,
            seed=seed,
            verbose=True
        )
    else:
        print(f"[+] Found existing database at '{db_path}'. Proceeding with existing data.")

    # -------------------------------------------------------------------------
    # Step 2: Data Ingestion & Relational Validation
    # -------------------------------------------------------------------------
    print(f"\n[STEP 2/5] Ingesting Transactional & Customer Dimensions...")
    loader = DataLoader(db_path=db_path)
    db_counts = loader.get_database_summary()
    print("    - Table Inventory:")
    for tbl, count in db_counts.items():
        print(f"      * {tbl.capitalize():<12}: {count:,} rows")

    clean_orders = loader.load_clean_transactions(exclude_cancelled_and_returned=True)
    total_revenue = clean_orders["total_amount"].sum()
    min_date = clean_orders["order_date"].min().strftime("%Y-%m-%d")
    max_date = clean_orders["order_date"].max().strftime("%Y-%m-%d")
    unique_cust = clean_orders["customer_id"].nunique()

    print(f"[+] Ingested {len(clean_orders):,} valid transactions from {min_date} to {max_date}")
    print(f"    - Active Purchasing Customers: {unique_cust:,}")
    print(f"    - Gross Net Revenue:           {format_currency(total_revenue)}")
    print(f"    - Overall Average Order Value: {format_currency(total_revenue / len(clean_orders))}")

    # -------------------------------------------------------------------------
    # Step 3: RFM Feature Engineering & Customer Segmentation
    # -------------------------------------------------------------------------
    print(f"\n[STEP 3/5] Executing RFM Segmentation & K-Means Clustering...")
    rfm_engine = RFMEngine(transactions_df=clean_orders)
    rfm_df = rfm_engine.calculate_rfm_metrics()
    rfm_engine.apply_log_transformations()
    rfm_engine.assign_rfm_scores()
    rfm_engine.segment_customers()
    rfm_df, sil_score = rfm_engine.fit_kmeans_clusters(n_clusters=n_clusters, random_state=seed)

    print(f"[+] Computed RFM metrics for {len(rfm_df):,} customers.")
    print(f"[+] Standardized Log-RFM K-Means clustering completed (k={n_clusters}, Silhouette Score: {sil_score:.4f})")

    segment_summary = rfm_engine.get_segment_summary()
    print("\n--- RFM Customer Segment Breakdown ---")
    header = f"{'Segment':<28} | {'Cust #':<7} | {'Base %':<7} | {'Revenue':<12} | {'Rev %':<7} | {'Avg Rec':<8} | {'Avg Freq':<8}"
    print(header)
    print("-" * len(header))
    for _, row in segment_summary.iterrows():
        print(
            f"{row['Segment']:<28} | "
            f"{row['Customer_Count']:<7,d} | "
            f"{row['Customer_Share_Pct']:<6.1f}% | "
            f"{format_currency(row['Total_Revenue']):<12} | "
            f"{row['Revenue_Share_Pct']:<6.1f}% | "
            f"{row['Avg_Recency_Days']:<7.1f}d | "
            f"{row['Avg_Frequency']:<7.1f}"
        )

    # Export RFM tables
    rfm_cust_path = os.path.join(csv_dir, "rfm_customer_profiles.csv")
    rfm_summary_path = os.path.join(csv_dir, "rfm_segment_summary.csv")
    rfm_df.to_csv(rfm_cust_path, index=False)
    segment_summary.to_csv(rfm_summary_path, index=False)
    print(f"[+] Exported Customer RFM Profiles -> {rfm_cust_path}")
    print(f"[+] Exported Segment Summary Report -> {rfm_summary_path}")

    # -------------------------------------------------------------------------
    # Step 4: Cohort Retention Analytics
    # -------------------------------------------------------------------------
    print(f"\n[STEP 4/5] Computing Cohort Retention & Churn Dynamics...")
    cohort_engine = CohortEngine(transactions_df=clean_orders)
    cohort_counts, retention_matrix, cohort_sizes = cohort_engine.compute_retention_matrix()
    revenue_matrix = cohort_engine.compute_revenue_matrix()
    arpu_matrix = cohort_engine.compute_cumulative_arpu_matrix()
    churn_matrix = cohort_engine.compute_churn_matrix()
    cohort_kpis = cohort_engine.get_retention_summary_kpis()

    print(f"[+] Computed retention across {len(retention_matrix)} monthly cohorts.")
    print(f"    - Average Month 1 Retention: {cohort_kpis.get('Avg_Month_1_Retention_Pct', 0.0):.1f}%")
    print(f"    - Average Month 3 Retention: {cohort_kpis.get('Avg_Month_3_Retention_Pct', 0.0):.1f}%")
    print(f"    - Average Month 6 Retention: {cohort_kpis.get('Avg_Month_6_Retention_Pct', 0.0):.1f}%")

    print("\n--- Cohort Retention Snapshot (Recent 6 Cohorts, First 6 Months) ---")
    recent_ret = retention_matrix.tail(6)[[c for c in [0, 1, 2, 3, 4, 5] if c in retention_matrix.columns]]
    print(recent_ret.to_string())

    # Export Cohort matrices
    retention_path = os.path.join(csv_dir, "cohort_retention_matrix_pct.csv")
    counts_path = os.path.join(csv_dir, "cohort_customer_counts.csv")
    revenue_path = os.path.join(csv_dir, "cohort_revenue_matrix.csv")
    arpu_path = os.path.join(csv_dir, "cohort_cumulative_arpu_ltv.csv")

    retention_matrix.to_csv(retention_path)
    cohort_counts.to_csv(counts_path)
    revenue_matrix.to_csv(revenue_path)
    arpu_matrix.to_csv(arpu_path)

    print(f"[+] Exported Cohort Retention Matrix -> {retention_path}")
    print(f"[+] Exported Cohort Cumulative ARPU  -> {arpu_path}")

    # -------------------------------------------------------------------------
    # Step 5: High-Resolution Visualizations
    # -------------------------------------------------------------------------
    if not skip_plots:
        print(f"\n[STEP 5/5] Generating High-Resolution Analytics Dashboards & Charts...")
        # 1. Cohort Retention Heatmap
        plot_cohort_retention_heatmap(
            retention_matrix=retention_matrix,
            cohort_sizes=cohort_sizes,
            save_path=os.path.join(plots_dir, "cohort_retention_heatmap.png")
        )

        # 2. RFM Distributions
        plot_rfm_distributions(
            rfm_df=rfm_df,
            save_path=os.path.join(plots_dir, "rfm_log_distributions.png")
        )

        # 3. RFM Customer Segments Pareto Breakdown
        plot_rfm_segments(
            rfm_df=rfm_df,
            save_path=os.path.join(plots_dir, "rfm_segment_pareto.png")
        )

        # 4. Behavioral 2D Scatter Map
        plot_rfm_scatter_2d(
            rfm_df=rfm_df,
            save_path=os.path.join(plots_dir, "rfm_behavioral_scatter.png")
        )

        # 5. Monthly Sales Trend
        plot_monthly_sales_trend(
            transactions_df=clean_orders,
            save_path=os.path.join(plots_dir, "monthly_sales_velocity.png")
        )

        # 6. Cohort LTV Progression Curves
        plot_cohort_ltv_curves(
            arpu_matrix=arpu_matrix,
            save_path=os.path.join(plots_dir, "cohort_ltv_curves.png")
        )

        # 7. Executive Summary Dashboard
        plot_executive_summary_dashboard(
            retention_matrix=retention_matrix,
            rfm_df=rfm_df,
            transactions_df=clean_orders,
            save_path=os.path.join(plots_dir, "executive_summary_dashboard.png")
        )
    else:
        print(f"\n[STEP 5/5] Skipping plots as requested (--no-plots).")

    elapsed = time.time() - start_time
    print(f"\n========================================================================================")
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print(f" All outputs exported to: {os.path.abspath(export_dir)}")
    print(f"========================================================================================\n")

    return {
        "total_revenue": total_revenue,
        "active_customers": unique_cust,
        "segment_summary": segment_summary,
        "cohort_kpis": cohort_kpis,
        "elapsed_seconds": elapsed
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retail Pulse: E-Commerce Sales, RFM Segmentation, & Cohort Retention Analytics Engine."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "database", "retail_pulse.db"),
        help="Path to SQLite database file."
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "reports"),
        help="Directory to save analytical charts and CSV reports."
    )
    parser.add_argument(
        "--rebuild-db",
        action="store_true",
        help="Rebuild SQLite schema and re-generate mock dataset."
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=1000,
        help="Number of mock customers to generate if rebuilding database."
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=4,
        help="Number of K-Means clusters for RFM analysis."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible mock generation and clustering."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable chart rendering (useful for headless CI or data-only runs)."
    )

    args = parser.parse_args()

    run_pipeline(
        db_path=args.db_path,
        export_dir=args.export_dir,
        rebuild_db=args.rebuild_db,
        num_customers=args.customers,
        n_clusters=args.clusters,
        seed=args.seed,
        skip_plots=args.no_plots
    )


if __name__ == "__main__":
    main()
