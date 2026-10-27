"""
Data Visualization Suite for Retail Pulse Analytics Engine.

Provides high-resolution, publication-ready figures for:
1. Cohort Retention Heatmap with percentage annotations and cohort sizes.
2. RFM Raw vs. Log-Transformed Feature Distributions.
3. RFM Customer Segments: Customer Count vs. Revenue Share Pareto breakdown.
4. 2D Behavioral Map: Recency vs. Frequency with Monetary bubble weighting.
5. Monthly E-Commerce Sales, Order Volume, and AOV Trends.
6. Cumulative Cohort LTV / ARPU Curves.
7. Unified Executive Summary Dashboard (1920x1080 / 300 DPI).
"""

import os
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless & production environments
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


# -----------------------------------------------------------------------------
# Design Theme & Color Palettes
# -----------------------------------------------------------------------------
PALETTE_SEGMENTS = {
    "Champions": "#1b9e77",
    "Loyal Customers": "#2b5c8f",
    "Potential Loyalists": "#386cb0",
    "Recent Customers": "#41b6c4",
    "Promising": "#7fcdbb",
    "Customers Needing Attention": "#e6ab02",
    "About to Sleep": "#fd8d3c",
    "At Risk": "#e7298a",
    "Can't Lose Them": "#d95f02",
    "Hibernating": "#7570b3",
    "Lost": "#999999"
}

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Lucida Grande"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#ebebeb",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})


def _ensure_dir(filepath: str) -> None:
    """Ensure directory for filepath exists."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)


def plot_cohort_retention_heatmap(
    retention_matrix: pd.DataFrame,
    cohort_sizes: Optional[pd.Series] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Renders an e-commerce triangular Cohort Retention Heatmap.
    
    Args:
        retention_matrix: DataFrame of retention percentages (cohorts as rows, months as cols).
        cohort_sizes: Series of Month 0 customer counts for row annotations.
        save_path: Optional output image file path.
        figsize: Figure dimensions.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Format y-axis labels with initial cohort size if provided
    y_labels = []
    for cohort in retention_matrix.index:
        if cohort_sizes is not None and cohort in cohort_sizes.index:
            size = int(cohort_sizes.loc[cohort])
            y_labels.append(f"{cohort}  (n={size:,})")
        else:
            y_labels.append(str(cohort))

    # Mask upper triangle NaN values cleanly
    mask = retention_matrix.isnull()

    # Draw Heatmap with modern YlGnBu colormap
    cbar_kws = {
        "label": "Retention Rate (%)",
        "shrink": 0.75,
        "aspect": 20,
        "format": "%.0f%%"
    }

    sns.heatmap(
        retention_matrix,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=100.0,
        linewidths=0.5,
        linecolor="#ffffff",
        cbar_kws=cbar_kws,
        ax=ax,
        annot_kws={"size": 8, "weight": "normal"}
    )

    ax.set_yticklabels(y_labels, rotation=0, fontsize=9)
    ax.set_xticklabels([f"Month {col}" for col in retention_matrix.columns], rotation=0, fontsize=9)
    ax.set_title(
        "Monthly Cohort Customer Retention Matrix (% Active Return Rate)",
        fontsize=14,
        fontweight="bold",
        pad=16,
        color="#222222"
    )
    ax.set_xlabel("Cohort Lifecycle (Months Elapsed Since First Purchase)", fontsize=10, labelpad=10)
    ax.set_ylabel("Customer Acquisition Cohort", fontsize=10, labelpad=10)

    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved Cohort Retention Heatmap -> {save_path}")

    return fig


def plot_rfm_distributions(
    rfm_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Subplot grid contrasting raw skewed RFM distributions with log1p-transformed distributions.
    
    Demonstrates variance stabilization and normality transformation for machine learning.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=300)

    # 1. Raw Distributions (Top Row)
    # Recency
    sns.histplot(rfm_df["Recency"], kde=True, ax=axes[0, 0], color="#2b5c8f", bins=30)
    axes[0, 0].set_title(f"Raw Recency (Days)\nSkewness: {rfm_df['Recency'].skew():.2f}", fontsize=10, fontweight="bold")
    axes[0, 0].set_xlabel("Days Since Last Order")

    # Frequency
    sns.histplot(rfm_df["Frequency"], kde=True, ax=axes[0, 1], color="#1b9e77", bins=20)
    axes[0, 1].set_title(f"Raw Frequency (Orders)\nSkewness: {rfm_df['Frequency'].skew():.2f}", fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel("Order Count")

    # Monetary
    sns.histplot(rfm_df["Monetary"], kde=True, ax=axes[0, 2], color="#d95f02", bins=30)
    axes[0, 2].set_title(f"Raw Monetary Spend ($)\nSkewness: {rfm_df['Monetary'].skew():.2f}", fontsize=10, fontweight="bold")
    axes[0, 2].set_xlabel("Total Spend ($)")
    axes[0, 2].xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))

    # 2. Log-Transformed Distributions (Bottom Row)
    # Recency_log
    sns.histplot(rfm_df["Recency_log"], kde=True, ax=axes[1, 0], color="#41b6c4", bins=30)
    axes[1, 0].set_title(f"Log Recency: ln(R + 1)\nSkewness: {rfm_df['Recency_log'].skew():.2f}", fontsize=10, fontweight="bold")
    axes[1, 0].set_xlabel("ln(Recency + 1)")

    # Frequency_log
    sns.histplot(rfm_df["Frequency_log"], kde=True, ax=axes[1, 1], color="#7fcdbb", bins=20)
    axes[1, 1].set_title(f"Log Frequency: ln(F + 1)\nSkewness: {rfm_df['Frequency_log'].skew():.2f}", fontsize=10, fontweight="bold")
    axes[1, 1].set_xlabel("ln(Frequency + 1)")

    # Monetary_log
    sns.histplot(rfm_df["Monetary_log"], kde=True, ax=axes[1, 2], color="#fd8d3c", bins=30)
    axes[1, 2].set_title(f"Log Monetary: ln(M + 1)\nSkewness: {rfm_df['Monetary_log'].skew():.2f}", fontsize=10, fontweight="bold")
    axes[1, 2].set_xlabel("ln(Monetary + 1)")

    fig.suptitle(
        "RFM Feature Engineering: Power-Law Skewness Correction via NumPy Log Transforms",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved RFM Distributions Chart -> {save_path}")

    return fig


def plot_rfm_segments(
    rfm_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Renders dual Pareto comparison of Customer Share vs. Revenue Contribution by Segment.
    """
    summary = rfm_df.groupby("Segment").agg(
        Customers=("customer_id", "count"),
        Revenue=("Monetary", "sum")
    ).reset_index()

    total_customers = summary["Customers"].sum()
    total_rev = summary["Revenue"].sum()

    summary["Customer_Pct"] = summary["Customers"] / total_customers * 100
    summary["Revenue_Pct"] = summary["Revenue"] / total_rev * 100

    # Sort by Revenue descending
    summary = summary.sort_values(by="Revenue", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=300)

    colors = [PALETTE_SEGMENTS.get(seg, "#4a7bb0") for seg in summary["Segment"]]

    # Panel 1: Customer Volume
    bars1 = ax1.barh(summary["Segment"], summary["Customers"], color=colors, alpha=0.85, edgecolor="#ffffff")
    ax1.set_title("Customer Base Volume by Segment", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Number of Customers")
    ax1.grid(axis="x", linestyle="--", alpha=0.7)

    for bar, pct in zip(bars1, summary["Customer_Pct"]):
        width = bar.get_width()
        ax1.text(width + 5, bar.get_y() + bar.get_height() / 2, f"{int(width):,} ({pct:.1f}%)",
                 va="center", fontsize=8, color="#333333")

    # Panel 2: Revenue Contribution
    bars2 = ax2.barh(summary["Segment"], summary["Revenue"], color=colors, alpha=0.85, edgecolor="#ffffff")
    ax2.set_title("Total Revenue Generated ($) by Segment", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Gross Monetary Spend ($)")
    ax2.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax2.grid(axis="x", linestyle="--", alpha=0.7)

    for bar, pct in zip(bars2, summary["Revenue_Pct"]):
        width = bar.get_width()
        ax2.text(width + (total_rev * 0.01), bar.get_y() + bar.get_height() / 2, f"${width:,.0f} ({pct:.1f}%)",
                 va="center", fontsize=8, color="#333333")

    fig.suptitle(
        "RFM Customer Segmentation: Pareto Value & Volume Disparity",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved RFM Segments Chart -> {save_path}")

    return fig


def plot_rfm_scatter_2d(
    rfm_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (11, 7)
) -> plt.Figure:
    """
    2D Behavioral Scatter Map: Recency vs. Frequency, with bubble size proportional to Monetary spend.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Normalize bubble sizes: 20 to 300
    monetary = rfm_df["Monetary"].clip(lower=1)
    bubble_sizes = 20 + 280 * ((monetary - monetary.min()) / (monetary.max() - monetary.min() + 1e-5))

    segments = sorted(rfm_df["Segment"].unique())

    for seg in segments:
        subset = rfm_df[rfm_df["Segment"] == seg]
        sub_sizes = bubble_sizes.loc[subset.index]
        color = PALETTE_SEGMENTS.get(seg, "#333333")

        ax.scatter(
            subset["Recency"],
            subset["Frequency"],
            s=sub_sizes,
            label=seg,
            alpha=0.65,
            edgecolors="#ffffff",
            linewidth=0.5,
            c=color
        )

    ax.set_title(
        "Customer Behavioral Landscape (Recency vs. Frequency, Sized by Total Spend)",
        fontsize=13,
        fontweight="bold",
        pad=14
    )
    ax.set_xlabel("Recency (Days Since Last Order - Lower is More Recent)", fontsize=10)
    ax.set_ylabel("Frequency (Total Repeat Orders Placed)", fontsize=10)
    ax.set_ylim(bottom=0.5)

    # Place legend neatly outside
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        title="Customer Segments",
        title_fontsize=9,
        fontsize=8
    )

    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved RFM 2D Scatter Chart -> {save_path}")

    return fig


def plot_monthly_sales_trend(
    transactions_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Monthly Sales Velocity: Gross Revenue, Transaction Volume, and Average Order Value (AOV).
    """
    df = transactions_df.copy()
    df["order_month"] = pd.to_datetime(df["order_date"]).dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby("order_month").agg(
        Revenue=("total_amount", "sum"),
        Orders=("order_id", "nunique"),
        Customers=("customer_id", "nunique")
    ).reset_index()

    monthly["AOV"] = monthly["Revenue"] / monthly["Orders"]
    monthly["Rev_Rolling_3M"] = monthly["Revenue"].rolling(window=3, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=figsize, dpi=300)

    # Monthly revenue bars
    bars = ax1.bar(
        monthly["order_month"],
        monthly["Revenue"],
        width=20,
        color="#2b5c8f",
        alpha=0.6,
        label="Monthly Gross Revenue ($)"
    )
    # Rolling trend line
    ax1.plot(
        monthly["order_month"],
        monthly["Rev_Rolling_3M"],
        color="#d95f02",
        linewidth=2.5,
        marker="o",
        label="3-Month Trailing Revenue Trend"
    )

    ax1.set_title("E-Commerce Sales Velocity & Order Growth Trend", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Gross Revenue ($)", fontsize=10, color="#2b5c8f")
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax1.tick_params(axis="y", labelcolor="#2b5c8f")

    # Secondary axis for Order Volume
    ax2 = ax1.twinx()
    ax2.plot(
        monthly["order_month"],
        monthly["Orders"],
        color="#1b9e77",
        linewidth=2.0,
        linestyle="--",
        marker="s",
        markersize=5,
        label="Monthly Order Volume"
    )
    ax2.set_ylabel("Order Count", fontsize=10, color="#1b9e77")
    ax2.tick_params(axis="y", labelcolor="#1b9e77")
    ax2.grid(False)  # Avoid overlapping gridlines

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved Monthly Sales Trend -> {save_path}")

    return fig


def plot_cohort_ltv_curves(
    arpu_matrix: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Renders Cumulative Average Revenue Per User (Cumulative ARPU / LTV) across Cohort Age.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Plot top cohorts (first 10 cohorts to keep chart clean and readable)
    palette = sns.color_palette("tab10", n_colors=min(10, len(arpu_matrix)))

    for i, (cohort, row) in enumerate(arpu_matrix.head(10).iterrows()):
        valid_series = row.dropna()
        ax.plot(
            valid_series.index,
            valid_series.values,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=str(cohort),
            color=palette[i]
        )

    ax.set_title(
        "Cohort Cumulative Customer Lifetime Value (Cumulative ARPU Progression)",
        fontsize=13,
        fontweight="bold",
        pad=14
    )
    ax.set_xlabel("Cohort Lifecycle (Months Elapsed Since Acquisition)", fontsize=10)
    ax.set_ylabel("Cumulative Spend per Acquired User ($)", fontsize=10)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend(title="Cohort Month", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved Cohort LTV Curves -> {save_path}")

    return fig


def plot_executive_summary_dashboard(
    retention_matrix: pd.DataFrame,
    rfm_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (18, 12)
) -> plt.Figure:
    """
    Assembles a high-level 4-panel executive analytics dashboard suitable for C-suite reporting.
    """
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25)

    # Top-Left: Monthly Sales Velocity
    ax1 = fig.add_subplot(gs[0, 0])
    df_m = transactions_df.copy()
    df_m["order_month"] = pd.to_datetime(df_m["order_date"]).dt.to_period("M").dt.to_timestamp()
    monthly = df_m.groupby("order_month")["total_amount"].sum()
    ax1.plot(monthly.index, monthly.values, marker="o", color="#2b5c8f", linewidth=2.2)
    ax1.fill_between(monthly.index, monthly.values, color="#2b5c8f", alpha=0.15)
    ax1.set_title("1. Monthly Revenue Trajectory ($)", fontsize=11, fontweight="bold")
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax1.set_ylabel("Gross Spend ($)")

    # Top-Right: Segment Revenue Pareto Share
    ax2 = fig.add_subplot(gs[0, 1])
    seg_rev = rfm_df.groupby("Segment")["Monetary"].sum().sort_values(ascending=True)
    colors = [PALETTE_SEGMENTS.get(s, "#333333") for s in seg_rev.index]
    ax2.barh(seg_rev.index, seg_rev.values, color=colors, alpha=0.85)
    ax2.set_title("2. Revenue Contribution by RFM Segment ($)", fontsize=11, fontweight="bold")
    ax2.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))

    # Bottom-Left: Cohort Retention (Recent 8 Cohorts, First 6 Months)
    ax3 = fig.add_subplot(gs[1, 0])
    recent_cohorts = retention_matrix.tail(8)
    sub_cols = [c for c in [0, 1, 2, 3, 4, 5, 6] if c in recent_cohorts.columns]
    recent_ret = recent_cohorts[sub_cols]
    sns.heatmap(
        recent_ret,
        mask=recent_ret.isnull(),
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar=False,
        ax=ax3,
        annot_kws={"size": 8}
    )
    ax3.set_title("3. Retention Matrix Snapshot (Recent Cohorts %)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Months Elapsed")
    ax3.set_ylabel("Cohort")

    # Bottom-Right: Behavioral RFM Distribution (Recency vs. Frequency)
    ax4 = fig.add_subplot(gs[1, 1])
    for seg in sorted(rfm_df["Segment"].unique()):
        sub = rfm_df[rfm_df["Segment"] == seg]
        ax4.scatter(
            sub["Recency"],
            sub["Frequency"],
            label=seg,
            alpha=0.6,
            s=25,
            c=PALETTE_SEGMENTS.get(seg, "#333333")
        )
    ax4.set_title("4. Recency vs. Frequency Clustering", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Recency (Days)")
    ax4.set_ylabel("Repeat Frequency")
    ax4.legend(loc="upper right", fontsize=7, frameon=True, ncol=2)

    # Dashboard Header
    total_revenue = rfm_df["Monetary"].sum()
    total_cust = len(rfm_df)
    aov = total_revenue / rfm_df["Frequency"].sum()
    m1_ret = retention_matrix[1].dropna().mean() if 1 in retention_matrix.columns else 0.0

    fig.suptitle(
        f"Retail Pulse Executive Dashboard\n"
        f"Total Revenue: ${total_revenue:,.2f}  |  Active Customers: {total_cust:,}  |  "
        f"Avg Order Value: ${aov:,.2f}  |  Avg Month-1 Retention: {m1_ret:.1f}%",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved Executive Summary Dashboard -> {save_path}")

    return fig
