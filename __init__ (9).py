"""
Visualization package for Retail Pulse Analytics.
Provides publication-quality charting functions for RFM and Cohort analytics.
"""

from .plots import (
    plot_cohort_retention_heatmap,
    plot_rfm_distributions,
    plot_rfm_segments,
    plot_rfm_scatter_2d,
    plot_monthly_sales_trend,
    plot_cohort_ltv_curves,
    plot_executive_summary_dashboard,
)

__all__ = [
    "plot_cohort_retention_heatmap",
    "plot_rfm_distributions",
    "plot_rfm_segments",
    "plot_rfm_scatter_2d",
    "plot_monthly_sales_trend",
    "plot_cohort_ltv_curves",
    "plot_executive_summary_dashboard",
]
