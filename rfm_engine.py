"""
RFM (Recency, Frequency, Monetary) Segmentation & Feature Engineering Engine.

Performs:
1. Behavioral aggregation: Recency, Frequency, Monetary value per customer.
2. Skew-correction transformations via NumPy log1p.
3. Robust quintile scoring (1 to 5) with collision-resistant rank bucketing.
4. Semantic rule-based segmentation (Champions, Loyalists, At Risk, etc.).
5. Unsupervised K-Means clustering with feature standardization and silhouette analysis.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class RFMEngine:
    """
    Computes RFM features, log transforms, quantile scores, customer segments,
    and K-Means clusters from transaction data.
    """

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        reference_date: Optional[pd.Timestamp] = None
    ):
        """
        Args:
            transactions_df: DataFrame with ['customer_id', 'order_id', 'order_date', 'total_amount'].
            reference_date: Analysis cutoff date. If None, snapshot is max(order_date) + 1 day.
        """
        if transactions_df.empty:
            raise ValueError("Transactions DataFrame cannot be empty.")

        self.raw_df = transactions_df.copy()
        self.raw_df["order_date"] = pd.to_datetime(self.raw_df["order_date"])

        if reference_date is None:
            self.reference_date = self.raw_df["order_date"].max() + pd.Timedelta(days=1)
        else:
            self.reference_date = pd.to_datetime(reference_date)

        self.rfm_df: Optional[pd.DataFrame] = None
        self.kmeans_model: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None

    def calculate_rfm_metrics(self) -> pd.DataFrame:
        """
        Aggregate transaction history into raw Recency, Frequency, and Monetary metrics.
        
        Returns:
            DataFrame indexed by customer_id with Recency, Frequency, Monetary,
            AOV (Average Order Value), and Customer_Tenure_Days.
        """
        grouped = self.raw_df.groupby("customer_id").agg(
            Last_Order_Date=("order_date", "max"),
            First_Order_Date=("order_date", "min"),
            Frequency=("order_id", "nunique"),
            Monetary=("total_amount", "sum")
        ).reset_index()

        # Recency in days from the reference snapshot
        grouped["Recency"] = (self.reference_date - grouped["Last_Order_Date"]).dt.days
        # Customer tenure in days
        grouped["Tenure_Days"] = (self.reference_date - grouped["First_Order_Date"]).dt.days
        # Average Order Value
        grouped["AOV"] = (grouped["Monetary"] / grouped["Frequency"]).round(2)
        grouped["Monetary"] = grouped["Monetary"].round(2)

        self.rfm_df = grouped
        return self.rfm_df

    def apply_log_transformations(self) -> pd.DataFrame:
        """
        Applies natural log transform log1p (ln(x + 1)) to handle heavy right-skew
        in e-commerce distributions (especially Monetary and Frequency power laws).
        """
        if self.rfm_df is None:
            self.calculate_rfm_metrics()

        assert self.rfm_df is not None
        self.rfm_df["Recency_log"] = np.log1p(np.maximum(self.rfm_df["Recency"], 0))
        self.rfm_df["Frequency_log"] = np.log1p(self.rfm_df["Frequency"])
        self.rfm_df["Monetary_log"] = np.log1p(np.maximum(self.rfm_df["Monetary"], 0))

        return self.rfm_df

    def assign_rfm_scores(self) -> pd.DataFrame:
        """
        Assign 1-5 quintile scores for Recency (R), Frequency (F), and Monetary (M).
        
        Uses rank(method='first') to ensure robust 5-bin quantiles even when
        large proportions of customers share identical repeat frequencies (e.g., F=1).
        
        Scoring logic:
        - Recency: 5 is best (most recent / lowest days), 1 is worst.
        - Frequency: 5 is best (highest purchase count), 1 is worst.
        - Monetary: 5 is best (highest spend), 1 is worst.
        """
        if self.rfm_df is None or "Recency_log" not in self.rfm_df.columns:
            self.apply_log_transformations()

        assert self.rfm_df is not None

        # Recency: Lower days = higher score (5 to 1)
        r_ranks = self.rfm_df["Recency"].rank(method="first", ascending=True)
        self.rfm_df["R_Score"] = pd.qcut(r_ranks, q=5, labels=[5, 4, 3, 2, 1]).astype(int)

        # Frequency: Higher count = higher score (1 to 5)
        f_ranks = self.rfm_df["Frequency"].rank(method="first", ascending=True)
        self.rfm_df["F_Score"] = pd.qcut(f_ranks, q=5, labels=[1, 2, 3, 4, 5]).astype(int)

        # Monetary: Higher spend = higher score (1 to 5)
        m_ranks = self.rfm_df["Monetary"].rank(method="first", ascending=True)
        self.rfm_df["M_Score"] = pd.qcut(m_ranks, q=5, labels=[1, 2, 3, 4, 5]).astype(int)

        # Composite RFM Score String (e.g. '555', '312')
        self.rfm_df["RFM_Score_Code"] = (
            self.rfm_df["R_Score"].astype(str) +
            self.rfm_df["F_Score"].astype(str) +
            self.rfm_df["M_Score"].astype(str)
        )

        # Composite Weighted Index (0-100 scale)
        # Standard retail weights: 20% Recency, 30% Frequency, 50% Monetary
        self.rfm_df["RFM_Index_Score"] = (
            (self.rfm_df["R_Score"] * 0.20 +
             self.rfm_df["F_Score"] * 0.30 +
             self.rfm_df["M_Score"] * 0.50) / 5.0 * 100
        ).round(1)

        return self.rfm_df

    def segment_customers(self) -> pd.DataFrame:
        """
        Map R and F scores into industry-standard business segments:
        - Champions
        - Loyal Customers
        - Potential Loyalists
        - Recent Customers
        - Promising
        - Customers Needing Attention
        - About to Sleep
        - At Risk
        - Can't Lose Them
        - Hibernating
        - Lost
        """
        if self.rfm_df is None or "R_Score" not in self.rfm_df.columns:
            self.assign_rfm_scores()

        assert self.rfm_df is not None

        def map_segment(row: pd.Series) -> str:
            r = row["R_Score"]
            f = row["F_Score"]
            m = row["M_Score"]

            if r in [4, 5] and f in [4, 5]:
                return "Champions"
            elif r in [3, 4, 5] and f in [3, 4, 5]:
                return "Loyal Customers"
            elif r in [4, 5] and f in [2, 3]:
                return "Potential Loyalists"
            elif r in [4, 5] and f == 1:
                return "Recent Customers"
            elif r == 3 and f in [1, 2]:
                return "Promising"
            elif r == 3 and f in [3, 4]:
                return "Customers Needing Attention"
            elif r == 2 and f in [1, 2]:
                return "About to Sleep"
            elif r in [1, 2] and f in [3, 4]:
                return "At Risk"
            elif r == 1 and f == 5:
                return "Can't Lose Them"
            elif r in [1, 2] and f in [1, 2]:
                return "Hibernating"
            else:
                return "Lost"

        self.rfm_df["Segment"] = self.rfm_df.apply(map_segment, axis=1)

        # Provide actionable strategy recommendations per segment
        action_map = {
            "Champions": "Reward loyalty, early access to new launches, VIP perks",
            "Loyal Customers": "Upsell higher value products, ask for reviews, referral programs",
            "Potential Loyalists": "Offer membership/loyalty program, recommend complementary products",
            "Recent Customers": "Onboarding nurture sequence, post-purchase education, incentive for 2nd order",
            "Promising": "Build brand engagement, offer time-sensitive discount codes",
            "Customers Needing Attention": "Reactivate with targeted promos, personalized product recommendations",
            "About to Sleep": "Send re-engagement campaigns before they go inactive, survey dissatisfaction",
            "At Risk": "Send strong personalized win-back offers, highlight new features/products",
            "Can't Lose Them": "High-urgency win-back outreach, direct outreach, heavy discount",
            "Hibernating": "Low-cost re-engagement or suppress to save marketing spend",
            "Lost": "Attempt revival campaign once; otherwise filter from paid channels"
        }
        self.rfm_df["Actionable_Strategy"] = self.rfm_df["Segment"].map(action_map)

        return self.rfm_df

    def fit_kmeans_clusters(
        self,
        n_clusters: int = 4,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, float]:
        """
        Fit K-Means clustering on standardized log-transformed RFM features.
        
        Args:
            n_clusters: Number of clusters (default 4).
            random_state: Seed for reproducibility.
            
        Returns:
            Tuple of (rfm_df with 'Cluster' column, silhouette_score).
        """
        if self.rfm_df is None or "Recency_log" not in self.rfm_df.columns:
            self.segment_customers()

        assert self.rfm_df is not None

        features = ["Recency_log", "Frequency_log", "Monetary_log"]
        X = self.rfm_df[features].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=random_state
        )
        cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        self.rfm_df["Cluster"] = cluster_labels

        sil_score = float(silhouette_score(X_scaled, cluster_labels))
        self.rfm_df["Cluster_Name"] = self.rfm_df["Cluster"].apply(lambda c: f"Cluster {c}")

        return self.rfm_df, sil_score

    def get_segment_summary(self) -> pd.DataFrame:
        """
        Calculate aggregate business KPIs grouped by Customer Segment.
        
        Returns:
            DataFrame with Customer Count, Base Share %, Total Revenue,
            Revenue Share %, Avg Recency, Avg Frequency, Avg Monetary, and AOV.
        """
        if self.rfm_df is None or "Segment" not in self.rfm_df.columns:
            self.segment_customers()

        assert self.rfm_df is not None

        total_customers = len(self.rfm_df)
        total_revenue = self.rfm_df["Monetary"].sum()

        summary = self.rfm_df.groupby("Segment").agg(
            Customer_Count=("customer_id", "count"),
            Total_Revenue=("Monetary", "sum"),
            Avg_Recency_Days=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Spend=("Monetary", "mean"),
            Avg_AOV=("AOV", "mean")
        ).reset_index()

        summary["Customer_Share_Pct"] = (summary["Customer_Count"] / total_customers * 100).round(2)
        summary["Revenue_Share_Pct"] = (summary["Total_Revenue"] / total_revenue * 100).round(2)
        summary["Avg_Recency_Days"] = summary["Avg_Recency_Days"].round(1)
        summary["Avg_Frequency"] = summary["Avg_Frequency"].round(2)
        summary["Avg_Spend"] = summary["Avg_Spend"].round(2)
        summary["Avg_AOV"] = summary["Avg_AOV"].round(2)
        summary["Total_Revenue"] = summary["Total_Revenue"].round(2)

        # Sort by Total Revenue descending
        summary = summary.sort_values(by="Total_Revenue", ascending=False).reset_index(drop=True)
        return summary

    def get_cluster_summary(self) -> pd.DataFrame:
        """Calculate aggregate business KPIs grouped by K-Means Cluster."""
        if self.rfm_df is None or "Cluster" not in self.rfm_df.columns:
            self.fit_kmeans_clusters()

        assert self.rfm_df is not None

        summary = self.rfm_df.groupby("Cluster").agg(
            Customer_Count=("customer_id", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum")
        ).reset_index()

        total_customers = len(self.rfm_df)
        total_revenue = self.rfm_df["Monetary"].sum()

        summary["Customer_Share_Pct"] = (summary["Customer_Count"] / total_customers * 100).round(2)
        summary["Revenue_Share_Pct"] = (summary["Total_Revenue"] / total_revenue * 100).round(2)
        summary["Avg_Recency"] = summary["Avg_Recency"].round(1)
        summary["Avg_Frequency"] = summary["Avg_Frequency"].round(2)
        summary["Avg_Monetary"] = summary["Avg_Monetary"].round(2)
        summary["Total_Revenue"] = summary["Total_Revenue"].round(2)

        return summary

    def run_full_pipeline(self, n_clusters: int = 4) -> pd.DataFrame:
        """Execute end-to-end RFM processing pipeline."""
        self.calculate_rfm_metrics()
        self.apply_log_transformations()
        self.assign_rfm_scores()
        self.segment_customers()
        self.fit_kmeans_clusters(n_clusters=n_clusters)
        return self.rfm_df
