from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def make_figures(clean, rfm, eval_df, clustered, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(clean["LineValue"], bins=50, ax=ax, color=sns.color_palette()[0])
    ax.set(title="Distribution of qualifying transaction-line values", xlabel="Line value", ylabel="Rows")
    fig.tight_layout(); fig.savefig(out/"transaction_value_distribution.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(rfm["Monetary"], bins=40, ax=ax, color=sns.color_palette()[0])
    ax.set(title="Customer monetary-value distribution", xlabel="Customer monetary value", ylabel="Customers")
    fig.tight_layout(); fig.savefig(out/"customer_monetary_distribution.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(rfm[["Recency","Frequency","Monetary"]].corr(), annot=True, cmap="vlag", center=0, ax=ax)
    ax.set_title("RFM feature correlations")
    fig.tight_layout(); fig.savefig(out/"rfm_correlation.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(eval_df["k"], eval_df["inertia"], marker="o")
    ax.set(title="K-Means elbow curve", xlabel="Number of clusters (k)", ylabel="Inertia")
    fig.tight_layout(); fig.savefig(out/"elbow_curve.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(eval_df["k"], eval_df["silhouette"], marker="o")
    ax.set(title="Silhouette score by cluster count", xlabel="Number of clusters (k)", ylabel="Silhouette score")
    fig.tight_layout(); fig.savefig(out/"silhouette_scores.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=clustered, x="Frequency", y="Monetary", hue="Cluster", palette="tab10", ax=ax)
    ax.set(title="Customer segments: frequency vs monetary value")
    fig.tight_layout(); fig.savefig(out/"customer_segments.png", dpi=150); plt.close(fig)
