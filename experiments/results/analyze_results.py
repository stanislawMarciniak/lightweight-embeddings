import pandas as pd

# Load CSV
df = pd.read_csv("results.csv")

# Metrics and whether higher values are better
metrics = {
    "pearson": True,
    "spearman": True,
    "mse": False,
    "cosine_error": False,
    "avg_sample_time": False,
    "model_size_mb": False
}

# Create a ranking for each metric
for metric, higher_is_better in metrics.items():
    df[f"{metric}_rank"] = df[metric].rank(ascending=not higher_is_better, method="min")

# Compute an overall rank (average of ranks)
rank_cols = [f"{m}_rank" for m in metrics.keys()]
df["overall_rank"] = df[rank_cols].mean(axis=1)

# Sort by overall rank
df_sorted = df.sort_values("overall_rank")

# Display overall ranking
print("\n=== Overall Ranking ===")
for i, row in enumerate(df_sorted.itertuples(), start=1):
    print(f"{i}. {row.model} ({row.precision}) - Overall Rank: {row.overall_rank:.2f}")

# Display ranking per metric
for metric, higher_is_better in metrics.items():
    print(f"\n=== Ranking by {metric} ({'higher is better' if higher_is_better else 'lower is better'}) ===")
    df_metric_sorted = df.sort_values(metric, ascending=not higher_is_better)
    for i, row in enumerate(df_metric_sorted.itertuples(), start=1):
        print(f"{i}. {row.model} ({row.precision}) - {metric}: {getattr(row, metric):.6f}")
