import pandas as pd
from pathlib import Path

csv_path = Path(__file__).parent / "sparsity.csv"
out_path = Path(__file__).parent / "data" / "all_data.parquet"

df = pd.read_csv(csv_path)
df["P"] = df["N_total"] / df["N_active"]

out_path.parent.mkdir(parents=True, exist_ok=True)
df[["P", "N_active", "D1", "D2", "Loss"]].rename(
    columns={"Loss": "loss"}
).to_parquet(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
