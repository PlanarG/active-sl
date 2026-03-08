import pandas as pd
from pathlib import Path

csv_path = Path(__file__).parent / "chinchilla_svg_extracted_data.csv"
out_path = Path(__file__).parent / "data" / "all_data.parquet"

df = pd.read_csv(csv_path)
df["Training Tokens"] = df["Training FLOP"] / (6.0 * df["Model Size"])

out_path.parent.mkdir(parents=True, exist_ok=True)
df[["Training Tokens", "Model Size", "loss"]].to_parquet(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
