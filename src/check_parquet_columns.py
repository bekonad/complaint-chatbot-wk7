# src/check_parquet.py
import pandas as pd

file = 'data/complaint_embeddings.parquet'
df = pd.read_parquet(file)

print("=== Parquet File Info ===")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst row (as dict):")
print(df.iloc[0].to_dict())

print("\nAny column containing 'text', 'chunk', 'narrative', 'content'?")
for col in df.columns:
    if any(word in col.lower() for word in ['text', 'chunk', 'narr', 'content', 'document']):
        print(f" → {col}")