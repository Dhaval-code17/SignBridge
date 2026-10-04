import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"D:\signbridge_model1\data\raw\CISLR")  # adjust if your CSVs live elsewhere

# Load the three annotation files
dataset = pd.read_csv(DATA_DIR / "dataset.csv")
prototype = pd.read_csv(DATA_DIR / "prototype.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# NOTE: adjust this column name if your gloss column isn't called "gloss"
GLOSS_COL = "gloss"

candidate_glosses = [
    "monday", "thursday", "interest", "friday", "pencil", "yellow",
    "christmas", "mother", "tuesday", "angry", "increase", "august",
    "wednesday", "bill", "wash", "rubber", "sunday", "name", "light",
    "date", "india", "iron", "bad", "sick", "month", "thank you",
    "reservation", "saturday", "week", "cat", "goat", "brain", "ant",
    "duck", "land", "leaf", "good", "hot", "grow"
]

rows = []
for g in candidate_glosses:
    total = (dataset[GLOSS_COL] == g).sum()
    in_test = (test[GLOSS_COL] == g).sum()
    in_proto = (prototype[GLOSS_COL] == g).sum()
    leftover = total - in_test  # rough pool available for train/val

    rows.append({
        "gloss": g,
        "total_in_dataset": total,
        "in_test": in_test,
        "in_prototype": in_proto,
        "leftover_for_train_val": leftover
    })

result = pd.DataFrame(rows).sort_values("leftover_for_train_val", ascending=False)

print(result.to_string(index=False))

out_path = Path(r"D:\signbridge_model1\preprocessing\class_split_check.csv")
result.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

print("\n--- Summary ---")
print(f"Classes with leftover >= 4 (usable for train/val split): {(result['leftover_for_train_val'] >= 4).sum()}")
print(f"Classes with leftover >= 3: {(result['leftover_for_train_val'] >= 3).sum()}")
print(f"Classes with leftover < 3 (risky): {(result['leftover_for_train_val'] < 3).sum()}")