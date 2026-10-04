import numpy as np
import pandas as pd

feature_file = "data/synthetic/sample_0000.npy"

features = np.load(feature_file)

labels = pd.read_csv(
    "data/synthetic/labels.csv"
)

print("Feature shape:", features.shape)
print()
print("First 5 labels:")
print(labels.head())