import numpy as np

features = np.load(
    "data/person1/continuous_feature_sequence.npy"
)

print("Feature shape:", features.shape)
print("Data type:", features.dtype)
print("Number of time steps:", features.shape[0])
print("Feature dimension:", features.shape[1])