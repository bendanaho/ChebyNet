import numpy as np

beta = 1.00
data = np.load(f"ground_truth/Convection/beta={beta:.2f}_conv_ground_truth.npz")
X = data["x"]
T = data["t"]
U = data["u_true"]
print("x:", X.shape, "t:", T.shape, "u_true:", U.shape, "u_true.size:", U.size)