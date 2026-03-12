import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C


# -------------------------
# Helper
# -------------------------
def min_dist_to_existing(X, x):
    return np.min(np.linalg.norm(X - x, axis=1))


# -------------------------
# Function 1
# -------------------------
f1_inputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_1\initial_inputs.npy")
f1_outputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_1\initial_outputs.npy")

kernel = C(1.0) * RBF(length_scale=[0.2, 0.2]) + WhiteKernel(noise_level=1e-8)

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=10,
    random_state=0
)

gp.fit(f1_inputs, f1_outputs)

beta = 2.0
rng = np.random.default_rng(42)
candidates = rng.random((200_000, 2))

mu, std = gp.predict(candidates, return_std=True)
ucb = mu + np.sqrt(beta) * std

x1 = candidates[np.argmax(ucb)]
submission1 = f"{x1[0]:.6f}-{x1[1]:.6f}"


# -------------------------
# Function 2
# -------------------------
f2_inputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_2\initial_inputs.npy")
f2_outputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_2\initial_outputs.npy")

kernel = C(1.0) * RBF(length_scale=[0.2, 0.2]) + WhiteKernel(noise_level=1e-8)

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=10,
    random_state=0
)

gp.fit(f2_inputs, f2_outputs)

beta = 2.0
rng = np.random.default_rng(42)
candidates = rng.random((200_000, 2))

mu, std = gp.predict(candidates, return_std=True)
ucb = mu + np.sqrt(beta) * std

x2 = candidates[np.argmax(ucb)]
submission2 = f"{x2[0]:.6f}-{x2[1]:.6f}"


# -------------------------
# Function 3
# -------------------------
f3_inputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_3\initial_inputs.npy")
f3_outputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_3\initial_outputs.npy")

kernel = C(1.0) * RBF(length_scale=[0.2, 0.2, 0.2]) + WhiteKernel(noise_level=1e-8)

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=10,
    random_state=0
)

gp.fit(f3_inputs, f3_outputs)

beta = 2.0
rng = np.random.default_rng(42)
candidates = rng.random((200_000, 3))

mu, std = gp.predict(candidates, return_std=True)
ucb = mu + np.sqrt(beta) * std

x3 = candidates[np.argmax(ucb)]
submission3 = f"{x3[0]:.6f}-{x3[1]:.6f}-{x3[2]:.6f}"


# -------------------------
# Function 4
# -------------------------
f4_inputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_4\initial_inputs.npy")
f4_outputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_4\initial_outputs.npy")

kernel = C(1.0) * RBF(length_scale=[0.2, 0.2, 0.2, 0.2]) + WhiteKernel(noise_level=1e-8)

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=10,
    random_state=0
)

gp.fit(f4_inputs, f4_outputs)

beta = 2.0
rng = np.random.default_rng(42)
candidates = rng.random((200_000, 4))

mu, std = gp.predict(candidates, return_std=True)
ucb = mu + np.sqrt(beta) * std

x4 = candidates[np.argmax(ucb)]
submission4 = f"{x4[0]:.6f}-{x4[1]:.6f}-{x4[2]:.6f}-{x4[3]:.6f}"


# -------------------------
# Function 5
# -------------------------
f5_inputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_5\initial_inputs.npy")
f5_outputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_5\initial_outputs.npy")

kernel = C(1.0) * RBF(length_scale=[0.2, 0.2, 0.2, 0.2]) + WhiteKernel(noise_level=1e-8)

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=10,
    random_state=0
)

gp.fit(f5_inputs, f5_outputs)

beta = 2.0
rng = np.random.default_rng(42)
candidates = rng.random((200_000, 4))

mu, std = gp.predict(candidates, return_std=True)
ucb = mu + np.sqrt(beta) * std

x5 = candidates[np.argmax(ucb)]
submission5 = f"{x5[0]:.6f}-{x5[1]:.6f}-{x5[2]:.6f}-{x5[3]:.6f}"


# -------------------------
# Function 6
# -------------------------
f6_inputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_6\initial_inputs.npy")
f6_outputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_6\initial_outputs.npy")

kernel = C(1.0) * RBF(length_scale=[0.2, 0.2, 0.2, 0.2, 0.2]) + WhiteKernel(noise_level=1e-8)

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=10,
    random_state=0
)

gp.fit(f6_inputs, f6_outputs)

beta = 2.0
rng = np.random.default_rng(42)
candidates = rng.random((200_000, 5))

mu, std = gp.predict(candidates, return_std=True)
ucb = mu + np.sqrt(beta) * std

x6 = candidates[np.argmax(ucb)]
submission6 = f"{x6[0]:.6f}-{x6[1]:.6f}-{x6[2]:.6f}-{x6[3]:.6f}-{x6[4]:.6f}"


# -------------------------
# Function 7
# -------------------------
f7_inputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_7\initial_inputs.npy")
f7_outputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_7\initial_outputs.npy")

kernel = C(1.0) * RBF(length_scale=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) + WhiteKernel(noise_level=1e-8)

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=10,
    random_state=0
)

gp.fit(f7_inputs, f7_outputs)

beta = 2.0
rng = np.random.default_rng(42)
candidates = rng.random((200_000, 6))

mu, std = gp.predict(candidates, return_std=True)
ucb = mu + np.sqrt(beta) * std

x7 = candidates[np.argmax(ucb)]
submission7 = f"{x7[0]:.6f}-{x7[1]:.6f}-{x7[2]:.6f}-{x7[3]:.6f}-{x7[4]:.6f}-{x7[5]:.6f}"


# -------------------------
# Function 8
# -------------------------
f8_inputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_8\initial_inputs.npy")
f8_outputs = np.load(r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data\function_8\initial_outputs.npy")

kernel = C(1.0) * RBF(length_scale=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) + WhiteKernel(noise_level=1e-8)

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=10,
    random_state=0
)

gp.fit(f8_inputs, f8_outputs)

beta = 2.0
rng = np.random.default_rng(42)
candidates = rng.random((200_000, 8))

mu, std = gp.predict(candidates, return_std=True)
ucb = mu + np.sqrt(beta) * std

x8 = candidates[np.argmax(ucb)]
submission8 = f"{x8[0]:.6f}-{x8[1]:.6f}-{x8[2]:.6f}-{x8[3]:.6f}-{x8[4]:.6f}-{x8[5]:.6f}-{x8[6]:.6f}-{x8[7]:.6f}"


# -------------------------
# Print final Week 1 queries
# -------------------------
print("Next query for Function 1:", submission1)
print("Next query for Function 2:", submission2)
print("Next query for Function 3:", submission3)
print("Next query for Function 4:", submission4)
print("Next query for Function 5:", submission5)
print("Next query for Function 6:", submission6)
print("Next query for Function 7:", submission7)
print("Next query for Function 8:", submission8)


# -------------------------
# Optional distance checks
# -------------------------
print("\nMinimum distances to existing points:")
print("F1:", min_dist_to_existing(f1_inputs, x1))
print("F2:", min_dist_to_existing(f2_inputs, x2))
print("F3:", min_dist_to_existing(f3_inputs, x3))
print("F4:", min_dist_to_existing(f4_inputs, x4))
print("F5:", min_dist_to_existing(f5_inputs, x5))
print("F6:", min_dist_to_existing(f6_inputs, x6))
print("F7:", min_dist_to_existing(f7_inputs, x7))
print("F8:", min_dist_to_existing(f8_inputs, x8))