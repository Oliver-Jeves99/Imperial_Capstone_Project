import numpy as np
import warnings
from math import erf, sqrt, pi

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning



def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return (1.0 / sqrt(2.0 * pi)) * np.exp(-0.5 * z * z)

def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def _fit_gp(X, y):
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    d = X.shape[1]

    kernel = (
        C(1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(length_scale=[0.2] * d, length_scale_bounds=(1e-3, 1e7))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=0,
        alpha=1e-12
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gp.fit(X, y)

    return gp


def _expected_improvement(mu, std, best_y, xi=0.01):

    std = np.maximum(std, 1e-12)
    improvement = mu - best_y - xi
    z = improvement / std
    return improvement * _normal_cdf(z) + std * _normal_pdf(z)


# ------------------- Interpretability helpers -------------------
def permutation_importance_gp(gp, X, y, n_repeats=8, seed=0):

    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    yhat = gp.predict(X)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    baseline_r2 = 1.0 - ss_res / ss_tot

    importances = np.zeros(X.shape[1], dtype=float)

    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            yhat_p = gp.predict(Xp)
            ss_res_p = np.sum((y - yhat_p) ** 2)
            r2_p = 1.0 - ss_res_p / ss_tot
            drops.append(baseline_r2 - r2_p)
        importances[j] = float(np.mean(drops))

    return importances


def importance_weighted_sigmas(importances, base_sigma):

    imps = np.asarray(importances, dtype=float)
    imps = imps - np.min(imps)

    if np.all(imps < 1e-12):
        w = np.ones_like(imps) / len(imps)
    else:
        w = imps / (np.sum(imps) + 1e-12)

    d = len(w)
    w = 0.2 * (np.ones(d) / d) + 0.8 * w  
    w = w / (np.max(w) + 1e-12)           

    return base_sigma * w


def propose_next_point_hybrid(
    X,
    y,
    beta=2.0,
    n_candidates=200_000,
    frac_local=0.6,
    top_k=3,
    local_sigma=0.10,
    seed=42,
    acq="ucb",              
    xi=0.01,                
    interpret=False,        
    interpret_repeats=8,    
    min_dist_threshold=0.0  
):

    X = np.asarray(X)
    y = np.asarray(y).ravel()
    d = X.shape[1]

    gp = _fit_gp(X, y)
    rng = np.random.default_rng(seed)

    n_local = int(n_candidates * frac_local)
    n_global = n_candidates - n_local


    top_idx = np.argsort(y)[-top_k:]
    anchors = X[top_idx]
    anchor_choices = anchors[rng.integers(0, len(anchors), size=n_local)]

    if interpret:
        imps = permutation_importance_gp(gp, X, y, n_repeats=interpret_repeats, seed=seed)
        order = np.argsort(imps)[::-1]

        print("\n[Interpretability] Permutation importances (highest first):")
        for j in order:
            print(f"  x{j+1}: {imps[j]:.6f}")

        sigmas = importance_weighted_sigmas(imps, local_sigma)
        noise = rng.normal(loc=0.0, scale=1.0, size=(n_local, d)) * sigmas
        local = anchor_choices + noise
    else:
        local = anchor_choices + rng.normal(loc=0.0, scale=local_sigma, size=(n_local, d))

    local = np.clip(local, 0.0, 1.0)

    
    global_cand = rng.random((n_global, d))
    candidates = np.vstack([local, global_cand])

    mu, std = gp.predict(candidates, return_std=True)

    if acq.lower() == "ucb":
        scores = mu + np.sqrt(beta) * std
    elif acq.lower() == "ei":
        best_y = np.max(y)
        scores = _expected_improvement(mu, std, best_y, xi=xi)
    else:
        raise ValueError("acq must be 'ucb' or 'ei'")


    if min_dist_threshold > 0.0:
        dists = np.min(np.linalg.norm(candidates[:, None, :] - X[None, :, :], axis=2), axis=1)
        valid = dists >= min_dist_threshold

        if np.any(valid):
            valid_idx = np.argmax(scores[valid])
            x_next = candidates[valid][valid_idx]
        else:
  
            x_next = candidates[np.argmax(dists)]
    else:
        x_next = candidates[np.argmax(scores)]

    return x_next


def min_dist_to_existing(X, x):
    X = np.asarray(X)
    x = np.asarray(x)
    return np.min(np.linalg.norm(X - x, axis=1))



base = r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data"

f1_inputs = np.load(fr"{base}\function_1\initial_inputs.npy")
f1_outputs = np.load(fr"{base}\function_1\initial_outputs.npy")

f2_inputs = np.load(fr"{base}\function_2\initial_inputs.npy")
f2_outputs = np.load(fr"{base}\function_2\initial_outputs.npy")

f3_inputs = np.load(fr"{base}\function_3\initial_inputs.npy")
f3_outputs = np.load(fr"{base}\function_3\initial_outputs.npy")

f4_inputs = np.load(fr"{base}\function_4\initial_inputs.npy")
f4_outputs = np.load(fr"{base}\function_4\initial_outputs.npy")

f5_inputs = np.load(fr"{base}\function_5\initial_inputs.npy")
f5_outputs = np.load(fr"{base}\function_5\initial_outputs.npy")

f6_inputs = np.load(fr"{base}\function_6\initial_inputs.npy")
f6_outputs = np.load(fr"{base}\function_6\initial_outputs.npy")

f7_inputs = np.load(fr"{base}\function_7\initial_inputs.npy")
f7_outputs = np.load(fr"{base}\function_7\initial_outputs.npy")

f8_inputs = np.load(fr"{base}\function_8\initial_inputs.npy")
f8_outputs = np.load(fr"{base}\function_8\initial_outputs.npy")



f1_inputs = np.vstack([f1_inputs, [0.704538, 0.931355]])
f1_outputs = np.append(f1_outputs, -3.322279659363804e-68)

f2_inputs = np.vstack([f2_inputs, [0.954386, 0.459885]])
f2_outputs = np.append(f2_outputs, 0.0819233324310763)

f3_inputs = np.vstack([f3_inputs, [0.561613, 0.998048, 0.855377]])
f3_outputs = np.append(f3_outputs, -0.07959591486997375)

f4_inputs = np.vstack([f4_inputs, [0.408605, 0.430020, 0.357243, 0.435747]])
f4_outputs = np.append(f4_outputs, 0.4787334082318968)

f5_inputs = np.vstack([f5_inputs, [0.263061, 0.975247, 0.993209, 0.984517]])
f5_outputs = np.append(f5_outputs, 3847.7596239129275)

f6_inputs = np.vstack([f6_inputs, [0.377244, 0.134038, 0.574764, 0.990875, 0.094520]])
f6_outputs = np.append(f6_outputs, -0.647366630972217)

f7_inputs = np.vstack([f7_inputs, [0.000143, 0.372762, 0.180309, 0.283532, 0.398956, 0.541385]])
f7_outputs = np.append(f7_outputs, 1.5720056973495211)

f8_inputs = np.vstack([f8_inputs, [0.064853, 0.072792, 0.242182, 0.024903, 0.874840, 0.479611, 0.007117, 0.501292]])
f8_outputs = np.append(f8_outputs, 9.8595663843116)



settings = {
    "f1": dict(acq="ucb", beta=4.0, frac_local=0.2, top_k=3, local_sigma=0.12),
    "f2": dict(acq="ucb", beta=1.0, frac_local=0.6, top_k=5, local_sigma=0.10),
    "f3": dict(acq="ucb", beta=3.0, frac_local=0.6, top_k=5, local_sigma=0.10),


    "f4": dict(acq="ei", xi=0.01, frac_local=0.9,  top_k=5, local_sigma=0.07, interpret=True),
    "f5": dict(acq="ei", xi=0.01, frac_local=0.95, top_k=5, local_sigma=0.06, interpret=True,
               min_dist_threshold=0.06),

    "f6": dict(acq="ucb", beta=2.0, frac_local=0.7, top_k=5, local_sigma=0.09),
    "f7": dict(acq="ucb", beta=2.0, frac_local=0.7, top_k=5, local_sigma=0.09),
    "f8": dict(acq="ucb", beta=2.0, frac_local=0.75, top_k=7, local_sigma=0.08),
}



x1 = propose_next_point_hybrid(f1_inputs, f1_outputs, **settings["f1"])
x2 = propose_next_point_hybrid(f2_inputs, f2_outputs, **settings["f2"])
x3 = propose_next_point_hybrid(f3_inputs, f3_outputs, **settings["f3"])
x4 = propose_next_point_hybrid(f4_inputs, f4_outputs, **settings["f4"])
x5 = propose_next_point_hybrid(f5_inputs, f5_outputs, **settings["f5"])
x6 = propose_next_point_hybrid(f6_inputs, f6_outputs, **settings["f6"])
x7 = propose_next_point_hybrid(f7_inputs, f7_outputs, **settings["f7"])
x8 = propose_next_point_hybrid(f8_inputs, f8_outputs, **settings["f8"])



submission1 = f"{x1[0]:.6f}-{x1[1]:.6f}"
submission2 = f"{x2[0]:.6f}-{x2[1]:.6f}"
submission3 = f"{x3[0]:.6f}-{x3[1]:.6f}-{x3[2]:.6f}"
submission4 = f"{x4[0]:.6f}-{x4[1]:.6f}-{x4[2]:.6f}-{x4[3]:.6f}"
submission5 = f"{x5[0]:.6f}-{x5[1]:.6f}-{x5[2]:.6f}-{x5[3]:.6f}"
submission6 = f"{x6[0]:.6f}-{x6[1]:.6f}-{x6[2]:.6f}-{x6[3]:.6f}-{x6[4]:.6f}"
submission7 = "-".join([f"{v:.6f}" for v in x7])
submission8 = "-".join([f"{v:.6f}" for v in x8])

print("Next query for Function 1:", submission1)
print("Next query for Function 2:", submission2)
print("Next query for Function 3:", submission3)
print("Next query for Function 4:", submission4, "(EI)")
print("Next query for Function 5:", submission5, "(EI)")
print("Next query for Function 6:", submission6)
print("Next query for Function 7:", submission7)
print("Next query for Function 8:", submission8)

print("\nMin distances to existing:")
print("F1:", min_dist_to_existing(f1_inputs, x1))
print("F2:", min_dist_to_existing(f2_inputs, x2))
print("F3:", min_dist_to_existing(f3_inputs, x3))
print("F4:", min_dist_to_existing(f4_inputs, x4))
print("F5:", min_dist_to_existing(f5_inputs, x5))
print("F6:", min_dist_to_existing(f6_inputs, x6))
print("F7:", min_dist_to_existing(f7_inputs, x7))
print("F8:", min_dist_to_existing(f8_inputs, x8))