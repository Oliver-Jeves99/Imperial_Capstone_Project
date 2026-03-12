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
    n_candidates=240_000,
    frac_local=0.6,
    top_k=3,
    local_sigma=0.10,
    seed=42,
    acq="ucb",
    xi=0.01,
    interpret=False,
    interpret_repeats=8,
    min_dist_threshold=0.0,
    interior_eps=0.0,
    max_step=None,
    boundary_penalty_weight=0.0,
    boundary_eps=0.02
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

    if max_step is not None:
        delta = local - anchor_choices
        norms = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, max_step / norms)
        local = anchor_choices + delta * scale

    if n_global > 0:
        global_cand = rng.random((n_global, d))
        candidates = np.vstack([local, global_cand])
    else:
        candidates = local

    if interior_eps and interior_eps > 0.0:
        candidates = np.clip(candidates, interior_eps, 1.0 - interior_eps)
    else:
        candidates = np.clip(candidates, 0.0, 1.0)

    mu, std = gp.predict(candidates, return_std=True)

    if acq.lower() == "ucb":
        scores = mu + np.sqrt(beta) * std
    elif acq.lower() == "ei":
        best_y = np.max(y)
        scores = _expected_improvement(mu, std, best_y, xi=xi)
    else:
        raise ValueError("acq must be 'ucb' or 'ei'")

    if boundary_penalty_weight and boundary_penalty_weight > 0.0:
        near_b = (candidates < boundary_eps) | (candidates > (1.0 - boundary_eps))
        penalty = np.mean(near_b, axis=1)
        scores = scores - boundary_penalty_weight * penalty

    if min_dist_threshold > 0.0:
        dists = np.min(np.linalg.norm(candidates[:, None, :] - X[None, :, :], axis=2), axis=1)
        valid = dists >= min_dist_threshold
        if np.any(valid):
            x_next = candidates[valid][np.argmax(scores[valid])]
        else:
            x_next = candidates[np.argmax(dists)]
    else:
        x_next = candidates[np.argmax(scores)]

    return x_next


def min_dist_to_existing(X, x):
    X = np.asarray(X)
    x = np.asarray(x)
    return np.min(np.linalg.norm(X - x, axis=1))


EPS_SUBMIT = 1e-6

def fmt_point(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, EPS_SUBMIT, 1.0 - EPS_SUBMIT)
    return "-".join([f"{v:.6f}" for v in x])


base = r"C:\Users\ollie\OneDrive\Documents\capstone\initial_data"

f1_inputs = np.load(fr"{base}\function_1\initial_inputs.npy"); f1_outputs = np.load(fr"{base}\function_1\initial_outputs.npy")
f2_inputs = np.load(fr"{base}\function_2\initial_inputs.npy"); f2_outputs = np.load(fr"{base}\function_2\initial_outputs.npy")
f3_inputs = np.load(fr"{base}\function_3\initial_inputs.npy"); f3_outputs = np.load(fr"{base}\function_3\initial_outputs.npy")
f4_inputs = np.load(fr"{base}\function_4\initial_inputs.npy"); f4_outputs = np.load(fr"{base}\function_4\initial_outputs.npy")
f5_inputs = np.load(fr"{base}\function_5\initial_inputs.npy"); f5_outputs = np.load(fr"{base}\function_5\initial_outputs.npy")
f6_inputs = np.load(fr"{base}\function_6\initial_inputs.npy"); f6_outputs = np.load(fr"{base}\function_6\initial_outputs.npy")
f7_inputs = np.load(fr"{base}\function_7\initial_inputs.npy"); f7_outputs = np.load(fr"{base}\function_7\initial_outputs.npy")
f8_inputs = np.load(fr"{base}\function_8\initial_inputs.npy"); f8_outputs = np.load(fr"{base}\function_8\initial_outputs.npy")

f1_inputs = np.vstack([f1_inputs, [0.704538, 0.931355]]); f1_outputs = np.append(f1_outputs, -3.322279659363804e-68)
f2_inputs = np.vstack([f2_inputs, [0.954386, 0.459885]]); f2_outputs = np.append(f2_outputs, 0.0819233324310763)
f3_inputs = np.vstack([f3_inputs, [0.561613, 0.998048, 0.855377]]); f3_outputs = np.append(f3_outputs, -0.07959591486997375)
f4_inputs = np.vstack([f4_inputs, [0.408605, 0.430020, 0.357243, 0.435747]]); f4_outputs = np.append(f4_outputs, 0.4787334082318968)
f5_inputs = np.vstack([f5_inputs, [0.263061, 0.975247, 0.993209, 0.984517]]); f5_outputs = np.append(f5_outputs, 3847.7596239129275)
f6_inputs = np.vstack([f6_inputs, [0.377244, 0.134038, 0.574764, 0.990875, 0.094520]]); f6_outputs = np.append(f6_outputs, -0.647366630972217)
f7_inputs = np.vstack([f7_inputs, [0.000143, 0.372762, 0.180309, 0.283532, 0.398956, 0.541385]]); f7_outputs = np.append(f7_outputs, 1.5720056973495211)
f8_inputs = np.vstack([f8_inputs, [0.064853, 0.072792, 0.242182, 0.024903, 0.874840, 0.479611, 0.007117, 0.501292]]); f8_outputs = np.append(f8_outputs, 9.8595663843116)

f1_inputs = np.vstack([f1_inputs, [0.764663, 0.905273]]); f1_outputs = np.append(f1_outputs, 8.015849166654494e-68)
f2_inputs = np.vstack([f2_inputs, [0.692725, 0.278929]]); f2_outputs = np.append(f2_outputs, 0.5605145847513243)
f3_inputs = np.vstack([f3_inputs, [0.369284, 0.416431, 0.301011]]); f3_outputs = np.append(f3_outputs, -0.07720936803871446)
f4_inputs = np.vstack([f4_inputs, [0.455240, 0.446060, 0.268139, 0.441884]]); f4_outputs = np.append(f4_outputs, -1.7128668059186611)
f5_inputs = np.vstack([f5_inputs, [0.231670, 0.850028, 1.000000, 0.889556]]); f5_outputs = np.append(f5_outputs, 1980.5547253042491)
f6_inputs = np.vstack([f6_inputs, [0.433148, 0.596539, 0.193715, 0.950096, 0.025464]]); f6_outputs = np.append(f6_outputs, -0.9172054373685039)
f7_inputs = np.vstack([f7_inputs, [0.000000, 0.381704, 0.139019, 0.294245, 0.465160, 0.557858]]); f7_outputs = np.append(f7_outputs, 1.161948203881051)
f8_inputs = np.vstack([f8_inputs, [0.197772, 0.083170, 0.101703, 0.000000, 0.971840, 0.425134, 0.239160, 0.541693]]); f8_outputs = np.append(f8_outputs, 9.9277364699241)

f1_inputs = np.vstack([f1_inputs, [0.787430, 0.334127]]); f1_outputs = np.append(f1_outputs, 2.365788342932173e-78)
f2_inputs = np.vstack([f2_inputs, [0.736243, 0.130613]]); f2_outputs = np.append(f2_outputs, 0.4834674378790105)
f3_inputs = np.vstack([f3_inputs, [0.020046, 0.000725, 0.001707]]); f3_outputs = np.append(f3_outputs, -0.19661686359454497)
f4_inputs = np.vstack([f4_inputs, [0.359551, 0.394786, 0.396309, 0.436492]]); f4_outputs = np.append(f4_outputs, 0.32520294207005707)
f5_inputs = np.vstack([f5_inputs, [0.237522, 0.999999, 0.999999, 0.908792]]); f5_outputs = np.append(f5_outputs, 3416.5459869854776)
f6_inputs = np.vstack([f6_inputs, [0.526194, 0.287897, 0.968220, 0.966925, 0.293788]]); f6_outputs = np.append(f6_outputs, -0.9450945167844301)
f7_inputs = np.vstack([f7_inputs, [0.000001, 0.345198, 0.000001, 0.269137, 0.327548, 0.492848]]); f7_outputs = np.append(f7_outputs, 1.3968552772682414)
f8_inputs = np.vstack([f8_inputs, [0.000001, 0.000001, 0.014196, 0.000001, 0.999999, 0.377710, 0.205227, 0.666597]]); f8_outputs = np.append(f8_outputs, 9.8593164975486)

f1_inputs = np.vstack([f1_inputs, [0.066507, 0.516191]]); f1_outputs = np.append(f1_outputs, 7.993697275108671e-96)
f2_inputs = np.vstack([f2_inputs, [0.000002, 0.344531]]); f2_outputs = np.append(f2_outputs, 0.07601510476325644)
f3_inputs = np.vstack([f3_inputs, [0.999999, 0.999999, 0.421755]]); f3_outputs = np.append(f3_outputs, -0.07244203114612231)
f4_inputs = np.vstack([f4_inputs, [0.388945, 0.421860, 0.405767, 0.444493]]); f4_outputs = np.append(f4_outputs, 0.1811737335858976)
f5_inputs = np.vstack([f5_inputs, [0.455631, 0.999044, 0.991342, 0.354808]]); f5_outputs = np.append(f5_outputs, 1673.9226983107021)
f6_inputs = np.vstack([f6_inputs, [0.458285, 0.213540, 0.575318, 0.563291, 0.000001]]); f6_outputs = np.append(f6_outputs, -0.5408553203250122)
f7_inputs = np.vstack([f7_inputs, [0.084086, 0.419577, 0.142716, 0.306046, 0.379254, 0.748677]]); f7_outputs = np.append(f7_outputs, 1.7323794839941804)
f8_inputs = np.vstack([f8_inputs, [0.071271, 0.132901, 0.165747, 0.089521, 0.999999, 0.505679, 0.246722, 0.577658]]); f8_outputs = np.append(f8_outputs, 9.9661178017431)

f1_inputs = np.vstack([f1_inputs, [0.999999, 0.803860]]); f1_outputs = np.append(f1_outputs, -5.690288824577651e-118)
f2_inputs = np.vstack([f2_inputs, [0.695867, 0.421840]]); f2_outputs = np.append(f2_outputs, 0.576655702229003)
f3_inputs = np.vstack([f3_inputs, [0.125727, 0.000001, 0.658349]]); f3_outputs = np.append(f3_outputs, -0.16208383278266597)
f4_inputs = np.vstack([f4_inputs, [0.395669, 0.387605, 0.415443, 0.426498]]); f4_outputs = np.append(f4_outputs, 0.4558524754225357)
f5_inputs = np.vstack([f5_inputs, [0.143300, 0.989604, 0.979809, 0.999287]]); f5_outputs = np.append(f5_outputs, 4028.952384702503)
f6_inputs = np.vstack([f6_inputs, [0.407584, 0.000001, 0.944557, 0.592214, 0.042562]]); f6_outputs = np.append(f6_outputs, -0.94632890127082)
f7_inputs = np.vstack([f7_inputs, [0.123119, 0.275810, 0.198763, 0.357377, 0.375446, 0.408981]]); f7_outputs = np.append(f7_outputs, 1.3986423834530184)
f8_inputs = np.vstack([f8_inputs, [0.056309, 0.101669, 0.220956, 0.030745, 0.953703, 0.552713, 0.268669, 0.433411]]); f8_outputs = np.append(f8_outputs, 9.9280085515564)

f1_inputs = np.vstack([f1_inputs, [0.725488, 0.206181]]); f1_outputs = np.append(f1_outputs, -2.329897378733848e-97)
f2_inputs = np.vstack([f2_inputs, [0.238697, 0.435399]]); f2_outputs = np.append(f2_outputs, 0.15146532804255686)
f3_inputs = np.vstack([f3_inputs, [0.347118, 0.055994, 0.862974]]); f3_outputs = np.append(f3_outputs, -0.10383613113196388)
f4_inputs = np.vstack([f4_inputs, [0.370115, 0.420104, 0.425088, 0.414856]]); f4_outputs = np.append(f4_outputs, 0.6837384870853502)
f5_inputs = np.vstack([f5_inputs, [0.142663, 0.968537, 0.886654, 0.969656]]); f5_outputs = np.append(f5_outputs, 2554.6832751977577)
f6_inputs = np.vstack([f6_inputs, [0.581393, 0.060652, 0.499427, 0.924351, 0.014431]]); f6_outputs = np.append(f6_outputs, -0.8749090644104073)
f7_inputs = np.vstack([f7_inputs, [0.030390, 0.343146, 0.272878, 0.353696, 0.368575, 0.733595]]); f7_outputs = np.append(f7_outputs, 2.0484842906448035)
f8_inputs = np.vstack([f8_inputs, [0.103742, 0.092350, 0.064259, 0.130930, 0.957985, 0.703618, 0.164961, 0.565136]]); f8_outputs = np.append(f8_outputs, 9.9268022373009)

f1_inputs = np.vstack([f1_inputs, [0.139316, 0.459704]]); f1_outputs = np.append(f1_outputs, 5.469294034655586e-59)
f2_inputs = np.vstack([f2_inputs, [0.695586, 0.339166]]); f2_outputs = np.append(f2_outputs, 0.4909592247155518)
f3_inputs = np.vstack([f3_inputs, [0.257882, 0.051885, 0.528719]]); f3_outputs = np.append(f3_outputs, -0.08843031377028249)
f4_inputs = np.vstack([f4_inputs, [0.332870, 0.437932, 0.428174, 0.459359]]); f4_outputs = np.append(f4_outputs, -0.4210788523009721)
f5_inputs = np.vstack([f5_inputs, [0.147370, 0.999999, 0.999999, 0.921713]]); f5_outputs = np.append(f5_outputs, 3528.4781492130905)
f6_inputs = np.vstack([f6_inputs, [0.340766, 0.435136, 0.496022, 0.778382, 0.170336]]); f6_outputs = np.append(f6_outputs, -0.42016788386372284)
f7_inputs = np.vstack([f7_inputs, [0.000001, 0.272669, 0.358553, 0.369157, 0.348012, 0.744467]]); f7_outputs = np.append(f7_outputs, 2.2261133616708637)
f8_inputs = np.vstack([f8_inputs, [0.199786, 0.044128, 0.211922, 0.166001, 0.999999, 0.563724, 0.191632, 0.598870]]); f8_outputs = np.append(f8_outputs, 9.9242862310565)

f1_inputs = np.vstack([f1_inputs, [0.530990, 0.942440]]); f1_outputs = np.append(f1_outputs, -1.2043236571944451e-75)
f2_inputs = np.vstack([f2_inputs, [0.694085, 0.766695]]); f2_outputs = np.append(f2_outputs, 0.48117013551504484)
f3_inputs = np.vstack([f3_inputs, [0.945059, 0.189242, 0.792464]]); f3_outputs = np.append(f3_outputs, -0.10824985555220536)
f4_inputs = np.vstack([f4_inputs, [0.415627, 0.418658, 0.431211, 0.428589]]); f4_outputs = np.append(f4_outputs, 0.5728577804041781)
f5_inputs = np.vstack([f5_inputs, [0.244817, 0.999999, 0.920287, 0.999999]]); f5_outputs = np.append(f5_outputs, 3526.3082081118246)
f6_inputs = np.vstack([f6_inputs, [0.057099, 0.262978, 0.146599, 0.742611, 0.065139]]); f6_outputs = np.append(f6_outputs, -1.0682313869783)
f7_inputs = np.vstack([f7_inputs, [0.010000, 0.173875, 0.523770, 0.374516, 0.345295, 0.850612]]); f7_outputs = np.append(f7_outputs, 1.9452312141572954)
f8_inputs = np.vstack([f8_inputs, [0.089628, 0.517858, 0.049501, 0.051871, 0.812636, 0.909260, 0.120901, 0.437052]]); f8_outputs = np.append(f8_outputs, 9.6526536769036)

settings = {
    "f1": dict(acq="ucb", beta=6.0, frac_local=0.25, top_k=4, local_sigma=0.18,
               boundary_penalty_weight=0.12, boundary_eps=0.03),

    "f2": dict(acq="ei", xi=0.03, frac_local=0.80, top_k=7, local_sigma=0.10,
               max_step=0.22, min_dist_threshold=0.06,
               boundary_penalty_weight=0.10, boundary_eps=0.03),

    "f3": dict(acq="ei", xi=0.03, frac_local=0.85, top_k=8, local_sigma=0.14,
               interior_eps=0.02, boundary_penalty_weight=0.22, boundary_eps=0.05,
               max_step=0.28, min_dist_threshold=0.06),

    "f4": dict(acq="ei", xi=0.03, frac_local=0.90, top_k=8, local_sigma=0.07,
               max_step=0.16, min_dist_threshold=0.05,
               interpret=True, interpret_repeats=10,
               boundary_penalty_weight=0.10, boundary_eps=0.03),

    "f5": dict(acq="ei", xi=0.05,
               frac_local=1.00,
               top_k=8,
               local_sigma=0.09,
               max_step=0.14,
               interior_eps=0.01,
               min_dist_threshold=0.10,
               interpret=True, interpret_repeats=10,
               boundary_penalty_weight=0.14, boundary_eps=0.05),

    "f6": dict(acq="ei", xi=0.04, frac_local=0.70, top_k=8, local_sigma=0.16,
               interior_eps=0.02,
               max_step=0.35, min_dist_threshold=0.08,
               boundary_penalty_weight=0.12, boundary_eps=0.04),

    "f7": dict(acq="ei", xi=0.03, frac_local=0.80, top_k=9, local_sigma=0.10,
               interior_eps=0.01,
               max_step=0.25, min_dist_threshold=0.06,
               boundary_penalty_weight=0.10, boundary_eps=0.03),

    "f8": dict(acq="ei", xi=0.02,
               frac_local=1.00,
               top_k=10,
               local_sigma=0.08,
               max_step=0.18,
               interior_eps=0.01,
               min_dist_threshold=0.10,
               boundary_penalty_weight=0.10, boundary_eps=0.03),
}

n_cand_default = 260_000
n_cand_high_d = 320_000

x1 = propose_next_point_hybrid(f1_inputs, f1_outputs, n_candidates=n_cand_default, **settings["f1"])
x2 = propose_next_point_hybrid(f2_inputs, f2_outputs, n_candidates=n_cand_default, **settings["f2"])
x3 = propose_next_point_hybrid(f3_inputs, f3_outputs, n_candidates=n_cand_default, **settings["f3"])
x4 = propose_next_point_hybrid(f4_inputs, f4_outputs, n_candidates=n_cand_default, **settings["f4"])
x5 = propose_next_point_hybrid(f5_inputs, f5_outputs, n_candidates=220_000, **settings["f5"])
x6 = propose_next_point_hybrid(f6_inputs, f6_outputs, n_candidates=n_cand_high_d, **settings["f6"])
x7 = propose_next_point_hybrid(f7_inputs, f7_outputs, n_candidates=n_cand_high_d, **settings["f7"])
x8 = propose_next_point_hybrid(f8_inputs, f8_outputs, n_candidates=n_cand_high_d, **settings["f8"])

print("\n--- Week 9 proposed queries (submit these) ---")
print("Function 1:", fmt_point(x1))
print("Function 2:", fmt_point(x2), "(EI)")
print("Function 3:", fmt_point(x3), "(EI + interior)")
print("Function 4:", fmt_point(x4), "(EI)")
print("Function 5:", fmt_point(x5), "(EI, locked local)")
print("Function 6:", fmt_point(x6), "(EI, safer)")
print("Function 7:", fmt_point(x7), "(EI)")
print("Function 8:", fmt_point(x8), "(EI, locked local)")

print("\nMin distances to existing:")
print("F1:", min_dist_to_existing(f1_inputs, x1))
print("F2:", min_dist_to_existing(f2_inputs, x2))
print("F3:", min_dist_to_existing(f3_inputs, x3))
print("F4:", min_dist_to_existing(f4_inputs, x4))
print("F5:", min_dist_to_existing(f5_inputs, x5))
print("F6:", min_dist_to_existing(f6_inputs, x6))
print("F7:", min_dist_to_existing(f7_inputs, x7))
print("F8:", min_dist_to_existing(f8_inputs, x8))