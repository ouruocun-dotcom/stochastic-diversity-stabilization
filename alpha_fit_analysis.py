"""
Supplementary analysis: Inflection-point σ_c and power-law fit
Reproduces Eq.(11): α ≈ 1.4, R² ≈ 0.96
Run AFTER fig2_code.py (or standalone — recomputes response curves).
Output: printed fit results (no figure generated)
Runtime: ~2 min
"""
import numpy as np
from numba import njit
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d

SEED = 42; DT = 1e-3; STEPS = 5000; BURN = 2500; TRIALS = 10; EPS = 1e-10
N_VALUES = [5, 8, 10, 15]  # N ≤ 15 only (see Section 3.2)
SIGMA_SCAN = np.linspace(0, 0.30, 60)

@njit
def run_mean_H(N, sigma, seed, dt, steps, burn, eps):
    np.random.seed(seed)
    x = np.random.dirichlet(np.ones(N))
    H_sum = 0.0; H_count = 0
    for t in range(steps):
        mean_x = np.mean(x)
        f = x * (x - mean_x)
        noise = np.random.standard_normal(N)
        x = x + f*dt + sigma*noise*np.sqrt(dt)
        for i in range(N):
            if x[i] < eps: x[i] = eps
        s = np.sum(x)
        for i in range(N): x[i] /= s
        if t >= burn:
            h = 0.0; ln_N = np.log(N)
            for i in range(N):
                p = max(x[i], eps); h -= p * np.log(p)
            H_sum += h / ln_N; H_count += 1
    return H_sum / H_count if H_count > 0 else 0.0

# JIT warmup
_ = run_mean_H(3, 0.01, 0, 1e-3, 10, 5, 1e-10)

print("Computing response curves and inflection points...")
print("=" * 50)

sigma_c_values = []
for ni, N in enumerate(N_VALUES):
    hc = []
    for si, sig in enumerate(SIGMA_SCAN):
        vals = [run_mean_H(N, sig, SEED + ni*10000 + si*100 + t, DT, STEPS, BURN, EPS)
                for t in range(TRIALS)]
        hc.append(np.mean(vals))
    hc = np.array(hc)

    # Smooth and find inflection point (max of dH/dσ)
    hc_smooth = uniform_filter1d(hc, size=3)
    dH = np.gradient(hc_smooth, SIGMA_SCAN)
    idx = np.argmax(dH)
    sc = SIGMA_SCAN[idx]
    sigma_c_values.append(sc)
    print(f"  N={N:2d}: sigma_c (inflection) = {sc:.4f}")

# Power-law fit: log(σ_c) = -α log(N) + const
Ns = np.array(N_VALUES, dtype=float)
scs = np.array(sigma_c_values)
slope, intercept, r_value, p_value, std_err = linregress(np.log(Ns), np.log(scs))
alpha = -slope
r_squared = r_value**2

print("=" * 50)
print(f"Power-law fit: sigma_c ~ N^(-alpha)")
print(f"  alpha   = {alpha:.2f}")
print(f"  R²      = {r_squared:.4f}")
print(f"  p-value = {p_value:.2e}")
print(f"  (Eq.(11) reports alpha ≈ 1.4, R² ≈ 0.96)")
