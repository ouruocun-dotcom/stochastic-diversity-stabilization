"""
Figure 3: Universality test — σ_c(N) at γ=2 and α(γ)
Output: fig3_universality.png
Runtime: ~5 min (with numba JIT)
Note: Uses dt=0.01 (matching original experimental notebooks)
      with 5 trials per point for reproducible averaging.
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.stats import linregress

N_VALUES = [5, 8, 10, 15, 20]
GAMMA_VALUES = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
TRIALS = 5; SEED = 42; EPS = 1e-8; H_THR = 0.7

# Panel (a): dt=0.01, steps=4000, burn=3000
DT_A = 0.01; STEPS_A = 4000; BURN_A = 3000
SIGMA_SCAN_A = np.linspace(0.0, 0.12, 25)

# Panel (b): dt=0.01, steps=3000, burn=2200
DT_B = 0.01; STEPS_B = 3000; BURN_B = 2200
SIGMA_SCAN_B = np.linspace(0.0, 0.12, 20)

@njit
def find_sigma_c(N, gamma, sigma_scan, dt, steps, burn, seed, eps, h_thr, trials):
    for si in range(len(sigma_scan)):
        sig = sigma_scan[si]
        H_total = 0.0; H_count = 0
        for trial in range(trials):
            np.random.seed(seed + si*1000 + trial)
            x = np.random.dirichlet(np.ones(N))
            for t in range(steps):
                xg = np.empty(N)
                for i in range(N): xg[i] = x[i]**gamma
                mean_xg = np.mean(xg)
                for i in range(N):
                    x[i] += x[i]*(xg[i] - mean_xg)*dt + sig*np.random.standard_normal()*np.sqrt(dt)
                    if x[i] < eps: x[i] = eps
                s = 0.0
                for i in range(N): s += x[i]
                for i in range(N): x[i] /= s
                if t > burn:
                    ln_N = np.log(N); h = 0.0
                    for i in range(N):
                        p = max(x[i], eps); h -= p * np.log(p)
                    H_total += h / ln_N; H_count += 1
        if H_count > 0 and H_total / H_count > h_thr:
            return sig
    return 0.0

# JIT warmup
_ = find_sigma_c(3, 1.0, np.array([0.01]), 0.01, 100, 50, 0, 1e-8, 0.7, 1)

# Panel (a)
print('Panel (a): sigma_c(N) for gamma=2.0')
sc_gamma2 = []
for N in N_VALUES:
    sc = find_sigma_c(N, 2.0, SIGMA_SCAN_A, DT_A, STEPS_A, BURN_A,
                      SEED + N*10000, EPS, H_THR, TRIALS)
    sc_gamma2.append(sc)
    print(f'  N={N}: sigma_c={sc:.4f}')

# Panel (b)
print('\nPanel (b): alpha vs gamma')
alphas = []
for gamma in GAMMA_VALUES:
    scs = []
    for N in N_VALUES:
        sc = find_sigma_c(N, gamma, SIGMA_SCAN_B, DT_B, STEPS_B, BURN_B,
                          SEED + int(gamma*10000) + N*100, EPS, H_THR, TRIALS)
        scs.append(sc)
    scs = np.array(scs, dtype=float); Ns = np.array(N_VALUES, dtype=float)
    scs_clip = np.clip(scs, 1e-6, None)
    mask = scs > 0
    if mask.sum() >= 2:
        sl, *_ = linregress(np.log(Ns[mask]), np.log(scs_clip[mask]))
        alpha = -sl
    else:
        alpha = np.nan
    alphas.append(alpha)
    print(f'  gamma={gamma:.1f}: alpha={alpha:.2f}')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(N_VALUES, sc_gamma2, 'o-', color='#2980b9', lw=2, ms=7)
axes[0].set_xlabel('System size $N$', fontsize=14)
axes[0].set_ylabel('Critical noise $\\sigma_c$', fontsize=14)
axes[0].tick_params(labelsize=12); axes[0].grid(True, alpha=0.3)
axes[0].text(0.02, 0.95, '(a)', transform=axes[0].transAxes, fontsize=15, fontweight='bold', va='top')

valid = [(g, a) for g, a in zip(GAMMA_VALUES, alphas) if not np.isnan(a)]
if valid:
    gv, av = zip(*valid)
    axes[1].plot(gv, av, 'o-', color='#e74c3c', lw=2, ms=7)
axes[1].set_xlabel('Interaction nonlinearity $\\gamma$', fontsize=14)
axes[1].set_ylabel('Scaling exponent $\\alpha$', fontsize=14)
axes[1].tick_params(labelsize=12); axes[1].grid(True, alpha=0.3)
axes[1].text(0.02, 0.95, '(b)', transform=axes[1].transAxes, fontsize=15, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('fig3_universality.png', dpi=300, bbox_inches='tight')
print('\nSaved: fig3_universality.png')
