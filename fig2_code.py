"""
Figure 2: Noise-entropy response curves for N=5..50
Output: fig2_response_curves.png
Runtime: ~2 min (with numba JIT)
Dependencies: numpy, matplotlib, numba
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

SEED = 42; DT = 1e-3; STEPS = 5000; BURN = 2500; TRIALS = 10; EPS = 1e-10
N_VALUES = [5, 8, 10, 15, 20, 30, 50]
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

curves = {}
for ni, N in enumerate(N_VALUES):
    print(f'N={N}...', end=' ', flush=True)
    hc = []
    for si, sig in enumerate(SIGMA_SCAN):
        vals = [run_mean_H(N, sig, SEED + ni*10000 + si*100 + t, DT, STEPS, BURN, EPS)
                for t in range(TRIALS)]
        hc.append(np.mean(vals))
    curves[N] = np.array(hc)
    print(f'done (H0={hc[0]:.3f})')

fig, ax = plt.subplots(figsize=(8, 5.5))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_VALUES)))
for ni, N in enumerate(N_VALUES):
    ax.plot(SIGMA_SCAN, curves[N], '-', color=colors[ni], lw=2.2, label=f'$N={N}$')
ax.set_xlabel('Noise strength $\\sigma$', fontsize=14)
ax.set_ylabel('$\\langle H \\rangle_\\mathrm{ss}$', fontsize=14)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11, ncol=2, loc='lower right')
ax.grid(True, alpha=0.3); ax.set_xlim(0, 0.30); ax.set_ylim(0.25, 0.95)
plt.tight_layout()
plt.savefig('fig2_response_curves.png', dpi=300, bbox_inches='tight')
print('Saved: fig2_response_curves.png')
