"""
Supplementary Figure S6: Finite-size verification (N=10, 20, 40)
Output: figS6_fss.png
Runtime: ~25 min total (with numba JIT)
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

SEED = 42; DT = 1e-3; STEPS = 6000; BURN = 3000; TRIALS = 20; EPS = 1e-10
GAMMA = np.linspace(0.5, 2.0, 14)
SIGMA = np.linspace(0.0, 0.08, 35)
N_VALUES = [10, 20, 40]

@njit
def run_trial_stats(N, gamma, sigma, seed, dt, steps, burn, eps):
    np.random.seed(seed)
    x = np.random.dirichlet(np.ones(N))
    ln_N = np.log(N); H_sum = 0.0; H_sq = 0.0; count = 0
    for t in range(steps):
        xg = np.empty(N)
        for i in range(N): xg[i] = x[i]**gamma
        mean_xg = np.mean(xg)
        for i in range(N):
            x[i] += x[i]*(xg[i] - mean_xg)*dt + sigma*np.random.standard_normal()*np.sqrt(dt)
            if x[i] < eps: x[i] = eps
        s = 0.0
        for i in range(N): s += x[i]
        for i in range(N): x[i] /= s
        if t >= burn:
            h = 0.0
            for i in range(N):
                p = max(x[i], eps); h -= p * np.log(p)
            h /= ln_N; H_sum += h; H_sq += h*h; count += 1
    return H_sum, H_sq, count

_ = run_trial_stats(5, 1.0, 0.01, 0, 1e-3, 100, 50, 1e-10)

results = {}
for N in N_VALUES:
    print(f'\n=== N={N} ===')
    ng, ns = len(GAMMA), len(SIGMA)
    mean_H = np.zeros((ng, ns)); chi_H = np.zeros((ng, ns))
    count = 0; total = ng*ns
    for gi, g in enumerate(GAMMA):
        for si, s in enumerate(SIGMA):
            ts = 0.0; tsq = 0.0; tc = 0
            for trial in range(TRIALS):
                seed = SEED + N*100000 + gi*1000 + si*100 + trial
                hs, hsq, hc = run_trial_stats(N, g, s, seed, DT, STEPS, BURN, EPS)
                ts += hs; tsq += hsq; tc += hc
            pm = ts/tc; pv = tsq/tc - pm**2
            mean_H[gi, si] = pm; chi_H[gi, si] = N*pv
            count += 1
            if count % 98 == 0 or count == total:
                print(f'  {count}/{total} ({100*count/total:.0f}%)')
    results[N] = (mean_H, chi_H)
    gi_pk = np.unravel_index(chi_H.argmax(), chi_H.shape)[0]
    print(f'  Peak chi_H={chi_H.max():.3f} at gamma={GAMMA[gi_pk]:.3f}')

ext = [SIGMA[0], SIGMA[-1], GAMMA[0], GAMMA[-1]]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for col, N in enumerate(N_VALUES):
    mean_H, chi_H = results[N]
    im0 = axes[0, col].imshow(mean_H, origin='lower', aspect='auto', extent=ext, cmap='viridis')
    cb0 = plt.colorbar(im0, ax=axes[0, col], shrink=0.85); cb0.ax.tick_params(labelsize=9)
    axes[0, col].set_title(f'$N={N}$:  $\\langle H \\rangle_{{\\mathrm{{ss}}}}$', fontsize=13)
    axes[0, col].set_xlabel('$\\sigma$', fontsize=12); axes[0, col].set_ylabel('$\\gamma$', fontsize=12)
    axes[0, col].tick_params(labelsize=10)
    im1 = axes[1, col].imshow(chi_H, origin='lower', aspect='auto', extent=ext, cmap='inferno')
    cb1 = plt.colorbar(im1, ax=axes[1, col], shrink=0.85); cb1.ax.tick_params(labelsize=9)
    axes[1, col].set_title(f'$N={N}$:  $\\chi_H = N\\,\\mathrm{{Var}}(H)$', fontsize=13)
    axes[1, col].set_xlabel('$\\sigma$', fontsize=12); axes[1, col].set_ylabel('$\\gamma$', fontsize=12)
    axes[1, col].tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('figS6_fss.png', dpi=300, bbox_inches='tight')
print('\nSaved: figS6_fss.png')
