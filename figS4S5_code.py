"""
Supplementary Figures S4 and S5
S4: Fluctuation boundary σ*(γ)
S5: dt robustness check (Δt=1e-3 vs 5e-4)
Output: figS4_fluctuation_boundary.png, figS5_dt_robustness.png
Runtime: ~15 min total (with numba JIT)
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

SEED = 42; STEPS = 6000; BURN = 3000; TRIALS = 10; N = 20; EPS = 1e-10
GAMMA = np.linspace(0.5, 2.0, 14)
SIGMA = np.linspace(0.0, 0.08, 35)

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

def scan_grid(N, dt, trials, seed_offset=0):
    ng, ns = len(GAMMA), len(SIGMA)
    mean_H = np.zeros((ng, ns)); chi_H = np.zeros((ng, ns))
    count = 0; total = ng*ns
    for gi, g in enumerate(GAMMA):
        for si, s in enumerate(SIGMA):
            ts = 0.0; tsq = 0.0; tc = 0
            for trial in range(trials):
                seed = SEED + seed_offset + gi*1000 + si*100 + trial
                hs, hsq, hc = run_trial_stats(N, g, s, seed, dt,
                              int(STEPS*(1e-3/dt)), int(BURN*(1e-3/dt)), EPS)
                ts += hs; tsq += hsq; tc += hc
            pm = ts/tc; pv = tsq/tc - pm**2
            mean_H[gi, si] = pm; chi_H[gi, si] = N*pv
            count += 1
            if count % 49 == 0 or count == total:
                print(f'  {count}/{total} ({100*count/total:.0f}%)')
    return mean_H, chi_H

# ── S4 ──
print('S4: Fluctuation boundary (dt=0.001)...')
mean_H_1, chi_H_1 = scan_grid(N, 1e-3, TRIALS)
peak_sigma_idx = chi_H_1.argmax(axis=1)
peak_sigma = SIGMA[peak_sigma_idx]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(GAMMA, peak_sigma, 'o-', color='#2980b9', lw=2, ms=6)
ax.set_xlabel('Nonlinearity $\\gamma$', fontsize=14)
ax.set_ylabel('$\\sigma$ at peak $\\chi_H$', fontsize=14)
ax.tick_params(labelsize=12); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figS4_fluctuation_boundary.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: figS4_fluctuation_boundary.png')

# ── S5 ──
print('\nS5: dt robustness (dt=0.0005)...')
mean_H_05, chi_H_05 = scan_grid(N, 5e-4, TRIALS, seed_offset=500000)

ext = [SIGMA[0], SIGMA[-1], GAMMA[0], GAMMA[-1]]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for col, (mH, cH, dt_label) in enumerate([
    (mean_H_1, chi_H_1, '$\\Delta t = 10^{-3}$'),
    (mean_H_05, chi_H_05, '$\\Delta t = 5\\times10^{-4}$')]):
    im0 = axes[0, col].imshow(mH, origin='lower', aspect='auto', extent=ext, cmap='viridis')
    plt.colorbar(im0, ax=axes[0, col])
    axes[0, col].set_title(f'Mean Entropy ({dt_label})', fontsize=12)
    axes[0, col].set_xlabel('$\\sigma$', fontsize=12); axes[0, col].set_ylabel('$\\gamma$', fontsize=12)
    axes[0, col].tick_params(labelsize=10)
    im1 = axes[1, col].imshow(cH, origin='lower', aspect='auto', extent=ext, cmap='inferno')
    plt.colorbar(im1, ax=axes[1, col])
    axes[1, col].set_title(f'$\\chi_H$ ({dt_label})', fontsize=12)
    axes[1, col].set_xlabel('$\\sigma$', fontsize=12); axes[1, col].set_ylabel('$\\gamma$', fontsize=12)
    axes[1, col].tick_params(labelsize=10)
plt.tight_layout()
plt.savefig('figS5_dt_robustness.png', dpi=300, bbox_inches='tight')
print('Saved: figS5_dt_robustness.png')
