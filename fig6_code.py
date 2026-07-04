"""
Figure 6: Relaxation dynamics — τ_relax(γ) with SE bars and boxplot
Output: fig6_relaxation.png
Runtime: ~5 min (with numba JIT)
Note: Uses dt=0.01, 6000 steps = 60 time units total,
      with relaxed convergence criterion EPS_H=5e-3.
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

SEED = 42; N = 20; SIGMA = 0.03; EPS = 1e-10
GAMMA = np.linspace(0.25, 2.0, 15)
DT = 0.01; STEPS = 6000  # 60 time units
WINDOW = 100; EPS_H = 5e-3; N_STABLE = 5
TRIALS_5 = 5; TRIALS_20 = 20

@njit
def run_relaxation(N, gamma, sigma, seed, dt, steps, window, eps_h, n_stable, eps):
    np.random.seed(seed)
    x = np.random.dirichlet(np.ones(N))
    ln_N = np.log(N); H_hist = np.empty(steps); stable = 0
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
        h = 0.0
        for i in range(N):
            p = max(x[i], eps); h -= p * np.log(p)
        H_hist[t] = h / ln_N
        if t > 2*window:
            recent = 0.0; prev = 0.0
            for k in range(window):
                recent += H_hist[t - k]; prev += H_hist[t - window - k]
            recent /= window; prev /= window
            if abs(recent - prev) < eps_h:
                stable += 1
            else:
                stable = 0
            if stable >= n_stable:
                return t * dt
    return steps * dt

# JIT warmup
_ = run_relaxation(5, 1.0, 0.03, 0, 0.01, 100, 10, 5e-3, 3, 1e-10)

print('Panel (a): 5 trials')
tau_5 = []
for gi, g in enumerate(GAMMA):
    vals = [run_relaxation(N, g, SIGMA, SEED+gi*100+t, DT, STEPS, WINDOW, EPS_H, N_STABLE, EPS)
            for t in range(TRIALS_5)]
    tau_5.append(np.mean(vals))
    print(f'  gamma={g:.2f}: tau={np.mean(vals):.2f}')

print('\nPanels (b)(c): 20 trials')
tau_all = {}
for gi, g in enumerate(GAMMA):
    vals = [run_relaxation(N, g, SIGMA, SEED+gi*100+t, DT, STEPS, WINDOW, EPS_H, N_STABLE, EPS)
            for t in range(TRIALS_20)]
    tau_all[g] = vals
    print(f'  gamma={g:.2f}: mean={np.mean(vals):.2f}, SD={np.std(vals):.2f}')

means_20 = [np.mean(tau_all[g]) for g in GAMMA]
ses_20 = [np.std(tau_all[g])/np.sqrt(len(tau_all[g])) for g in GAMMA]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].plot(GAMMA, tau_5, 'o-', color='#2980b9', lw=2, ms=5)
axes[0].set_xlabel('$\\gamma$', fontsize=14); axes[0].set_ylabel('$\\tau_\\mathrm{relax}$', fontsize=14)
axes[0].tick_params(labelsize=12); axes[0].grid(True, alpha=0.3)
axes[0].text(0.03, 0.95, '(a)', transform=axes[0].transAxes, fontsize=15, fontweight='bold', va='top')

axes[1].errorbar(GAMMA, means_20, yerr=ses_20, fmt='o-', color='#2980b9', lw=2, ms=5, capsize=3, capthick=1.5)
axes[1].set_xlabel('$\\gamma$', fontsize=14); axes[1].set_ylabel('$\\tau_\\mathrm{relax}$', fontsize=14)
axes[1].tick_params(labelsize=12); axes[1].grid(True, alpha=0.3)
axes[1].text(0.03, 0.95, '(b)', transform=axes[1].transAxes, fontsize=15, fontweight='bold', va='top')

bp_data = [tau_all[g] for g in GAMMA]
bp = axes[2].boxplot(bp_data, positions=range(len(GAMMA)), widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='#d5e8f0', edgecolor='#2980b9'),
                     medianprops=dict(color='#e74c3c', lw=2),
                     whiskerprops=dict(color='#2980b9'), capprops=dict(color='#2980b9'),
                     flierprops=dict(marker='o', markerfacecolor='#95a5a6', markersize=3, alpha=0.5))
axes[2].set_xticks(range(len(GAMMA)))
axes[2].set_xticklabels([f'{g:.1f}' if i%3==0 else '' for i,g in enumerate(GAMMA)], fontsize=10)
axes[2].set_xlabel('$\\gamma$', fontsize=14); axes[2].set_ylabel('$\\tau_\\mathrm{relax}$', fontsize=14)
axes[2].tick_params(labelsize=12); axes[2].grid(True, alpha=0.3, axis='y')
axes[2].text(0.03, 0.95, '(c)', transform=axes[2].transAxes, fontsize=15, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('fig6_relaxation.png', dpi=300, bbox_inches='tight')
print('\nSaved: fig6_relaxation.png')
