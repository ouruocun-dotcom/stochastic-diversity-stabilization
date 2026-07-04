"""
Figure 4: Phase diagram + fluctuation analysis (2x2 combined)
Output: fig4_combined.png
Runtime: ~8 min (with numba JIT)
Also saves raw data to fig4_data.npz for use by figS4S5 and figS6.
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

SEED = 42; DT = 1e-3; STEPS = 6000; BURN = 3000; TRIALS = 20; N = 20; EPS = 1e-10
GAMMA = np.linspace(0.5, 2.0, 14)
SIGMA = np.linspace(0.0, 0.08, 35)

@njit
def run_trial_stats(N, gamma, sigma, seed, dt, steps, burn, eps):
    """Returns (sum_H, sum_H², count) for pooling across trials."""
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

# JIT warmup
_ = run_trial_stats(5, 1.0, 0.01, 0, 1e-3, 100, 50, 1e-10)

ng, ns = len(GAMMA), len(SIGMA)
mean_H = np.zeros((ng, ns)); var_H = np.zeros((ng, ns)); chi_H = np.zeros((ng, ns))
total = ng*ns; count = 0

print(f'Grid: {ng}x{ns}={total} points, {TRIALS} trials each')
for gi, g in enumerate(GAMMA):
    for si, s in enumerate(SIGMA):
        ts = 0.0; tsq = 0.0; tc = 0
        for trial in range(TRIALS):
            seed = SEED + N*100000 + gi*1000 + si*100 + trial
            hs, hsq, hc = run_trial_stats(N, g, s, seed, DT, STEPS, BURN, EPS)
            ts += hs; tsq += hsq; tc += hc
        pm = ts/tc; pv = tsq/tc - pm**2
        mean_H[gi, si] = pm; var_H[gi, si] = pv; chi_H[gi, si] = N*pv
        count += 1
        if count % 49 == 0 or count == total:
            print(f'  {count}/{total} ({100*count/total:.0f}%)')

np.savez('fig4_data.npz', mean_H=mean_H, var_H=var_H, chi_H=chi_H,
         GAMMA=GAMMA, SIGMA=SIGMA)

peak_chi = chi_H.max(axis=1)
gi_pk, si_pk = np.unravel_index(chi_H.argmax(), chi_H.shape)
ext = [SIGMA[0], SIGMA[-1], GAMMA[0], GAMMA[-1]]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

im0 = axes[0,0].imshow(mean_H, origin='lower', aspect='auto', extent=ext, cmap='viridis')
cb0 = plt.colorbar(im0, ax=axes[0,0]); cb0.set_label('$\\langle H \\rangle_\\mathrm{ss}$', fontsize=13)
cb0.ax.tick_params(labelsize=11)
axes[0,0].set_xlabel('Noise strength $\\sigma$', fontsize=14)
axes[0,0].set_ylabel('Nonlinearity $\\gamma$', fontsize=14)
axes[0,0].tick_params(labelsize=12)
axes[0,0].text(0.03, 0.95, '(a)', transform=axes[0,0].transAxes, fontsize=15, fontweight='bold',
               va='top', color='white', bbox=dict(facecolor='black', alpha=0.4, pad=2, edgecolor='none'))

im1 = axes[0,1].imshow(var_H, origin='lower', aspect='auto', extent=ext, cmap='plasma')
cb1 = plt.colorbar(im1, ax=axes[0,1]); cb1.set_label('Var$(H)$', fontsize=13)
cb1.ax.tick_params(labelsize=11)
axes[0,1].set_xlabel('$\\sigma$', fontsize=14); axes[0,1].set_ylabel('$\\gamma$', fontsize=14)
axes[0,1].tick_params(labelsize=12)
axes[0,1].text(0.03, 0.95, '(b)', transform=axes[0,1].transAxes, fontsize=15, fontweight='bold',
               va='top', color='white', bbox=dict(facecolor='black', alpha=0.4, pad=2, edgecolor='none'))

im2 = axes[1,0].imshow(chi_H, origin='lower', aspect='auto', extent=ext, cmap='plasma')
cb2 = plt.colorbar(im2, ax=axes[1,0]); cb2.set_label('$\\chi_H = N\\,\\mathrm{Var}(H)$', fontsize=13)
cb2.ax.tick_params(labelsize=11)
axes[1,0].set_xlabel('$\\sigma$', fontsize=14); axes[1,0].set_ylabel('$\\gamma$', fontsize=14)
axes[1,0].tick_params(labelsize=12)
axes[1,0].text(0.03, 0.95, '(c)', transform=axes[1,0].transAxes, fontsize=15, fontweight='bold',
               va='top', color='white', bbox=dict(facecolor='black', alpha=0.4, pad=2, edgecolor='none'))

axes[1,1].plot(GAMMA, peak_chi, 'o-', color='#2980b9', lw=2, ms=6)
axes[1,1].axvline(GAMMA[gi_pk], color='#e74c3c', ls='--', lw=1.2, alpha=0.6)
axes[1,1].set_xlabel('$\\gamma$', fontsize=14); axes[1,1].set_ylabel('Peak $\\chi_H$', fontsize=14)
axes[1,1].tick_params(labelsize=12); axes[1,1].grid(True, alpha=0.3)
axes[1,1].text(0.03, 0.95, '(d)', transform=axes[1,1].transAxes, fontsize=15, fontweight='bold', va='top')

plt.tight_layout(pad=1.5)
plt.savefig('fig4_combined.png', dpi=300, bbox_inches='tight')
print(f'\nSaved: fig4_combined.png')
print(f'Peak chi_H = {chi_H.max():.4f} at gamma={GAMMA[gi_pk]:.3f}, sigma={SIGMA[si_pk]:.4f}')
