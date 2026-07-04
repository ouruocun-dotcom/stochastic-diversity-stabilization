"""
Figure 1: Basic phenomenology (simplex trajectories + noise response)
Output: fig1_basic_phenomenology.png
Runtime: ~1 min
"""
import numpy as np
import matplotlib.pyplot as plt

SEED = 42; DT = 0.01; STEPS = 5000; N = 3; EPS = 1e-8

def run_simplex(sigma, seed):
    rng = np.random.default_rng(seed)
    x = np.array([0.34, 0.33, 0.33])
    traj = [x.copy()]
    for _ in range(STEPS):
        f = x * (x - x.mean())
        x = x + f*DT + sigma*rng.standard_normal(N)*np.sqrt(DT)
        x = np.maximum(x, EPS); x /= x.sum()
        traj.append(x.copy())
    return np.array(traj)

def to_2d(traj):
    return traj[:,0] + 0.5*traj[:,1], np.sqrt(3)/2*traj[:,1]

def normalized_entropy(x):
    p = np.maximum(x, EPS); p /= p.sum()
    return -np.sum(p*np.log(p))/np.log(len(p))

traj_det = run_simplex(0.0, SEED)
traj_sto = run_simplex(0.08, SEED)

sigmas = np.linspace(0, 0.25, 50)
H_mean = []
for sig in sigmas:
    rng = np.random.default_rng(SEED)
    x = rng.dirichlet(np.ones(N))
    H_vals = []
    for t in range(STEPS):
        f = x*(x - x.mean())
        x = x + f*DT + sig*rng.standard_normal(N)*np.sqrt(DT)
        x = np.maximum(x, EPS); x /= x.sum()
        if t > 1000:
            H_vals.append(normalized_entropy(x))
    H_mean.append(np.mean(H_vals))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_i, (traj, label, color) in enumerate([
    (traj_det, 'Deterministic', '#2c3e50'),
    (traj_sto, 'Stochastic ($\\sigma=0.08$)', '#2980b9')]):
    ax = axes[0]
    tri_x = [0, 1, 0.5, 0]; tri_y = [0, 0, np.sqrt(3)/2, 0]
    if ax_i == 0: ax.plot(tri_x, tri_y, 'k-', lw=1.5)
    xx, yy = to_2d(traj)
    ax.plot(xx, yy, color=color, lw=0.5, alpha=0.7, label=label)
    ax.plot(xx[0], yy[0], 'o', color=color, ms=5, zorder=5)

axes[0].set_xlim(-0.05, 1.05); axes[0].set_ylim(-0.05, 0.95)
axes[0].set_aspect('equal'); axes[0].legend(fontsize=11, loc='upper right')
axes[0].set_xlabel('$x_1 + 0.5\\,x_2$', fontsize=13)
axes[0].set_ylabel('$\\sqrt{3}/2\\;x_2$', fontsize=13)
axes[0].tick_params(labelsize=11)
axes[0].text(0.02, 0.95, '(a)', transform=axes[0].transAxes, fontsize=15, fontweight='bold', va='top')

axes[1].plot(sigmas, H_mean, '-', color='#2980b9', lw=2)
axes[1].set_xlabel('$\\sigma$', fontsize=13)
axes[1].set_ylabel('$\\langle H \\rangle_\\mathrm{ss}$', fontsize=13)
axes[1].tick_params(labelsize=11); axes[1].grid(True, alpha=0.3)
axes[1].text(0.02, 0.95, '(b)', transform=axes[1].transAxes, fontsize=15, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('fig1_basic_phenomenology.png', dpi=300, bbox_inches='tight')
print('Saved: fig1_basic_phenomenology.png')
