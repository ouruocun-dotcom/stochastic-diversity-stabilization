"""
Fig 6 Δt convergence: FIXED — window defined in physical time (1.0 unit),
not in steps. This ensures fair comparison across different Δt.
"""
import numpy as np
from numba import njit

SEED = 42; N = 20; SIGMA = 0.03; EPS = 1e-10
GAMMA = np.linspace(0.25, 2.0, 15)
WINDOW_TIME = 1.0  # physical time per window
EPS_H = 5e-3; N_STABLE = 5; TRIALS = 20
TOTAL_TIME = 60.0

@njit
def run_relaxation(N, gamma, sigma, seed, dt, steps, window_steps, eps_h, n_stable, eps):
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
        if t > 2*window_steps:
            recent = 0.0; prev = 0.0
            for k in range(window_steps):
                recent += H_hist[t - k]; prev += H_hist[t - window_steps - k]
            recent /= window_steps; prev /= window_steps
            if abs(recent - prev) < eps_h:
                stable += 1
            else:
                stable = 0
            if stable >= n_stable:
                return t * dt
    return steps * dt

_ = run_relaxation(5, 1.0, 0.03, 0, 0.01, 100, 10, 5e-3, 3, 1e-10)

print(f"Window = {WINDOW_TIME} time unit (steps adjusted per Δt)")
print(f"{'gamma':>6} | {'dt=0.01':>8} | {'dt=0.005':>8} | {'diff':>6} | {'%diff':>6}")
print("-" * 50)

results = {}
for dt in [0.01, 0.005]:
    steps = int(TOTAL_TIME / dt)
    window_steps = int(WINDOW_TIME / dt)  # 100 for 0.01, 200 for 0.005
    means = []
    for gi, g in enumerate(GAMMA):
        vals = [run_relaxation(N, g, SIGMA, SEED+gi*100+t, dt, steps, window_steps, EPS_H, N_STABLE, EPS)
                for t in range(TRIALS)]
        means.append(np.mean(vals))
    results[dt] = means
    print(f"  dt={dt}: window_steps={window_steps}, total_steps={steps}")

print()
for gi, g in enumerate(GAMMA):
    t1 = results[0.01][gi]
    t2 = results[0.005][gi]
    diff = abs(t1 - t2)
    pct = diff / max(t1, 0.01) * 100
    print(f"{g:6.2f} | {t1:8.2f} | {t2:8.2f} | {diff:6.2f} | {pct:5.1f}%")

gm1 = np.mean(results[0.01])
gm2 = np.mean(results[0.005])
print("-" * 50)
print(f"{'MEAN':>6} | {gm1:8.2f} | {gm2:8.2f} | {abs(gm1-gm2):6.2f} | {abs(gm1-gm2)/gm1*100:5.1f}%")
