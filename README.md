# Stochastic Diversity Stabilization — Simulation Code

Reproducible code for all figures in:

> **Stochastic Diversity Stabilization in Nonlinear Competitive Dynamics: A Statistical Mechanics Approach to Fluctuation Suppression and Finite-Size Scaling**
> Ruocun Ou, Nanyang Normal University
> Submitted to *Physica A: Statistical Mechanics and its Applications*

## Requirements

```
Python >= 3.9
numpy >= 1.24
matplotlib >= 3.7
numba >= 0.57
scipy >= 1.10
```

Install: `pip install numpy matplotlib numba scipy`

## Figures

| Script | Output | Approx. Runtime |
|--------|--------|-----------------|
| `fig1_code.py` | `fig1_basic_phenomenology.png` | 1 min |
| `fig2_code.py` | `fig2_response_curves.png` | 2 min |
| `fig3_code.py` | `fig3_universality.png` | 5 min |
| `fig4_code.py` | `fig4_combined.png` + `fig4_data.npz` | 8 min |
| `fig6_code.py` | `fig6_relaxation.png` | 5 min |
| `figS4S5_code.py` | `figS4_fluctuation_boundary.png` + `figS5_dt_robustness.png` | 15 min |
| `figS6_code.py` | `figS6_fss.png` | 25 min |

Runtimes are approximate on a modern laptop with numba JIT compilation.
First run may be slower due to JIT warm-up.

## Reproducibility

All simulations use fixed random seeds (`SEED = 42` with `np.random.seed()` inside numba-compiled functions). Running any script twice will produce identical output.

## Notes

- **Figs 1–4, S4–S6** (stationary-state observables) use `Δt = 10⁻³`, 6000 steps (6 time units), with ≥5 trials per parameter point.
- **Fig 6** (relaxation dynamics) uses `Δt = 0.01`, 6000 steps (60 time units), to ensure macroscopic convergence of the entropy relaxation criterion.
- The `numba` package is required for acceptable runtimes; without it, simulations would take 10–50× longer.

## License

MIT
