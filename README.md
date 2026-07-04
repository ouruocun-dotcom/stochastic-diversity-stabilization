# Stochastic Diversity Stabilization — Simulation Code

Reproducible code for all figures in:

> **Stochastic Diversity Stabilization in Nonlinear Competitive Dynamics:
> A Statistical Mechanics Approach to Fluctuation Suppression and Finite-Size Scaling**
>
> Ruocun Ou, Nanyang Normal University
>
> Submitted to *Physica A: Statistical Mechanics and its Applications*

## Requirements

```
Python >= 3.9
numpy >= 1.24
matplotlib >= 3.7
numba >= 0.57
scipy >= 1.10
```

Install all dependencies:

```bash
pip install numpy matplotlib numba scipy
```

**Google Colab:** All scripts run in Colab without modification.
Colab provides `numpy`, `matplotlib`, and `scipy` by default;
only `numba` needs to be installed (`!pip install numba -q`).
Paste each script into a cell and run.

## Figures

| Script            | Output                                                        | Runtime  |
|-------------------|---------------------------------------------------------------|----------|
| `fig1_code.py`    | `fig1_basic_phenomenology.png`                                | ~1 min   |
| `fig2_code.py`    | `fig2_response_curves.png`                                    | ~2 min   |
| `fig3_code.py`    | `fig3_universality.png`                                       | ~5 min   |
| `fig4_code.py`    | `fig4_combined.png` + `fig4_data.npz`                         | ~8 min   |
| `fig6_code.py`    | `fig6_relaxation.png`                                         | ~5 min   |
| `figS4S5_code.py` | `figS4_fluctuation_boundary.png` + `figS5_dt_robustness.png`  | ~15 min  |
| `figS6_code.py`   | `figS6_fss.png`                                               | ~25 min  |
| `alpha_fit_analysis.py` | *(printed output only: α and R² for Eq.(11))* | ~2 min  |

Runtimes are approximate on a modern laptop with numba JIT compilation.
First run may be slower due to JIT warm-up.

## Reproducibility

All simulations use fixed random seeds (`SEED = 42`) with `np.random.seed()` inside
numba-compiled functions. Running any script twice produces identical output.

## Notes on simulation parameters

**Stationary-state experiments (Figs 1–4, S4–S6)** use `Δt = 10⁻³` and 6000 steps
(total physical time: 6 time units), with a 3000-step burn-in followed by 3000 steps
of averaging. Stationary-state observables (entropy mean and variance) are *N*-fold
averaged quantities whose running means converge within the burn-in period; the
Supplementary S5 convergence check confirms that halving `Δt` does not change the
qualitative structure of the results.

**Relaxation experiment (Fig 6)** uses `Δt = 0.01` and 6000 steps (total physical time:
60 time units). The longer integration time is needed because the relaxation-time
criterion tracks a *single-trajectory* running-window average, which requires the
macroscopic entropy to have fully stabilized — a more demanding condition than the
ensemble-averaged stationary-state observables above.

The `numba` package is required for acceptable runtimes; without it, simulations
would take 10–50× longer.

## License

MIT
