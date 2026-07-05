# Stochastic Diversity Stabilization — Simulation Code

Reproducible code for all figures in:

> **Stochastic Diversity Stabilization in Nonlinear Competitive Dynamics:
> A Statistical Mechanics Approach to Fluctuation Suppression and Finite-Size Scaling**
>
> Ruocun Ou, Nanyang Normal University
>
> Submitted to *Physica A: Statistical Mechanics and its Applications*

## Quickest way to reproduce (Google Colab)

All scripts run in [Google Colab](https://colab.research.google.com/) without any setup.
Colab provides `numpy`, `matplotlib`, `scipy`, and `numba` by default.

1. Open a new Colab notebook.
2. Paste any script below into a cell and run.
3. Output figures are saved in the Colab working directory and also displayed inline.

No package installation is needed. Total runtime for all 9 scripts is approximately 10–30 minutes, depending on the Colab instance assigned.

## Local installation (alternative)

If you prefer to run locally:

```bash
pip install numpy matplotlib numba scipy
```

Requires Python >= 3.9.

## Scripts

| Script                  | Output                                                       |
|-------------------------|--------------------------------------------------------------|
| `fig1_code.py`          | `fig1_basic_phenomenology.png`                               |
| `fig2_code.py`          | `fig2_response_curves.png`                                   |
| `fig3_code.py`          | `fig3_universality.png`                                      |
| `fig4_code.py`          | `fig4_combined.png` + `fig4_data.npz`                        |
| `fig6_code.py`          | `fig6_relaxation.png`                                        |
| `figS4S5_code.py`       | `figS4_fluctuation_boundary.png` + `figS5_dt_robustness.png` |
| `figS6_code.py`         | `figS6_fss.png`                                              |
| `alpha_fit_analysis.py` | *(printed output only: α and R² for Eq.(11))*                |
| `fig6_dt_convergence.py` | *(printed output only: τ at Δt=0.01 vs 0.005)*              |

**Note on runtimes:** Individual script runtimes range from under 1 minute to approximately 10 minutes, but vary significantly depending on hardware (CPU type, numba JIT cache state). In our tests on Colab, total runtime for all 9 scripts ranged from 10 to 40 minutes across different sessions. Runtime differences do not affect output values — all results are deterministic.

## Reproducibility

All simulations use fixed random seeds (`SEED = 42`) with `np.random.seed()` inside numba-compiled functions. Running any script multiple times, on any machine, produces identical numerical output and identical figures.

## Notes on simulation parameters

**Stationary-state experiments (Figs 1–4, S4–S6)** use `Δt = 10⁻³` and 6000 steps (total physical time: 6 time units), with a 3000-step burn-in followed by 3000 steps of averaging. Stationary-state observables (entropy mean and variance) are *N*-fold averaged quantities whose running means converge within the burn-in period; the Supplementary S5 convergence check confirms that halving `Δt` does not change the qualitative structure of the results.

**Relaxation experiment (Fig 6)** uses `Δt = 0.01` and 6000 steps (total physical time: 60 time units). The longer integration time is needed because the relaxation-time criterion tracks a single-trajectory running-window average, which requires the macroscopic entropy to have fully stabilized — a more demanding condition than the ensemble-averaged stationary-state observables above.

## License

MIT
