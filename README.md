# dt-zero-trust-lab

Simulation code for the paper:

> A. K. Mahamed, R. Subramani, and S. P. K. Mandapaka, "Digital-Twin-Driven Experiential Learning for Cyber-Physical Security: Teaching Zero Trust Through Real-Time Industrial Process Consequences," in *Proc. 19th IEEE MCSoC*, 2026.

Learners make a Zero-Trust access decision (allow, step-up, or deny) using an explainable six-factor risk score, then see what that decision does to a simulated heated mixing tank. The tank has two PI loops in the PLC and an independent safety instrumented function (SIF). The code reproduces every number, table, and figure in the paper from a fixed seed (`20261004`).

## Quick start

```bash
pip install -r requirements.txt
python run_all.py        # writes results.json (about 1 min on one core)
python make_figures.py   # writes fig2_temperature.png and fig3_loss.png
```

Each script can also be run on its own:

| Script | What it does | Paper section |
|---|---|---|
| `twin_sim.py` | Process twin, SIF, four scenarios, risk score, request population, policy comparison, weight sensitivity, seed/stealth robustness | III, V-B, V-C (V2, V4) |
| `window.py` | Response-delay sweep (0–900 s, 1 s steps) and safe response windows | V-A (V1), Table V |
| `batch_twin.py` | Vectorized twin, timing, scalar/batch exact-match check, two-core scaling, memory footprint | V-D (V5) |
| `param_robust.py` | ±20 % plant-physics corners (β, γ, c_v, q_max) | V-E (V6) |
| `run_all.py` | Runs everything, writes `results.json`, prints the SHA-256 used for the determinism check | V-C (V3) |
| `make_figures.py` | Fig. 2 and Fig. 3 | V-A |

## Determinism (V3)

```bash
python run_all.py --hash
python run_all.py --hash   # same digest every time
```

The hash covers every deterministic result; timing is machine dependent and excluded. The paper reports the prefix `614ac8ce0807fcef`.

## Model summary

| Quantity | Value |
|---|---|
| Tank area A / max inflow q_max | 2.0 m² / 0.03 m³/s |
| Outflow coefficient c_v | 0.01 m^2.5/s |
| Heater gain β / loss γ | 0.6 K·m/s / 5×10⁻⁴ s⁻¹ |
| Feed / ambient temperature | 20 °C / 20 °C |
| Setpoints L / T | 1.5 m / 60 °C (spec ±5 °C) |
| Level PI gains K_P / K_I | 2.0 m⁻¹ / 0.01 m⁻¹s⁻¹ |
| Temperature PI gains K_P / K_I | 0.25 K⁻¹ / 0.005 K⁻¹s⁻¹ |
| PI implementation | output clamped to [0, 1], conditional-integration anti-windup |
| SIF trips | T ≥ 85 °C or L ≥ 2.5 m (independent shutoff of inlet and heater) |
| Overflow / restart after trip | 3.0 m / 15 min |
| Step / episode / attack time | 1 s (explicit Euler) / 40 min / t = 5 min |

Scenarios at magnitude m ∈ [0, 1] (full magnitude m = 1):

* **Credential misuse:** T setpoint raised to 60 + 35m °C until the response.
* **Unsafe engineering:** level SIF bypassed and L setpoint raised to 1.5 + 1.6m m until the response. The temperature SIF stays active.
* **Ransomware:** HMI lost at the attack. 120 s later the inlet valve sticks at 0.4 + 0.6m open (fully open at m = 1). The HMI and the valve are restored together at the response.
* **Vendor firmware:** PLC outputs frozen until the response. Feed temperature drops to 20 − 12m °C at the attack and stays there.

Production loss is the time off specification before any trip, plus 15 min if the SIF trips.

## Scope

The twin is a two-state teaching model. It is not calibrated against a real plant. The request population is synthetic, and its distributions are set in `twin_sim.py`. See the paper's Limitations section.

## License

MIT. See `LICENSE`. If you use this code, please cite the paper (`CITATION.cff`).
