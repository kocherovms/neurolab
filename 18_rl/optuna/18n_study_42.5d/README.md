# 18n_study_42.5d — annealing SHAPE of `learn_rate` (draft, not launched)

5-й шаг `42`-линейки, продолжение `42.4` (→ `42.5b` / `42.5c`). Parent = best agent of `42.4`
(`18n_ppo_tr_frostbite_07_old:160`). Target notebook: `18n_ppo_tr_frostbite_07_old.ipynb`
(VisionHead/encoder base `18d_world_model_09:183`, VH frozen).

## The one axis under test

**The SHAPE of the `learn_rate` schedule inside a step** — `const` vs linear decay vs warmup+cosine.
`ent_coef` is held `const(0.023)` in every arm, so only the lr curve moves.

## Why this design (the confound we split)

`42.5c` swept `const(lr)` and the metric fell monotonically with lr — but a low `const` can win
simply because it **damages the warm-started parent less** (weak-shock + less AdamW weight decay of
inherited weights). That is a **level** effect, not a **shape** effect. So the arms are
**level-matched**: `const_match`, `linear_down`, `shark_fin` all have the same *time-averaged* lr
(≈ 5.5e-5) and differ **only in shape**.

| arm (`schedule`) | value | time-mean lr | what it probes |
|---|---|---|---|
| `const_lo` | `const(0.00003)` | 3.00e-5 | low flat control (42.5c favourite) — the *level* knob |
| `const_match` | `const(0.000055)` | 5.50e-5 | level-matched flat control — the *shape=flat* baseline |
| `linear_down` | `linear(0.0001, 0.00001)` | 5.50e-5 | level-matched, straight decay |
| `shark_fin` | `shark_fin(0.0001, 0.00001, 0.05)` | 5.47e-5 | level-matched, warmup → cosine ("cool the pan") |
| `shark_fin_hot` | `shark_fin(0.00025, 0.00001, 0.05)` | 1.30e-4 | same shape from a hotter peak (cold-Adam / task-shift arm) |

Hypothesis: warmup→cosine crystallizes the policy better than flat at the *same* average lr.

## Replications (paired design)

`random_seed` is also sampled from the grid → **11 seeds × 5 arms = 55 cells**, `launches_count = 50`
→ **10 runs per arm**, balanced (verified). Arms share seeds, so the comparison is paired.

Everything else is identical to `42.5c` except:
- parent is hardcoded `:160` (as in `42.5c`), not sampled;
- `clip_coef` fixed at `0.12` (42.5c found clip ~0.12–0.15 best; keeps it off the noise axes);
- `lr`/`ent` `suggest_float` replaced by the `schedule` categorical + seed categorical.

## How to read the result

- **`shark_fin` vs `const_match`** and **`linear_down` vs `const_match`** (paired by seed; Welch +
  paired) → does **shape** matter at equal average lr?
- **`const_lo` vs `const_match`** → value of simply going *lower* (level, not shape).
- If shapes ≈ `const_match` and `const_lo` is best → **level** rules and "annealing" is a warm-start
  illusion. If `shark_fin` (or `linear_down`) > `const_match` → **shape** matters and the "cooling
  pan" metaphor holds literally.
- `shark_fin_hot` tells whether a high start **hurts** (parent-shock) even when it anneals down.

## Open questions (for Boss)

1. Number of arms / launches (currently 5 × 10 = 50). More seeds per arm would sharpen paired tests
   at the cost of compute — 10/arm is the balance point given the ~0.44 sd of the metric.
2. Should the *step-boundary* re-warm be tested here too (chain-level schedule), or keep 42.5d
   strictly within-step?
3. `decisive_metric` is `test/levels_passed_mean` (cold 30-env test, as in the chain). A group
   cold-level5 slice could be added later.

## Status

Draft only. No `.optuna` journal, not launched. Validation passed: valid JSON (byte-identical
round-trip), `set_hyperparameters` compiles, one live `@launchit.collect`, all schedule strings parse
against the notebook's `hp_parse_universal_module` grammar, `nine_lives` patch exists, GridSampler
gives a balanced 10/arm split at `launches_count = 50`.
