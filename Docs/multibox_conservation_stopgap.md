# Multi-box scalar-conservation stopgap (anelastic)

**Status:** ad-hoc mitigation, not a root-cause fix. Enabled by default in the anelastic
(`fixed_density`) path; controlled by the input knob `erf.conservation_fix_coeff`
(`1.0` = on/full, `0.0` = off; see §3 Toggle).

This note documents the *fix* (applied in this fork). The underlying *bug* and a minimal,
stock reproducer are documented separately in the upstream-tracking clone:
`BUG_REPORT_multibox_anelastic_conservation.md` and
`Exec/RegTests/FlowInABox/inputs_multibox_conservation_bug` (measure `int(rhotheta)` with
AMReX's `fvolumesum`: constant for one box, drifts for multiple boxes).

---

## 1. The problem it works around

In the anelastic solver, the domain integral of an advected conserved scalar
(`rho*theta`, `rho*qv`, ...) is **not conserved when the level is split into more than one
grid box** (an AMReX *box* = a rectangular block of cells, sized by `amr.max_grid_size`;
not a cell, MPI rank, or thread). With a single box the integral is conserved to machine
precision; with multiple boxes it drifts monotonically and grows with the flow.

Root cause (see the bug report): the multi-box anelastic projection leaves a small divergence
residual in the advecting momentum (`max|div(rho0 u)| ~ 1e-7` relative; `~1e-17` for one box).
It is a null-space mode — re-projecting (even the direct FFT solve) does not remove it, and it
is identical for FFT and MLMG, so it is **not** a linear-solver accuracy/tolerance issue. The
residual makes the discrete flux-form scalar advection fail to telescope to zero over a closed
domain. Because the anelastic path uses `fixed_density` (the `Rho` advection source is forced
to zero), the matching *mass*-conservation error is hidden; only `rho*theta` and the other
advected scalars expose it.

Physically this manifests as a spurious heating/cooling (and moistening/drying) drift. In the
closed, wall-driven CloudChamber case it produced a steady ~1.5–2 K cold bias because the
spurious sink overwhelmed the (correct) wall heat flux.

---

## 2. What the stopgap does

For a **closed** domain, advection only redistributes a conserved scalar internally — its
domain integral must be exactly constant. So the spurious non-conservation is, by definition,
the nonzero domain integral of the *advective* tendency. The stopgap removes exactly that:

1. **Snapshot** the advection-only tendency of each corrected component, taken right after the
   advection call and **before** diffusion/sources are added (so diffusion and wall fluxes —
   which legitimately change the integral — are excluded).
2. After the per-box loop, compute the spurious **domain-mean** of that advective tendency,
   `m(c) = ( sum_cells advSrc(c) ) / Ncells`.
3. **Subtract** a uniform `fac * m(c)` so the advective contribution integrates to zero.

A uniform shift is the minimum-norm correction that restores the integral; the per-cell change
is tiny (`O(1e-9)`) and does not disturb the local field structure.

### Per-stage RK factor
The anelastic time integrator is a 2-stage RK where the explicit RHS enters with weight `1` at
the first stage (`nrk==0`) and `1/2` at the second (`nrk==1`). The correction uses the same
weight so the per-stage corrections cancel the advective spurious term exactly across the step:

```
fac = (anelastic && nrk == 1) ? 1/2 : 1
```

### Closed-domain guard
The correction is **only valid for a closed domain** (no net advective flux through any domain
boundary). For inflow/outflow/open boundaries the advective integral is legitimately nonzero and
must not be zeroed. BC *type* is ambiguous (a solid SlipWall and an inflow both use `ext_dir`
on the normal velocity), so the domain is classified at runtime from the advecting momentum:

```
closed  <=>  max |normal avg_mom| over the six domain-boundary faces  <  1e-12
```
This is exactly zero at solid walls and `O(rho*U)` for inflow/outflow/open, so the guard
disables the correction automatically when it would be wrong.

---

## 3. Where it is implemented

| Quantity | File / function | What is corrected |
|---|---|---|
| `rho*theta` | `Source/TimeIntegration/ERF_SlowRhsPre.cpp` (`erf_slow_rhs_pre`) | the **tendency** `S_rhs[cons]` (the MRI integrates it afterward) |
| `rho*RhoQ1` (prognostic moisture; **total water `q_t`** for SuperDroplets) | `Source/TimeIntegration/ERF_SlowRhsPost.cpp` (`erf_slow_rhs_post`) | the **updated state** (`S_data[cons]` and `S_new[cons]`), with a positivity clip |

Why two different targets:

- `SlowRhsPre` returns the tendency `S_rhs`, which the MRI integrates later, so correcting the
  tendency is sufficient. The snapshot is taken before the RK-average scales `cell_rhs`, which
  is why `fac` matches the already-applied scaling.
- `SlowRhsPost` integrates the slow scalars **in place** (`cur_cons = S_data[cons]`) and copies
  `new_cons = cur_cons` **inside** the MFIter loop. The post-loop correction must therefore
  shift **both** `S_data[cons]` (carried into the next RK stage) **and** `S_new[cons]` (the
  output state). Correcting only `S_data` is a silent no-op, because `S_new` was already copied.

### Moisture specifics (positivity + the `q_t` invariant)
Moisture variables are **not** independent unbounded tracers, so two extra care points apply:

- **Which component.** For SuperDroplets the conserved-state moisture order is
  `q_t, q_v, q_c, ...` (`MicVar_SD` enum), so **`RhoQ1_comp` is total water `q_t`**, not vapor.
  That is the right thing to conserve (advection + phase change conserve total water); the
  microphysics re-derives `q_v = q_t - q_c`. So we correct **only `RhoQ1_comp`**.
- **Positivity clip.** `q_t` is `>= 0` and has zero-valued cells. An unclipped shift drives
  those cells slightly negative, after which the phase change sets `q_r = q_t < 0` and trips
  `AMREX_ALWAYS_ASSERT(q_r >= 0)` in `ERF_SuperDropletsMoistPhaseChange.cpp` on the first step.
  The correction therefore clips: `q_t -> max(q_t - shift, 0)`. Where the clip fires it adds a
  negligible amount of water (the shift is `~1e-14` early, `~1e-7` at developed flow, and the
  clip only fires in cells that were already ~0), so domain conservation is preserved in
  practice while positivity is guaranteed.

### Coverage
- Corrected: `rho*theta` and `RhoQ1_comp` (total water `q_t` for SuperDroplets).
- **Not** corrected: `q_v`/`q_c`/`q_r` individually (derived from `q_t` by the microphysics),
  `RhoScalar`, `RhoKE`. They have the same leak but are derived/constrained.

### Toggle (input-file knob)
- `erf.conservation_fix_coeff` (real, default `1.0`): scales the correction applied to both
  `rho*theta` and total water. `1.0` = full correction (on), `0.0` = off (reproduces the
  multi-box leak), in-between = partial. Read into `SolverChoice::conservation_fix_coeff`.
- Default is **on** for anelastic + closed domain; the closed-domain guard auto-disables it
  for inflow/outflow/open regardless of the coefficient.

---

## 4. Validation

A gated conservation monitor prints the change in domain-integrated `rho*theta` and
`rho*RhoQ1` (total water) across the dycore: set `ERF_RHOTHETA_PROBE=1`; output line
`[RHOTHETA-PROBE] ... d_dycore=... d_qv=...` (the `d_qv` label is `d(int RhoQ1)` = total
water). For an adiabatic, closed restart both must be ~0.

CloudChamber Case3 restart, 4 ranks (real multi-box), adiabatic
(`erf.use_wall_fluxes=false erf.molec_diff_type=None erf.les_type=None`):

| quantity | `erf.conservation_fix_coeff=0` (off) | `=1` default (on) |
|---|---|---|
| `d_dycore` (rho*theta)        | `-2.8e-3` per step (leaks) | `~8e-13` (machine zero) |
| `d_qv` (rho*RhoQ1 = total water) | `-8.9e-8` per step (leaks) | `~1.6e-17` (machine zero) |

Full Case3 (walls + SuperDroplets), fresh from `t=0`: runs cleanly through development (60+
steps verified, no `q_r >= 0` assertion); the chamber warms toward the correct ~286–288 K
equilibrium instead of the prior ~284.5 K cold bias.

Single grid box (any case): the correction is identically zero (`shift == 0`) — verified
no-op, so single-box answers are unchanged.

> Note: the developed-turbulence moisture validation (`d_qv -8.9e-8 -> 1.6e-17`) was measured
> on a restart *before* the positivity clip was added. The clip only changes behavior in cells
> where `q_t` is already ~0, so it does not alter that result (`q_t > 0` at developed flow); it
> only adds robustness for the fresh `t=0` start (where `q_t` has exact-zero cells).

---

## 5. Caveats / limitations

- **Mitigation, not a cure.** It restores the *domain integrals* of `rho*theta` and total
  water; the local fields still carry the tiny (`~1e-7`) projection divergence error, and the
  remaining advected scalars (`RhoScalar`, `RhoKE`) are not conserved. The proper fix is to
  make the multi-box anelastic projection produce a discretely divergence-free advecting
  momentum (or otherwise make the scalar advection conservative across box boundaries), which
  would fix every advected scalar at once. That is tracked in the upstream bug report.
- **Closed domains only.** Auto-disabled (via the momentum guard) for inflow/outflow/open.
- **Positivity clip on moisture.** `q_t` is clipped to `>= 0` after the shift; where the clip
  fires (cells already ~0) it adds a negligible amount of water, so strict conservation is
  traded for guaranteed positivity. (`rho*theta` is unbounded and needs no clip.)
- **`rho*theta` + total water (`RhoQ1`) only.** `RhoScalar`/`RhoKE` are uncorrected.
- **Anelastic/`fixed_density` only.** Compressible runs evolve `Rho` and are not affected.
