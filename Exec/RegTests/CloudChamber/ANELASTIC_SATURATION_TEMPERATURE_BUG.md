# Anelastic saturation-temperature bug (SuperDroplets / CloudChamber)

**Status:** root cause found and fixed. Two distinct bugs of the same class were fixed and
verified:
1. **Streak-vs-ambient RH sign** (spatial): SDM saturation used the θ-coupled EOS temperature —
   fixed in `Copy_State_to_Micro`, phase-change, and the plotfile (see below).
2. **Bulk mean T and RH** (domain average): the wall-flux `θ_wall`/`q_wall` used the θ-coupled EOS,
   and `prob.rho_0` was inconsistent with the dry-EOS base state — both fixed (see
   "Bulk mean temperature and RH" below). Verified: bulk T 285.5 K → 287.7 K (→288) and bulk RH
   1.188 → 1.025 (DNS target ≈1.03).

---

## Symptom

In the CloudChamber (anelastic + SuperDroplets) case, a 2-D `rel_humidity` cross-section at the
**first z-level** shows the convective **streaks** — cells with high temperature, high `qv`, and
positive `w` (the warm/moist updraft plumes coming off the warm wet bottom wall) — reading
**lower** relative humidity than the surrounding ambient/downdraft air.

This is **backwards**. The rising, warm, moist plume should be the *most* saturated. A reference LES
(PINACLES) at the *same* resolution (32³), advection (WENO5), and SGS (Smagorinsky), and DNS, both
show the streaks with **higher** RH than ambient.

Quantitatively (k=0 plane, 1024 cells, conditioned on `w`):

| | STREAKS (w>0) | AMBIENT (w<0) |
|---|---|---|
| before fix | RH = 1.0196 | RH = 1.0457  ✗ (streaks lower) |
| after fix  | RH = 1.1714 | RH = 1.1623  ✓ (streaks higher) |

## Root cause

The model is **anelastic**: density `ρ` is pinned to the hydrostatic base `ρ₀(z)`, and the
thermodynamic pressure is the base hydrostatic pressure `p_hse(z)`. The physically-correct
temperature is therefore

```
T = θ · π₀(z)          (π₀ = base Exner = (p_hse/p_ref)^(R_d/c_p))
p = p_hse(z)
```

But the SuperDroplets model formed the temperature/pressure used for **saturation** from the
θ-coupled **compressible** EOS instead:

```
T_SDM = getTgivenRandRTh(ρ, ρθ, qv)
p_SDM = getPgivenRTh(ρθ, qv)
```

With `ρ` fixed to `ρ₀`, `getTgivenRandRTh` reduces to
`T_SDM = θ · [ (θ/θ₀)·(1+1.61 qv)/(1+1.61 qv₀) ]^0.4`, which **over-responds to θ** (amplifies the
θ′ perturbation by ~1.4× and adds a `+qv′` term). Result: `T_SDM` is **inflated in warm/moist
cells** (~0.66 K in the streaks, ~0 in ambient) → `qsat(T_SDM)` inflated → `RH = qv/qsat` depressed
**precisely in the updraft streaks**, flipping the sign of the supersaturation–`w` correlation.

PINACLES avoids this because it uses static energy `s` → `T = (s − g z)/c_p` (the physical
temperature) at the reference pressure for `qsat`.

**Verification that this is the cause:** recomputing RH from the existing fields with the physical
temperature (`θ` instead of `T_SDM`) flips the streaks from `1.02 < ambient` to `1.055 > ambient`.

### What it is *not* (ruled out by direct test)
- Resolution (64³ gives the same wrong sign).
- Wall-flux closure (matching PINACLES's constant-K wall flux did not fix the sign).
- Advection scheme (WENOZ5 / WENOZ7 / centered all identical near-wall).
- Condensation (qc = qr = 0; no droplets until injection `t_start`).
- SGS / molecular diffusivity asymmetry (θ and qv transport are component-symmetric;
  `Pr_t ≈ Sc_t`, and `α_C > α_T` actually favors moisture).
- Buoyancy moisture coupling (standard virtual-θ, vapor enters correctly).

## Severity
- **Diagnostic** (no droplets): the plotted `rel_humidity` field is wrong (warm streaks look dry).
- **Physics** (with droplets): the SDM phase-change `qsat` uses the same inflated `T`, so the
  supersaturation that drives **droplet activation/growth** is suppressed exactly in the updrafts
  where droplets should form. **This corrupts the cloud microphysics, not just the plot.**

---

## The fix — files to update

Physically correct value, gated on anelastic (compressible path unchanged):
`T = θ · π₀`, `p = p_hse`, read from the **base state** (`BaseState::pi0_comp`, `BaseState::p0_comp`).
The base state is time-independent; a pointer is plumbed into the SDM and refreshed each step
(regrid-safe). Because the base state has only 3 ghost cells vs the SDM vars' ~5, read it with
indices clamped into the base fab (it is horizontally uniform and vertically smooth, so clamping
ghosts is exact).

### Plumbing (deliver `base_state` to the SDM)
1. `Source/Microphysics/Null/ERF_NullMoist.H`
   — add `virtual void Set_Base_State(int lev, const amrex::MultiFab&) {}`.
2. `Source/Microphysics/ERF_Microphysics.H` (wrapper interface)
   — add `virtual void Set_Base_State(const int&, const amrex::MultiFab&) { }` (default no-op,
     so Eulerian needs no change).
3. `Source/Microphysics/ERF_LagrangianMicrophysics.H`
   — override `Set_Base_State` to forward to `m_moist_model->Set_Base_State(lev, base_state)`.
4. `Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoist.H`
   — override `Set_Base_State` (store per-level pointer); add members
     `bool m_anelastic` and `amrex::Vector<const amrex::MultiFab*> m_base_state`.
5. `Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoistInit.cpp` (`Define`)
   — set `m_anelastic = (a_sc.anelastic[0] == 1);`.
6. `Source/TimeIntegration/ERF_AdvanceMicrophysics.cpp`
   — before `Update_Micro_Vars_Lev`, call `micro->Set_Base_State(lev, base_state[lev]);`.

### The two saturation-T/p recompute sites (the actual fix)
Both must be changed; the **phase-change one runs last and feeds the plotted `rel_humidity`
(`qmoist[5] = MicVar_SD::rh`) and the droplet-activation `qsat`** — fixing only the first does
nothing.

7. `Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoistUtils.cpp`
   — `Copy_State_to_Micro`, the `t_arr/p_arr` ParallelFor.
8. `Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoistPhaseChange.cpp`
   — the `t_arr/p_arr` recompute (~line 174).

Both use this pattern:
```cpp
const bool l_anelastic = (m_anelastic && lev < (int)m_base_state.size() && m_base_state[lev]);
const Array4<Real const> base_arr = l_anelastic ? m_base_state[lev]->const_array(mfi)
                                                : Array4<Real const>{};
const Dim3 blo = l_anelastic ? amrex::lbound(base_arr) : Dim3{0,0,0};
const Dim3 bhi = l_anelastic ? amrex::ubound(base_arr) : Dim3{0,0,0};
ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
    Real theta = ρθ / ρ;
    if (l_anelastic) {
        int bi = min(max(i, blo.x), bhi.x);   // clamp ghosts into the 3-ghost base fab
        int bj = min(max(j, blo.y), bhi.y);
        int bk = min(max(k, blo.z), bhi.z);
        t_arr(i,j,k,0) = theta * base_arr(bi,bj,bk, BaseState::pi0_comp);
        p_arr(i,j,k,0) =         base_arr(bi,bj,bk, BaseState::p0_comp);
    } else {
        t_arr(i,j,k,0) = getTgivenRandRTh(ρ, ρθ, qv);   // compressible: unchanged
        p_arr(i,j,k,0) = getPgivenRTh(ρθ, qv);
    }
});
```

### Plotfile diagnostic consistency
9. `Source/IO/ERF_Plotfile.cpp` — the `"qsat"` block: use `T = θ·π₀` (alias
   `pi_hse(base_state[lev], make_alias, BaseState::pi0_comp, 1)`) for anelastic, keep `p = p_hse`.

---

## Bulk mean temperature and RH (the second bug)

### Symptom
Separate from the streak *sign*, the **domain-averaged** state sat in the wrong place: bulk
**T ≈ 285.5 K** (should be ~288 K) and bulk **RH ≈ 1.18** (should be ~1.03 — DNS has mean T = 288 K,
mean qv ≈ 0.011 kg/kg → RH ≈ 1.03). The cold bias and the high RH are the *same* error: too-cold
air has a lower `qsat`, so `RH = qv/qsat` is inflated.

### Two compounding causes, both the anelastic-pressure mistake

**(a) Wall-flux `θ_wall` cold bias.** The wall heat flux converts the user's wall temperature
`T_wall` to a potential temperature with the **θ-coupled EOS**:
```cpp
Real p_pa  = getPgivenRTh(rhotheta, qv);     // ← inflated by the moisture factor
Real exner = getExnergivenP(p_pa, R_d/Cp_d);
Real theta_wall = T_wall / exner;
```
In anelastic, `ρ` is pinned to `ρ₀`, so `getPgivenRTh(ρθ, qv)` is inflated mainly by the moisture
factor `(1 + 1.61 qv)^1.4 ≈ 1.025` at `qv ≈ 0.011`. That gives `exner ≈ 1.007`, so
`θ_wall = T_wall/1.007` is **~2 K too low on every wall**. The walls drive the whole enclosed
domain, so the bulk θ is pulled ~2 K cold → bulk T ≈ 286 instead of 288 → `qsat` too low → bulk RH
too high. The same `getPgivenRTh` pressure fed `compute_qsat_wall`, so `q_wall` was off too.

**(b) `prob.rho_0` inconsistent with the dry-EOS base state.** The base-state HSE init
(`erf_init_const_dens_and_th_hse` in `Source/Prob/ERF_InitDensityHSE.H`) computes the base pressure
with the **dry** EOS: `p_hse = getPgivenRTh(rho_0 · T_0)`. With the old `rho_0 = 1.189416` (a value
chosen to be *moist*-consistent with `p=1e5, T=288, qv=qsat`) this gives `p_hse ≈ 97 645 Pa` and
`π₀ ≈ 0.993`, **not** `1e5` / `1`. Then everywhere the corrected code uses `T = θ·π₀`, the
temperature is depressed by 0.7 % (~2 K at 288 K) and `qsat(T, p_hse)` shifts — again inflating RH.
The dry-consistent value is `rho_0 = p_0/(R_d·T_0) = 1e5/(287·288) = 1.20984`, which gives
`p_hse = 1e5`, `π₀ = 1`, so `T = θ` exactly.

### The fix
**(a)** Plumb the base state into the wall-flux routine and use `p_hse`/`π₀` instead of the EOS
(anelastic-gated; compressible path unchanged):
- `Source/BoundaryConditions/ERF_WallFluxesAllFaces.{H,cpp}` — `impose_WallFluxes_all_faces` gains
  `const MultiFab* base_state` + `bool l_anelastic`. All six face kernels now read
  `p_pa = base_arr(..,BaseState::p0_comp)` and `exner = base_arr(..,BaseState::pi0_comp)` (indices
  clamped into the 3-ghost base fab via `lbound`/`ubound`), so `θ_wall = T_wall/π₀` and
  `q_wall = qsat(T_wall, p_hse)`.
- `Source/TimeIntegration/ERF_SlowRhsPre.cpp`, `ERF_TI_slow_headers.H`, `ERF_TI_slow_rhs_pre.H` —
  thread `base_state[level]` through `erf_slow_rhs_pre` to the call site.

**(b)** Set `prob.rho_0 = 1.20984` in `Exec/RegTests/CloudChamber/inputs`.

### Verification (PiChamber: 2×2×1 m, 64×64×32, 16 000 steps, equilibrated t ≥ 200 s)
| | bulk T | bulk RH |
|---|---|---|
| before (`rho_0=1.189416`, EOS wall flux) | 285.50 K | 1.188 |
| after  (`rho_0=1.20984`, base-state wall flux) | 287.67 K | 1.025 |
| DNS target | 288 K | ~1.03 |

(Read off the SDM `*_domain_timeseries.txt` columns 2/3. Note that file appends across reruns and
is only truncated at step 0 — delete it between fresh, non-restart runs or the averages mix runs.)

---

## Remaining work (same bug class, lower priority)

- **Plotfile `temp` field** — routes through the generic `erf_dermoisttemp` derive (no base-state
  access), so it still shows the EOS temperature. Either make the derive anelastic-aware or compute
  `temp` inline like `qsat`. Secondary (you plot `rel_humidity`, which is fixed).
- ~~**`compute_qsat_wall`** / wall-flux `θ_wall`~~ — **DONE** (see "Bulk mean temperature and RH").
  This turned out *not* to be small: it was the dominant driver of the bulk cold bias / high RH.
- **`ERFPCEvolve.cpp:326`** (`tabs = getTgivenRandRTh(...)`) — this is the **generic tracer**
  container `ERFPC`, **not** `SuperDropletPC`. Irrelevant to SuperDroplets; only matters if you use
  temperature-based tracer particles.

## Notes / gotchas
- **Reconciled:** the inputs now set `prob.rho_0 = 1.20984` so the dry-EOS base state gives
  `p_hse = 1e5`, `π₀ = 1` (`T = θ`). The old `rho_0 = 1.189416` gave `p_hse ≈ 97,645 Pa`, `π₀ ≈
  0.993`, which is one of the two bulk-RH causes above. (`rho_0 = 1.189416` was moist-consistent
  for `p=1e5,T=288,qv=qsat`, but the base-state HSE init uses the **dry** EOS, so the dry-consistent
  `rho_0 = p_0/(R_d T_0)` is the right value.)
- Quick check that the fix is active: the plotted `rel_humidity` (`= MicVar_SD::rh = qmoist[5]`)
  should change; the plotfile `temp` field will *not* (it's the separate EOS derive).
- Robust verification metric: conditional mean of `rel_humidity` over the full k=0 plane split by
  `sign(w)` — single fextract lines are too noisy to trust.
