# Super-Droplet boundary-deposition wall-loss

Size-dependent loss of super-droplets to **all chamber boundary surfaces** —
the four lateral side walls, the bottom, and the top — implemented as a
first-order (linear) absorbing sink in boundary-adjacent Eulerian cells.

Files:
- `ERF_SDWallLoss.H` — configuration struct, deposition-velocity model(s),
  stochastic thinning sampler, and coefficient/probability/bin helpers (all
  GPU-callable, documented with equations and references).
- `ERF_SuperDropletPCWallLoss.cpp` — `SuperDropletPC::WallLoss(...)`, applied once
  per step from `SuperDropletsMoist::Advance` (after advection).

## Formulation

For a boundary-adjacent cell of volume `V_cell`, the total first-order loss
coefficient for a droplet of radius `r` sums over every chamber boundary face
`b` the cell touches:

```
k_total(r) = sum_b vd_b(r) * A_b / V_cell                   [1/s]
P_loss(r)  = 1 - exp(-k_total(r) * dt)
```

Face areas: an x-side face contributes `dy*dz`, a y-side face `dx*dz`, and a
top/bottom face `dx*dy`. A cell touching several boundaries (e.g. a corner)
receives all contributions — exactly the sum of independent first-order sinks.

Per-boundary deposition velocities:

```
vd_side(r)   = vd_turb_side(r)
vd_top(r)    = max(vd_turb_top(r)    - v_settle(r), 0)
vd_bottom(r) = vd_turb_bottom(r)     + v_settle(r)
```

`v_settle(r)` is the gravitational terminal settling velocity (the SD's own
terminal velocity `term_vel`); settling enhances floor deposition and opposes
ceiling deposition. The settling correction is optional (off by default).

A face counts as absorbing only if non-periodic with a wall-type BC
(`no_slip_wall`, `slip_wall`, or `surface_layer`). Top/bottom adjacency uses the
domain z index (assumes flat z boundaries).

### Removal methods

- **Option A — deterministic** (default): `xi_new = xi_old * exp(-k_total*dt)`.
- **Option B — stochastic thinning**: draw removed multiplicity from
  `Binomial(xi, P_loss)`, approximated by Poisson (rare-event regime) or a
  clamped Gaussian (large mean); `xi_new = xi_old - removed`.

### Deactivation

If `deactivate_on_depletion` is on, a super-droplet whose multiplicity falls
below `deactivate_threshold` is removed (`id = -1`, `active = 0`) and its
remaining multiplicity is counted as deposited (mass-conserving).

### Turbulent deposition velocity `vd_turb(r)`

- `constant` (default): the per-boundary constants `vd_side/top/bottom`.
- `diffusion` (size-dependent hook): Stokes–Einstein `vd = D(r)/delta`,
  `D(r) = kB*T/(6*pi*mu*r)` (applied to all faces).

References: Crump & Seinfeld (1981); McMurry & Rader (1985); Shima et al. (2009).

## Inputs keys (prefix `super_droplets.`)

| key | default | meaning |
|-----|---------|---------|
| `wall_loss.enable` | `false` | **(1)** master on/off |
| `wall_loss.method` | `deterministic` | **(2)** `deterministic` or `stochastic` |
| `wall_loss.deactivate_on_depletion` | `false` | **(3)** remove depleted SDs |
| `wall_loss.deactivate_threshold` | `1e-30` | multiplicity threshold for removal |
| `wall_loss.deposition_velocity_model` | `constant` | `constant` or `diffusion` |
| `wall_loss.deposition_velocity` | — | sets vd_side/top/bottom at once (default) |
| `wall_loss.vd_side` | `1e-4` | side-wall vd [m/s] |
| `wall_loss.vd_top` | `1e-4` | top vd [m/s] |
| `wall_loss.vd_bottom` | `1e-4` | bottom vd [m/s] |
| `wall_loss.settling_correction` | `false` | apply ∓ v_settle on top/bottom |
| `wall_loss.diffusion_temperature` | `293` | T for `D(r)` [K] |
| `wall_loss.diffusion_viscosity` | `1.8e-5` | gas viscosity [Pa s] |
| `wall_loss.diffusion_delta` | `1e-3` | wall boundary-layer thickness [m] |
| `wall_loss.vd_floor` | `0` | lower bound on vd(r) [m/s] |
| `wall_loss.num_bins` | `16` | diagnostic size bins (log-spaced, ≤ 64) |
| `wall_loss.bin_rmin` / `bin_rmax` | `1e-8` / `1e-4` | bin radius range [m] |
| `wall_loss.diag_interval` | `100` | steps between diagnostic writes (0 disables) |

## Diagnostics output

Written to `super_droplets_wall_loss.txt` (level 0, IO rank) every
`diag_interval` steps. Columns:

1. time (s)
2. dt (s)
3. multiplicity lost to side walls this step (#)
4. multiplicity lost to top boundary this step (#)
5. multiplicity lost to bottom boundary this step (#)
6. cumulative deposited mass on side walls (kg)
7. cumulative deposited mass on top boundary (kg)
8. cumulative deposited mass on bottom boundary (kg)
9… size-resolved deposition rate (#/s), low-r to high-r

Per-boundary attribution splits each droplet's removal by that boundary's share
of `k_total` (`k_b / k_total`).

## Verification: exponential decay in a well-mixed chamber

For a uniform concentration with constant deposition velocities, summing the
per-cell sink over all boundary-adjacent cells gives

```
N(t) = N0 * exp(-k_eff * t)
k_eff = (vd_side*A_side + vd_top*A_top + vd_bottom*A_bottom) / V_chamber
```

For a box `Lx × Ly × Lz`:

```
A_side   = 2*Lz*(Lx + Ly)      (four side walls)
A_top    = Lx*Ly
A_bottom = Lx*Ly
V_chamber = Lx*Ly*Lz
```

For zero settling and equal velocities on all surfaces this reduces to

```
k_eff = vd * A_total / V_chamber,   A_total = A_side + A_top + A_bottom
```

Procedure:
1. Uniform super-droplet field in a closed (walled) chamber;
   `method = deterministic`, `deposition_velocity_model = constant`,
   `settling_correction = false`; disable phase change / coalescence / injection
   so only deposition acts on the total number.
2. Track total physical particle number `N(t)`
   (`SuperDropletPC::TotalNumberOfParticles()` or the domain time series).
3. Fit `ln N(t)` vs `t`; the slope should match `-k_eff` above.
