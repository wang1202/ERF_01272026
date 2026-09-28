# ERF superdroplet collision efficiency (XYLI)

## Purpose and status

This note describes the opt-in collision kernels added to ERF's superdroplet model from Dhanasekaran, Roy, and Koch (2021), *Collision rate of bidisperse, hydrodynamically interacting spheres settling in a turbulent flow*, Journal of Fluid Mechanics 912, A5, [doi:10.1017/jfm.2020.1113](https://doi.org/10.1017/jfm.2020.1113).

ERF now offers two paper-based selections:

- `paper_ideal`: the paper's ideal (no hydrodynamic-interaction efficiency) pair kernel.
- `statistical_efficiency`: the same ideal kernel multiplied by the paper's fitted collision efficiency.

The existing `sedimentation` default and the `golovin`, `Longs`, and `Halls` selections remain available and unchanged. Brownian collisions remain an independently enabled additive contribution.

## Pairwise ideal collision kernel

The paper fits its ideal volumetric collision rate as

\[
\frac{K^0_{12}}{n_1n_2\,\Gamma_\eta(a_1+a_2)^3}
= \frac{1.55\,Re_\lambda^{-0.09}}{1+2.1Q} + \frac{\pi}{2}Q.
\]

ERF uses the corresponding **pairwise kernel** in m³/s:

\[
K_{\mathrm{ideal}}(a_1,a_2)
= \Gamma_\eta(a_1+a_2)^3
  \left[\frac{1.55\,Re_\lambda^{-0.09}}{1+2.1Q}+\frac{\pi}{2}Q\right].
\]

Number densities and superdroplet multiplicities do not belong inside this kernel. ERF applies its usual bin-volume and multiplicity factors in the stochastic collision algorithm after the kernel is computed.

The paper definitions used are

\[
\Gamma_\eta=\sqrt{\epsilon/\nu},\qquad
 a_*=(a_1+a_2)/2,\qquad
 Q=\frac{V_{\mathrm{settle}}}{\Gamma_\eta a_*},
\]

with

\[
V_{\mathrm{settle}}=
\frac{2\rho_p g |a_1^2-a_2^2|}{9\mu},\qquad
Kn=\frac{\lambda_g}{a_*},\qquad
\kappa=\frac{\min(a_1,a_2)}{\max(a_1,a_2)}.
\]

The absolute value makes Q a nonnegative speed ratio. The paper's printed numerator and its stated `κ <= 1` radius ordering have opposite signs; the implementation uses the physical settling-speed magnitude, consistent with the paper's nonnegative Q plots. In the large-Q limit, the ideal kernel tends to the Smoluchowski geometric-cross-section rate `π(a1+a2)² V_settle`.

## Environmental properties used by ERF

The coalescence method receives pressure and temperature for each particle's cell. The paper kernels calculate:

- Dynamic air viscosity `μ(T)` using the same piecewise temperature fit already used by ERF's Brownian kernel.
- Gas mean free path `λ_g(p,T,μ)` using the Brownian kernel's existing expression.
- Dry-air density `ρ_air = p/(R_d T)` and kinematic viscosity `ν = μ/ρ_air`.
- Particle density `ρ_p` from the H2O material density and gravity from ERF's `CONST_GRAV`.

This is a prescribed-parameter turbulence model. ERF does not infer local dissipation or Taylor-scale Reynolds number from LES/DNS fields in this implementation. Moist-air density corrections are not included in the density estimate above.

## Statistical-efficiency kernel

The statistical-efficiency mode applies

\[
K_{\mathrm{statistical-efficiency}}=\beta K_{\mathrm{ideal}},
\]

where `β = K_actual/K_ideal` and the paper's Eq. 5.7 is

\[
\beta=\frac{p_1(Q)Kn^{q_1}}
{[p_2(Q)+\ln(1/Kn)]^{q_2}}.
\]

The exponents use the mobility coefficients in Table 2:

\[
q_1=\frac{3B_0(3\cos^2\theta'-1)+M_0Q\cos\theta'}
{2A_1(3\cos^2\theta'-1)+L_1Q\cos\theta'},\quad
q_2=\frac{3B_1(3\cos^2\theta'-1)+M_1Q\cos\theta'}
{2A_1(3\cos^2\theta'-1)+L_1Q\cos\theta'}.
\]

ERF uses `θ' = 53°`, `l1 = 1`, and

\[
l_2=\frac{1}{\kappa Kn\ln(1/Kn)},\qquad
p_i(Q)=e^{-l_iQ}p_{i,\epsilon}+(1-e^{-l_iQ})p_{i,g}.
\]

The Table 2 mobility coefficients and Table 3 fit values are encoded at their published `κ` nodes from 0.3 through 0.99. The paper says it polynomial-fitted values between nodes but does not publish the polynomial coefficients. ERF therefore uses piecewise-linear interpolation, which exactly reproduces all table nodes. For equal-size pairs (`κ > 0.99` through 1), it holds the 0.99 endpoint instead of extrapolating.

The `statistical_efficiency` selection requires `Re_lambda = 2500`, the Reynolds number at which the efficiency fit was made. It accepts pairs only inside the stated fit limits:

- `0.3 <= κ <= 1` (with endpoint hold above 0.99)
- `1e-4 <= Kn <= 0.2`
- `0 <= Q <= 1e5`
- Both radii at least 15 μm

Invalid statistical-efficiency-fit inputs cause the collision kernel to abort with an error. The lower-radius guard reflects the paper's statement that van der Waals attraction matters for smaller droplets and is omitted from Eq. 5.7. The first implementation is intended for warm-liquid, inertialess droplets; it does not include ice physics, particle inertia, inertial clustering, or a dynamic LES/DNS turbulence closure.

## Runtime inputs and examples

All parameters use ERF's superdroplet input prefix, normally `super_droplets_moisture`.

| Input | Default | Meaning |
|---|---:|---|
| `coalescence_kernel` | `sedimentation` | Choose `paper_ideal` or `statistical_efficiency` to enable these kernels. |
| `paper_epsilon` | `1.0e-2` | Prescribed dissipation in m²/s³. The paper discusses `1e-3` and `1e-1` as low/high sensitivity cases. |
| `paper_re_lambda` | `2500` | Prescribed Taylor-scale Reynolds number. `statistical_efficiency` requires 2500. |
| `include_brownian_coalescence` | `false` | Brownian kernel is added separately when enabled. Keep it false for direct paper-rate comparisons. |

Example ideal-rate run for the paper's `Re_lambda=90` curve:

```ini
super_droplets_moisture.coalescence_kernel = "paper_ideal"
super_droplets_moisture.paper_epsilon = 1.0e-2
super_droplets_moisture.paper_re_lambda = 90
super_droplets_moisture.include_brownian_coalescence = false
```

For an ideal-versus-statistical-efficiency comparison, use identical initial particles, environment, bins, multiplicities, time step, and random seed in both runs. Set `paper_re_lambda = 2500`, keep Brownian collisions disabled, and change only `coalescence_kernel` from `paper_ideal` to `statistical_efficiency`. Vary the radius pairs at fixed pressure and temperature to span the Q range in Fig. 2. To test the paper's dissipation sensitivity, repeat with `paper_epsilon = 1e-3` and `1e-1`; keep the other benchmark settings fixed and record the chosen environmental state.

The ideal-rate comparison at `Re_lambda=90` and `2500` is made against the paper's Eq. 3.2 after normalizing the pair kernel by `Gamma_eta*(a1+a2)^3`. Do not interpret a single stochastic SDM realization as a deterministic reproduction of the paper's rate curve. Aggregate collision statistics require controlled ensembles or a separate kernel-level oracle.

## Implementation locations

- `Source/Particles/ERF_SuperDropletPCCoalescence.H`: pair-property calculation, ideal kernel, coefficient tables/interpolation, and efficiency fit.
- `Source/Particles/ERF_SuperDropletPCCoalescence.cpp`: pair-loop dispatch and the shared collision-probability integration point.
- `Source/Particles/ERF_SuperDropletPCDefinitions.H`: kernel enumeration.
- `Source/Particles/ERF_SuperDropletPCInitializations.cpp`: input parsing, validation, and startup reporting.
- `Source/Particles/ERF_SuperDropletPC.H`: stored prescribed turbulence parameters.
- `Docs/sphinx_doc/theory/Microphysics.rst`: SDM model and input documentation.
- [make_comparison.py](Exec/RegTests/RayleighBenardSDM/post-processing/make_comparison.py): generates the number-based statistical-efficiency versus Hall figure from run logs and binned DSD data. The committed PNG is saved beside the script.

Run the plot script with the two simulation-output directories:

```sh
python3 Exec/RegTests/RayleighBenardSDM/post-processing/make_comparison.py \
  --statistical-efficiency-run /path/to/statistical_efficiency_run \
  --hall-run /path/to/hall_run
```

Each run directory needs `run.log` and the binned DSD text files for steps 00500, 01000, and 01500. The script requires Python 3, NumPy, and Pillow. It writes the PNG and derived CSV files into `post-processing` unless `--output-dir` is supplied.

## Verification status

A host-side C++ harness compiled the kernel header with warnings treated as errors and passed checks for all Table 2–3 values at the listed nodes, the Eq. 3.2 cases at `Re_lambda=90` and 2500, the Q=0 efficiency value, pair-order symmetry, epsilon sensitivity inputs, fit-domain rejection, the 15 μm radius guard, and the large-Q settling limit.

The particle-enabled ERF application built successfully with the repository Make setup. A 10-step, 3D Rayleigh-Benard smoke run using `coalescence_kernel = statistical_efficiency` completed to 0.01 s and confirmed the selected kernel in startup output. The 1.5 s plate-deposition comparison is documented in the case README. The equation checks and smoke run validate code execution but do not establish physical validation of the model or its interpolation choice.
