# Super Droplet Method: GPU Scaling Analysis

Analysis of GPU vs CPU execution in the ERF Super Droplet Method (SDM) implementation, and why runtime still scales with particle count on GPU.

## Operation-Level Summary

| Operation | GPU? | Sync Points | MPI Comm | Main Cost Driver |
|-----------|------|-------------|----------|-----------------|
| Phase Change | 100% GPU | 1 | None | Per-particle ODE solver (Newton-Raphson) |
| Advection | 100% GPU | 1 | Redistribute() | Velocity interpolation + position update |
| Coalescence | ~95% GPU | 7 | ReduceRealSum | Binning, shuffling, pairing, RNG kernel |
| Injection | 100% GPU | Multiple | None | ParallelForRNG for positioning |
| Recycling | 100% GPU | Multiple | Redistribute() | RNG repositioning |
| Diagnostics | 100% GPU | Multiple | Global reduce | ReduceOps over particles |
| Boundaries | 100% GPU | 1 | None | Per-particle boundary checks |

## Fully GPU-Accelerated Operations

### Phase Change (`ERF_SuperDropletPCMassChange.cpp`)
- `ParallelFor()` over particles with AMREX_GPU_DEVICE lambda
- Per-particle Newton-Raphson solver with configurable schemes (RK4, RK3BS, BE, CN, DIRK2)
- CIC interpolation to particle locations on GPU
- Single `Gpu::synchronize()` at end
- Scales well

### Advection (`ERF_SuperDropletPCAdvection.cpp`)
- `ParallelFor()` over particles
- MAC grid velocity interpolation + terminal velocity on GPU
- Single `Gpu::synchronize()` after all particles
- Followed by `Redistribute()` (MPI collective, cost grows with particle count/movement)

### Boundaries (`ERF_SuperDropletPCBoundaries.cpp`)
- `ParallelFor()` over particles
- Deactivation logic fully on GPU
- Single sync at end

## Main Bottleneck: Coalescence

**File:** `ERF_SuperDropletPCCoalescence.cpp`

Coalescence is the most expensive operation due to multiple host/device transitions and sync points:

### Step-by-step execution

1. **Host-side tile loop** (serial, NOT OpenMP-safe)
   - Code comment: `"Do NOT add OpenMP here; building DenseBins is not thread-safe"`
   - Iterates over tiles sequentially on host

2. **DenseBins construction** (HOST)
   - Builds spatial bins for particle pairing
   - Not GPU-accelerated
   - Cost grows with particle count

3. **Shuffling** (HYBRID)
   - GPU: `ParallelFor()` to get max bin size → `Gpu::synchronize()`
   - HOST: `std::shuffle` on stencil vector (random ordering)
   - Copy stencil to device → `Gpu::synchronize()`
   - GPU: `ParallelFor()` to shuffle particles within bins → `Gpu::synchronize()`

4. **Pairing** (GPU)
   - `ParallelFor()` over bins, pairs particles from opposite ends of shuffled list
   - `Gpu::synchronize()`

5. **Coalescence kernel** (GPU)
   - `ParallelForRNG()` over particles with GPU random number generation
   - Collision kernel + Brownian coalescence computed per pair
   - Atomic accumulation: `Gpu::Atomic::Add()` for collision count
   - `Gpu::synchronize()`

6. **Attribute updates** (GPU)
   - `ParallelFor()` to update mass, radius, multiplicity for successful events
   - `Gpu::synchronize()`

7. **MPI reductions** (HOST)
   - `ParallelDescriptor::ReduceRealSum()` for collision statistics
   - `ParallelDescriptor::ReduceRealMax()` for timing

### Why coalescence is the bottleneck
- **7 `Gpu::synchronize()` calls** — each one stalls the GPU pipeline
- **Host-side DenseBins** — serial construction, not GPU-accelerated
- **Host-side std::shuffle** — random permutation on CPU, then copied to device
- **Serial tile loop** — cannot use OpenMP due to DenseBins thread-safety

## Why Runtime Scales with Particle Count

GPU parallelism reduces the constant factor but doesn't change algorithmic complexity:

1. **All operations are O(N)** — doubling particles roughly doubles GPU work
2. **Coalescence sync overhead grows** — more particles = more work between each of the 7 sync points = longer stalls
3. **DenseBins construction** — serial host-side cost grows with N
4. **Redistribute()** — MPI communication volume grows with particle count and movement
5. **Per-tile GPU launches** — more particles per tile = larger kernels, but tile iteration is serial on host

## Potential Improvements (for future discussion)

- GPU-accelerate DenseBins construction
- Reduce coalescence sync points (combine operations)
- Batch tile operations for coalescence to reduce host-side serialization
- Profile Redistribute() overhead
- Consider fusing shuffle + pairing into fewer kernel launches

## GPU Performance Validation Workflow

Use a clean build with the particle and CUDA options enabled, for example:

```bash
cmake -S . -B build-sdm-gpu \
  -DERF_ENABLE_MPI=ON -DERF_ENABLE_CUDA=ON -DERF_ENABLE_PARTICLES=ON \
  -DERF_ENABLE_TESTS=ON
cmake --build build-sdm-gpu -j
```

Before collecting timings, enumerate the SDM tests and select the relevant
cloud-chamber or scaling cases:

```bash
ctest --test-dir build-sdm-gpu -N
ctest --test-dir build-sdm-gpu -R SDM --output-on-failure
```

For repeatable comparisons, set `erf.fix_random_seed = 1` in the input file
(the SDM initializer then uses its fixed seed of 1024). Keep the mesh,
particle initialization, timestep, and MPI rank count identical between runs.
Record total wall time and the SDM diagnostics for total particle count,
water/species/aerosol mass, cloud and rain mass, collision count, phase-change
non-convergence count, and the first successful injection timestep.

Run a full-physics baseline and isolating variants with the same input:

1. full SDM;
2. coalescence disabled;
3. phase change disabled;
4. particle advection disabled.

Use TinyProfiler (or the configured AMReX profiling output) for the same
regions in each run: `SuperDropletPC::Coalescence`,
`SuperDropletPC::MassChange`, `SuperDropletPC::AdvectParticles`,
`ERFPCParticleToMesh`, `SuperDropletPC::Diagnostics`, and particle
redistribution. GPU kernel timings may require synchronization around the
profiled region; document any profiler setting that changes synchronization.
Do not enable profiler-induced synchronization in production measurements
unless it is explicitly stated. Compare diagnostics and conservation first,
then compare wall time and region timings. Do not claim a speedup until the
runs are on the same backend and configuration; report neutral or slower
results honestly.
