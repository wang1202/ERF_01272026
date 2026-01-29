# Wall Temperature & Wetness Patterns: Developer Guide

How to use and extend the sidewall temperature/wetness pattern system for wall flux boundary conditions.

## Overview

The wall flux system supports spatial patterns for `T_wall` and `wetness` on the four side walls (xlo, xhi, ylo, yhi). The pattern does **not** affect:
- Momentum coefficients (Cm, Ch, Cq) -- these remain per-face
- Top/bottom faces (zlo, zhi) -- these always use their per-face values

## Files Involved

| File | Role |
|------|------|
| `Source/BoundaryConditions/ERF_WallFluxesAllFaces.H` | `WallFluxParams` struct (pattern data fields) |
| `Source/DataStructs/ERF_DataStruct.H` | `init_wallflux_params_level()` (input parsing) |
| `Source/BoundaryConditions/ERF_WallFluxesAllFaces.cpp` | GPU kernels (4 side walls use pattern data) |

## Existing Patterns

### `uniform` (default)

Each face uses its own `T_wall` and `wetness` from per-face parameters. No special fields needed.

```
wallflux.T_wall.xlo = 290.0
wallflux.T_wall.xhi = 290.0
wallflux.wetness.xlo = 1.0
wallflux.wetness.xhi = 1.0
# ... etc for ylo, yhi
```

### `two_layer`

Side walls split into upper/lower regions at a separation height. All four side walls share the same upper/lower values.

```
wallflux.sidewall_pattern = two_layer
wallflux.sidewall_separation_height = 4.5   # [m]
wallflux.T_wall_upper = 280.0               # [K], z >= 4.5m
wallflux.T_wall_lower = 299.0               # [K], z < 4.5m
wallflux.wetness_upper = 1.0                # z >= 4.5m
wallflux.wetness_lower = 1.0                # z < 4.5m
```

Per-face `T_wall.{xlo,...}` and `wetness.{xlo,...}` are ignored for side walls when `two_layer` is active.

## How to Add a New Pattern

### Step 1: Add data fields to `WallFluxParams` (ERF_WallFluxesAllFaces.H)

Add your pattern's parameters to the struct. Use only POD types (bool, Real, int) since the struct is captured by GPU lambdas.

```cpp
struct WallFluxParams {
    bool enable{false};
    WallFaceParams face[6];

    // Two-layer sidewall pattern
    bool two_layer_sidewall{false};
    amrex::Real sidewall_separation_height{0.0};
    amrex::Real T_wall_upper{293.0};
    amrex::Real wetness_upper{1.0};
    amrex::Real T_wall_lower{293.0};
    amrex::Real wetness_lower{1.0};

    // YOUR NEW PATTERN: add fields here
    // bool my_pattern{false};
    // amrex::Real my_param{0.0};
};
```

**Important**: No `std::string`, `std::vector`, or other non-trivially-copyable types -- these cannot be captured by CUDA device lambdas.

### Step 2: Parse input parameters (ERF_DataStruct.H)

In `init_wallflux_params_level()`, add an `else if` branch for your pattern name:

```cpp
std::string sidewall_pattern = "uniform";
pp_wf.query("sidewall_pattern", sidewall_pattern);

if (sidewall_pattern == "two_layer") {
    wallflux_level.two_layer_sidewall = true;
    pp_wf.get("sidewall_separation_height", wallflux_level.sidewall_separation_height);
    pp_wf.get("T_wall_upper", wallflux_level.T_wall_upper);
    pp_wf.get("T_wall_lower", wallflux_level.T_wall_lower);
    pp_wf.get("wetness_upper", wallflux_level.wetness_upper);
    pp_wf.get("wetness_lower", wallflux_level.wetness_lower);
} else if (sidewall_pattern == "my_pattern") {
    // wallflux_level.my_pattern = true;
    // pp_wf.get("my_param", wallflux_level.my_param);
} else if (sidewall_pattern != "uniform") {
    amrex::Abort("Invalid wallflux.sidewall_pattern: " + sidewall_pattern +
                 ". Valid options: uniform, two_layer, my_pattern");
}
```

Use `pp_wf.get()` (not `query`) for required parameters so it aborts with a clear message if missing.

### Step 3: Modify GPU kernels (ERF_WallFluxesAllFaces.cpp)

Each side wall face (xlo, xhi, ylo, yhi) has a `ParallelFor` lambda. Inside each, there is a block that computes `T_wall_eff` and `wetness_eff`. Add your pattern logic there:

```cpp
// Determine effective T_wall and wetness
Real T_wall_eff = face_params.T_wall;
Real wetness_eff = face_params.wetness;
if (params.two_layer_sidewall) {
    Real z_cell = prob_lo_z + (kk + Real(0.5)) * dz;
    if (z_cell >= params.sidewall_separation_height) {
        T_wall_eff = params.T_wall_upper;
        wetness_eff = params.wetness_upper;
    } else {
        T_wall_eff = params.T_wall_lower;
        wetness_eff = params.wetness_lower;
    }
} else if (params.my_pattern) {
    // Your logic here using kk, ii, jj, dz, prob_lo_z, etc.
    // T_wall_eff = ...;
    // wetness_eff = ...;
}
```

**You must update all 4 side wall kernels** (xlo, xhi, ylo, yhi). They are located at approximately:

| Face | Search string to find the kernel |
|------|----------------------------------|
| xlo  | `// X-lo face (face = 0)` |
| xhi  | `// X-hi face (face = 1)` |
| ylo  | `// Y-lo face (face = 2)` |
| yhi  | `// Y-hi face (face = 3)` |

### Available variables inside the GPU kernels

| Variable | Description |
|----------|-------------|
| `kk` | Cell-centered z-index of the interior cell |
| `ii`, `jj` | Cell-centered x,y indices |
| `prob_lo_z` | Physical z-coordinate of domain origin [m] |
| `dz` | Cell size in z [m] |
| `dx`, `dy` | Cell sizes in x, y [m] |
| `face_params` | Per-face parameters (Cm, Ch, Cq, T_wall, wetness) |
| `params` | Full `WallFluxParams` (your new fields are here) |

Cell-center z-coordinate: `prob_lo_z + (kk + 0.5) * dz`

### What NOT to modify

- **Z-face kernels** (zlo, zhi): These should always use `face_params.T_wall` / `face_params.wetness` directly.
- **Momentum fluxes**: The pattern only affects `T_wall_eff` and `wetness_eff`, not `Cm`, `Ch`, `Cq`.

## Testing a New Pattern

1. Build with the CloudChamber case (or any case using `wallflux.enable = true`)
2. Verify `wallflux.sidewall_pattern = uniform` still works identically to before
3. Test your new pattern and check output for expected temperature/moisture flux distributions
4. Verify zlo/zhi faces are unaffected
