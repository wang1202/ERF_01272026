#include <cmath>

#include <gtest/gtest.h>

#include <AMReX_REAL.H>

// The wall-loss module is only available when particles are enabled.
#ifdef ERF_USE_PARTICLES

#include "ERF_SDWallLoss.H"
#include "ERF_Constants.H"

using amrex::Real;

namespace {

// Replicate the deterministic per-cell multiplicity update used in
// SuperDropletPC::WallLoss, summing the total physical particle number over a
// uniform grid that is marched nt steps of size dt.  For grids where every cell
// is identically boundary-adjacent (a 1x1xN column, or a 1x1x1 cell) the total
// number is a single exponential and must match the analytic well-mixed law.
struct DecayConfig {
    int nx, ny, nz;
    Real dx, dy, dz;
    bool xlo, xhi, ylo, yhi, zlo, zhi; // which faces are absorbing
    Real n0_per_cell;                  // initial multiplicity per cell
    Real r;                            // (constant) droplet radius
};

Real simulate_total_number (const DecayConfig& c, const SDWallLossParams& p,
                            Real dt, int nt)
{
    const Real A_x = c.dy * c.dz;
    const Real A_y = c.dx * c.dz;
    const Real A_z = c.dx * c.dy;
    const Real V   = c.dx * c.dy * c.dz;

    const Real v_settle = Real(0.0); // settling tested separately

    const Real vd_side = sd_vd_turbulent(c.r, p, p.vd_side);
    const Real vd_top  = std::max(sd_vd_turbulent(c.r, p, p.vd_top) - v_settle, Real(0.0));
    const Real vd_bot  = sd_vd_turbulent(c.r, p, p.vd_bottom) + v_settle;

    Real total = Real(0.0);
    for (int k = 0; k < c.nz; ++k) {
        for (int j = 0; j < c.ny; ++j) {
            for (int i = 0; i < c.nx; ++i) {
                Real A_side = Real(0.0);
                if (c.xlo && i == 0)        { A_side += A_x; }
                if (c.xhi && i == c.nx - 1) { A_side += A_x; }
                if (c.ylo && j == 0)        { A_side += A_y; }
                if (c.yhi && j == c.ny - 1) { A_side += A_y; }
                const Real A_top = (c.zhi && k == c.nz - 1) ? A_z : Real(0.0);
                const Real A_bot = (c.zlo && k == 0)        ? A_z : Real(0.0);

                const Real ktot = sd_wall_loss_coefficient(vd_side, A_side, V)
                                + sd_wall_loss_coefficient(vd_top,  A_top, V)
                                + sd_wall_loss_coefficient(vd_bot,  A_bot, V);

                Real n = c.n0_per_cell;
                for (int s = 0; s < nt; ++s) { n *= std::exp(-ktot * dt); }
                total += n;
            }
        }
    }
    return total;
}

// Analytic well-mixed effective rate for the configured (absorbing) faces.
Real analytic_keff (const DecayConfig& c, const SDWallLossParams& p)
{
    const Real Lx = c.nx * c.dx, Ly = c.ny * c.dy, Lz = c.nz * c.dz;
    const Real V  = Lx * Ly * Lz;
    const Real A_side_total = ((c.xlo ? Real(1.0) : Real(0.0)) + (c.xhi ? Real(1.0) : Real(0.0))) * (Ly * Lz)
                            + ((c.ylo ? Real(1.0) : Real(0.0)) + (c.yhi ? Real(1.0) : Real(0.0))) * (Lx * Lz);
    const Real A_top    = (c.zhi ? Real(1.0) : Real(0.0)) * (Lx * Ly);
    const Real A_bottom = (c.zlo ? Real(1.0) : Real(0.0)) * (Lx * Ly);
    return (p.vd_side * A_side_total + p.vd_top * A_top + p.vd_bottom * A_bottom) / V;
}

constexpr Real kRelTol = Real(1.0e-10);

} // namespace

// The first-order coefficient and loss probability are the building blocks of
// the whole parameterization; lock in their exact closed forms.
TEST(SDWallLoss, CoefficientAndProbability)
{
    const Real vd = Real(2.5e-3), A = Real(0.4), V = Real(0.08), dt = Real(0.05);
    const Real k = sd_wall_loss_coefficient(vd, A, V);
    EXPECT_NEAR(k, vd * A / V, std::abs(vd * A / V) * kRelTol);
    EXPECT_NEAR(sd_wall_loss_probability(k, dt),
                Real(1.0) - std::exp(-k * dt),
                kRelTol);
}

// Constant model returns the per-face constant; vd_floor is a hard lower bound.
TEST(SDWallLoss, ConstantDepositionVelocity)
{
    SDWallLossParams p;
    p.dep_vel_model = SDWL_Constant;
    p.vd_side = Real(1.0e-4);
    EXPECT_NEAR(sd_vd_turbulent(Real(1.0e-6), p, p.vd_side), p.vd_side, p.vd_side * kRelTol);

    p.vd_floor = Real(5.0e-4);
    EXPECT_NEAR(sd_vd_turbulent(Real(1.0e-6), p, p.vd_side), p.vd_floor, p.vd_floor * kRelTol);
}

// Diffusion hook: vd(r) = D/delta, D = kB T / (6 pi mu r); monotonically
// decreasing in r.
TEST(SDWallLoss, DiffusionDepositionVelocity)
{
    SDWallLossParams p;
    p.dep_vel_model = SDWL_Diffusion;
    p.diff_temperature = Real(293.0);
    p.diff_viscosity   = Real(1.8e-5);
    p.diff_delta       = Real(1.0e-3);

    const Real r = Real(1.0e-7);
    const Real D = SDWL_kB * p.diff_temperature / (Real(6.0) * PI * p.diff_viscosity * r);
    const Real expected = D / p.diff_delta;
    EXPECT_NEAR(sd_vd_turbulent(r, p, p.vd_side), expected, expected * kRelTol);

    EXPECT_GT(sd_vd_turbulent(Real(1.0e-7), p, p.vd_side),
              sd_vd_turbulent(Real(1.0e-6), p, p.vd_side));
}

// Log-spaced bins are monotonic and clamped to [0, nbins-1].
TEST(SDWallLoss, SizeBinIndex)
{
    SDWallLossParams p;
    p.num_bins = 10;
    p.bin_rmin = Real(1.0e-8);
    p.bin_rmax = Real(1.0e-4);
    EXPECT_EQ(sd_wall_loss_bin_index(Real(1.0e-12), p), 0);          // below range -> first
    EXPECT_EQ(sd_wall_loss_bin_index(Real(1.0e-2), p), p.num_bins-1); // above range -> last
    EXPECT_LE(sd_wall_loss_bin_index(Real(1.0e-7), p),
              sd_wall_loss_bin_index(Real(1.0e-5), p));
}

// Verification (side walls only): for a 1x1xN column every cell touches all
// four side walls identically, so the total number is an exact single
// exponential N(t) = N0 exp(-k_eff t) with k_eff = vd_side * A_side / V_chamber.
TEST(SDWallLoss, ExponentialDecaySideWalls)
{
    DecayConfig c{};
    c.nx = 1; c.ny = 1; c.nz = 8;
    c.dx = Real(0.05); c.dy = Real(0.07); c.dz = Real(0.03);
    c.xlo = c.xhi = c.ylo = c.yhi = true;
    c.zlo = c.zhi = false;             // top/bottom off
    c.n0_per_cell = Real(1.0e6);
    c.r = Real(1.0e-6);

    SDWallLossParams p;
    p.dep_vel_model = SDWL_Constant;
    p.vd_side = Real(2.0e-3); p.vd_top = Real(0.0); p.vd_bottom = Real(0.0);

    const Real dt = Real(0.01);
    const int  nt = 200;
    const Real keff = analytic_keff(c, p);

    const Real N0 = Real(c.nx * c.ny * c.nz) * c.n0_per_cell;
    const Real N_expected = N0 * std::exp(-keff * (nt * dt));
    const Real N_sim = simulate_total_number(c, p, dt, nt);
    EXPECT_NEAR(N_sim, N_expected, N_expected * kRelTol);
}

// Verification (all boundaries): a single 1x1x1 cell touches all six faces, so
// k_total = (vd_side*A_side + vd_top*A_top + vd_bottom*A_bottom)/V exactly, and
// the total number follows the analytic exponential.
TEST(SDWallLoss, ExponentialDecayAllBoundaries)
{
    DecayConfig c{};
    c.nx = 1; c.ny = 1; c.nz = 1;
    c.dx = Real(0.04); c.dy = Real(0.06); c.dz = Real(0.05);
    c.xlo = c.xhi = c.ylo = c.yhi = c.zlo = c.zhi = true;
    c.n0_per_cell = Real(2.0e6);
    c.r = Real(5.0e-7);

    SDWallLossParams p;
    p.dep_vel_model = SDWL_Constant;
    p.vd_side = Real(1.5e-3); p.vd_top = Real(8.0e-4); p.vd_bottom = Real(3.0e-3);

    const Real dt = Real(0.02);
    const int  nt = 150;
    const Real keff = analytic_keff(c, p);

    const Real N0 = c.n0_per_cell;
    const Real N_expected = N0 * std::exp(-keff * (nt * dt));
    const Real N_sim = simulate_total_number(c, p, dt, nt);
    EXPECT_NEAR(N_sim, N_expected, N_expected * kRelTol);
}

// Reduction property: with equal deposition velocities on every face and zero
// settling, k_eff collapses to vd * A_total / V_chamber.
TEST(SDWallLoss, ReductionToUniformRate)
{
    DecayConfig c{};
    c.nx = 1; c.ny = 1; c.nz = 1;
    c.dx = Real(0.04); c.dy = Real(0.06); c.dz = Real(0.05);
    c.xlo = c.xhi = c.ylo = c.yhi = c.zlo = c.zhi = true;

    const Real vd = Real(1.2e-3);
    SDWallLossParams p;
    p.dep_vel_model = SDWL_Constant;
    p.vd_side = vd; p.vd_top = vd; p.vd_bottom = vd;

    const Real Lx = c.dx, Ly = c.dy, Lz = c.dz;
    const Real A_total = Real(2.0) * (Lx*Ly + Ly*Lz + Lx*Lz);
    const Real V = Lx * Ly * Lz;
    const Real keff_expected = vd * A_total / V;

    EXPECT_NEAR(analytic_keff(c, p), keff_expected, keff_expected * kRelTol);
}

#else // ERF_USE_PARTICLES

// Particles disabled in this build: register a trivially-passing placeholder so
// the translation unit is never empty.
TEST(SDWallLoss, DISABLED_RequiresParticles) { SUCCEED(); }

#endif // ERF_USE_PARTICLES
