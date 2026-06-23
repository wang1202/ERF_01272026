#ifndef ERF_SD_WALL_LOSS_H_
#define ERF_SD_WALL_LOSS_H_

#ifdef ERF_USE_PARTICLES

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_Random.H>
#include "ERF_Constants.H"

/*! \file ERF_SDWallLoss.H
 *  \brief Size-dependent boundary-surface deposition loss for the Super-Droplet Method.
 *
 *  Particle loss to the chamber boundary surfaces is modeled as a first-order
 *  (linear) absorbing sink acting only in the Eulerian cells adjacent to a
 *  chamber boundary.  Unlike the original side-wall-only version, this applies
 *  to *all* chamber boundaries: the four lateral side walls, the bottom, and the
 *  top.  Each boundary face may carry its own deposition velocity.
 *
 *  For a boundary-adjacent cell of volume \f$V_{cell}\f$, the total first-order
 *  loss coefficient for a droplet of radius \f$r\f$ is the sum over every
 *  chamber boundary face \f$b\f$ touched by the cell:
 *
 *  \f[ k_{total}(r) = \sum_b v_{d,b}(r)\, \frac{A_b}{V_{cell}} \f]
 *
 *  with per-boundary deposition velocities
 *
 *  \f[ v_{d,side}(r)   = v_{d,turb,side}(r) \f]
 *  \f[ v_{d,top}(r)    = \max\!\big(v_{d,turb,top}(r) - v_{settle}(r),\,0\big) \f]
 *  \f[ v_{d,bottom}(r) = v_{d,turb,bottom}(r) + v_{settle}(r) \f]
 *
 *  where \f$v_{settle}(r)\f$ is the gravitational terminal settling velocity
 *  (settling enhances deposition on the floor and opposes it on the ceiling).
 *  The settling correction is optional.
 *
 *  The per-step loss probability is
 *  \f$P_{loss}(r) = 1 - \exp(-k_{total}(r)\,\Delta t)\f$, and the loss is applied
 *  to each super-droplet's multiplicity either deterministically (Option A),
 *
 *  \f[ \xi_{new} = \xi_{old}\, \exp\!\left(-k_{total}(r)\,\Delta t\right), \f]
 *
 *  or by stochastic thinning (Option B): draw the removed multiplicity from a
 *  Binomial\f$(\xi,P_{loss})\f$ distribution (here approximated by Poisson /
 *  Gaussian for large \f$\xi\f$) and set \f$\xi_{new} = \xi_{old} -\f$ removed.
 *
 *  For a well-mixed chamber with constant deposition velocities, summing the
 *  per-cell sink over all boundary-adjacent cells recovers the analytic decay
 *  of the total particle number,
 *
 *  \f[ N(t) = N_0\, e^{-k_{eff}\,t}, \qquad
 *      k_{eff} = \frac{v_{d,side}A_{side} + v_{d,top}A_{top}
 *                      + v_{d,bottom}A_{bottom}}{V_{chamber}}, \f]
 *
 *  which for zero settling and equal velocities reduces to
 *  \f$k_{eff} = v_d\,A_{total}/V_{chamber}\f$ (see ERF_SuperDropletPCWallLoss.cpp).
 *
 *  References:
 *   - Crump, J. G. and Seinfeld, J. H. (1981), J. Aerosol Sci., 12(5), 405-415.
 *   - McMurry, P. H. and Rader, D. J. (1985), Aerosol Sci. Technol., 4(3), 249-268.
 *   - Shima et al. (2009), Q.J.R. Meteorol. Soc. (super-droplet method).
 */

/*! \brief Turbulent deposition-velocity model (POD-friendly int dispatch). */
enum SDWallLossDepVelModel {
    SDWL_Constant  = 0, /*!< Constant, size-independent turbulent deposition velocity */
    SDWL_Diffusion = 1  /*!< Stokes-Einstein Brownian-diffusion deposition velocity (hook) */
};

/*! \brief Multiplicity-loss method. */
enum SDWallLossMethod {
    SDWL_Deterministic = 0, /*!< Option A: xi_new = xi_old * exp(-k*dt) */
    SDWL_Stochastic    = 1  /*!< Option B: binomial/Poisson/Gaussian thinning */
};

/*! \brief Maximum number of diagnostic size bins (fixed for GPU-friendly arrays). */
inline constexpr int SDWL_MAX_BINS = 64;

/*! \brief Configuration parameters for the boundary-deposition wall-loss model.
 *
 *  Plain-old-data so it can be captured by value into GPU device kernels.  All
 *  inputs are user configurable via `super_droplets.wall_loss.*`.
 */
struct SDWallLossParams {
    bool enable{false};                    /*!< Master enable flag */
    int  method{SDWL_Deterministic};       /*!< Deterministic (A) or stochastic (B) */
    int  dep_vel_model{SDWL_Constant};     /*!< Turbulent v_d(r) model */

    // Deactivation of depleted super-droplets
    bool deactivate_on_depletion{false};   /*!< Remove SDs whose xi falls below threshold */
    amrex::Real deactivate_threshold{1.0e-30}; /*!< Multiplicity threshold for removal */

    // Per-boundary turbulent deposition velocities [m/s] (constant model)
    amrex::Real vd_side{1.0e-4};           /*!< Side walls (x/y) */
    amrex::Real vd_top{1.0e-4};            /*!< Top boundary */
    amrex::Real vd_bottom{1.0e-4};         /*!< Bottom boundary */

    // Optional gravitational settling correction on top/bottom
    bool settling_correction{false};       /*!< Apply -/+ v_settle on top/bottom */

    // Stokes-Einstein diffusion model parameters (size-dependent hook)
    amrex::Real diff_temperature{293.0};   /*!< Gas temperature for D(r) [K] */
    amrex::Real diff_viscosity{1.8e-5};    /*!< Gas dynamic viscosity [Pa s] */
    amrex::Real diff_delta{1.0e-3};        /*!< Wall boundary-layer thickness [m] */
    amrex::Real vd_floor{0.0};             /*!< Lower bound on v_d(r) [m/s] */

    // Diagnostics binning
    int  num_bins{16};                     /*!< Number of (log-spaced) size bins */
    amrex::Real bin_rmin{1.0e-8};          /*!< Min radius for binning [m] */
    amrex::Real bin_rmax{1.0e-4};          /*!< Max radius for binning [m] */
    int  diag_interval{100};               /*!< Steps between diagnostic writes (0 disables) */
};

/*! \brief Boltzmann constant [J/K] (local to keep the module self-contained). */
inline constexpr amrex::Real SDWL_kB = amrex::Real(1.380649e-23);

/*! \brief Turbulent wall deposition velocity [m/s] for one boundary face.
 *
 *  Dispatches on params.dep_vel_model: SDWL_Constant returns the per-face
 *  constant vd_const_face; SDWL_Diffusion returns the size-dependent
 *  Stokes-Einstein value D(r)/delta (same for all faces).  Extend here for
 *  Cunningham slip or turbulent-inertial deposition without touching call sites.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sd_vd_turbulent (amrex::Real r, const SDWallLossParams& p, amrex::Real vd_const_face)
{
    amrex::Real vd;
    if (p.dep_vel_model == SDWL_Diffusion) {
        const amrex::Real r_safe = amrex::max(r, amrex::Real(1.0e-12));
        const amrex::Real D = SDWL_kB * p.diff_temperature
                            / (amrex::Real(6.0) * PI * p.diff_viscosity * r_safe);
        vd = D / amrex::max(p.diff_delta, amrex::Real(1.0e-12));
    } else {
        vd = vd_const_face;
    }
    return amrex::max(vd, p.vd_floor);
}

/*! \brief First-order loss coefficient contribution k = v_d * A / V_cell [1/s]. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sd_wall_loss_coefficient (amrex::Real vd, amrex::Real area, amrex::Real V_cell)
{
    return vd * area / V_cell;
}

/*! \brief Per-step loss probability P_loss = 1 - exp(-k_total * dt). */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sd_wall_loss_probability (amrex::Real k_total, amrex::Real dt)
{
    return amrex::Real(1.0) - std::exp(-k_total * dt);
}

/*! \brief Stochastic (Option B) removed multiplicity ~ Binomial(xi, P).
 *
 *  Multiplicities are typically large, so the Binomial is approximated by a
 *  Poisson in the rare-event regime (mean < 30) and by a clamped Gaussian
 *  otherwise (mean, variance = xi*P*(1-P)).  Both are standard large-n limits
 *  of the Binomial distribution.  The result is clamped to [0, xi].
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real sd_wall_loss_stochastic_removed (amrex::Real xi, amrex::Real P,
                                             amrex::RandomEngine const& engine)
{
    if (xi <= amrex::Real(0.0) || P <= amrex::Real(0.0)) { return amrex::Real(0.0); }
    if (P >= amrex::Real(1.0)) { return xi; }
    const amrex::Real mean = xi * P;
    amrex::Real removed;
    if (mean < amrex::Real(30.0)) {
        removed = amrex::Real(amrex::RandomPoisson(mean, engine));
    } else {
        const amrex::Real sd = std::sqrt(mean * (amrex::Real(1.0) - P));
        removed = amrex::RandomNormal(mean, sd, engine);
        if (removed < amrex::Real(0.0)) { removed = amrex::Real(0.0); }
    }
    return amrex::min(removed, xi);
}

/*! \brief Diagnostic size-bin index for radius r (log-spaced), clamped to [0,nbins-1]. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int sd_wall_loss_bin_index (amrex::Real r, const SDWallLossParams& p)
{
    const int nbins = (p.num_bins < 1) ? 1 : ((p.num_bins > SDWL_MAX_BINS) ? SDWL_MAX_BINS : p.num_bins);
    const amrex::Real rmin = amrex::max(p.bin_rmin, amrex::Real(1.0e-12));
    const amrex::Real rmax = amrex::max(p.bin_rmax, rmin * amrex::Real(1.0000001));
    const amrex::Real lr   = std::log(amrex::max(r, amrex::Real(1.0e-12)));
    const amrex::Real lo   = std::log(rmin);
    const amrex::Real hi   = std::log(rmax);
    int b = int(amrex::Math::floor((lr - lo) / (hi - lo) * amrex::Real(nbins)));
    if (b < 0)       { b = 0; }
    if (b >= nbins)  { b = nbins - 1; }
    return b;
}

#endif // ERF_USE_PARTICLES
#endif // ERF_SD_WALL_LOSS_H_
