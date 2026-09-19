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
 *  A second, chamber-calibrated mode accepts measured first-order loss-rate
 *  profiles k(Dp).  This is applied to every suspended particle, consistent
 *  with the well-mixed chamber analysis used to infer those profiles.  A
 *  second profile may be supplied to represent time-varying chamber conditions.
 *
 *  References:
 *   - Crump, J. G. and Seinfeld, J. H. (1981), J. Aerosol Sci., 12(5), 405-415.
 *   - McMurry, P. H. and Rader, D. J. (1985), Aerosol Sci. Technol., 4(3), 249-268.
 *   - Lai, A. C. K. (2004), Indoor Air, particle-deposition review.
 *   - Nah et al. (2017), Atmos. Chem. Phys., 17, 2297-2310.
 *   - Wang et al. (2018), Atmos. Meas. Tech., 11, 6577-6588.
 *   - Shima et al. (2009), Q.J.R. Meteorol. Soc. (super-droplet method).
 */

/*! \brief Wall-loss coefficient model (POD-friendly int dispatch). */
enum SDWallLossDepVelModel {
    SDWL_Constant     = 0, /*!< Constant, size-independent turbulent deposition velocity */
    SDWL_Diffusion    = 1, /*!< Slip-corrected Brownian-diffusion deposition velocity */
    SDWL_MeasuredRate = 2  /*!< Measured, size-dependent whole-chamber loss rate k(Dp) */
};

/*! \brief Multiplicity-loss method. */
enum SDWallLossMethod {
    SDWL_Deterministic = 0, /*!< Option A: xi_new = xi_old * exp(-k*dt) */
    SDWL_Stochastic    = 1  /*!< Option B: binomial/Poisson/Gaussian thinning */
};

/*! \brief Maximum number of diagnostic size bins (fixed for GPU-friendly arrays). */
inline constexpr int SDWL_MAX_BINS = 64;

/*! \brief Maximum number of points in a measured loss-rate profile. */
inline constexpr int SDWL_MAX_PROFILE_POINTS = 64;

/*! \brief Configuration parameters for the boundary-deposition wall-loss model.
 *
 *  Plain-old-data so it can be captured by value into GPU device kernels.  All
 *  inputs are user configurable via `super_droplets.wall_loss.*`.
 */
struct SDWallLossParams {
    bool enable{false};                    /*!< Master enable flag */
    int  method{SDWL_Deterministic};       /*!< Deterministic (A) or stochastic (B) */
    int  dep_vel_model{SDWL_Constant};     /*!< Face-velocity or measured-rate model */

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
    bool diff_slip_correction{true};       /*!< Apply Cunningham correction to D(r) */
    amrex::Real diff_mean_free_path{6.6e-8}; /*!< Gas mean free path [m] */
    amrex::Real vd_floor{0.0};             /*!< Lower bound on v_d(r) [m/s] */

    // Chamber-calibrated first-order loss-rate profiles. Diameters are [m],
    // rates are [1/s], and the diameter grid is shared by both profiles.
    int profile_size{0};
    amrex::Real profile_diameter[SDWL_MAX_PROFILE_POINTS]{};
    amrex::Real profile_rate_start[SDWL_MAX_PROFILE_POINTS]{};
    amrex::Real profile_rate_end[SDWL_MAX_PROFILE_POINTS]{};
    bool profile_time_varying{false};
    amrex::Real profile_time_start{0.0};   /*!< Time of start profile [s] */
    amrex::Real profile_time_end{0.0};     /*!< Time of end profile [s] */

    // Diagnostics binning
    int  num_bins{16};                     /*!< Number of (log-spaced) size bins */
    amrex::Real bin_rmin{1.0e-8};          /*!< Min radius for binning [m] */
    amrex::Real bin_rmax{1.0e-4};          /*!< Max radius for binning [m] */
    int  diag_interval{100};               /*!< Steps between diagnostic writes (0 disables) */
};

/*! \brief Boltzmann constant [J/K] (local to keep the module self-contained). */
inline constexpr amrex::Real SDWL_kB = amrex::Real(1.380649e-23);

/*! \brief Cunningham slip correction for a particle of radius r. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sd_cunningham_slip_correction (amrex::Real r, amrex::Real mean_free_path)
{
    const amrex::Real r_safe = amrex::max(r, amrex::Real(1.0e-12));
    const amrex::Real lambda = amrex::max(mean_free_path, amrex::Real(0.0));
    if (lambda == amrex::Real(0.0)) { return amrex::Real(1.0); }
    const amrex::Real Kn = lambda / r_safe;
    return amrex::Real(1.0) + Kn *
           (amrex::Real(1.257) + amrex::Real(0.4) * std::exp(amrex::Real(-1.1) / Kn));
}

/*! \brief Stokes-Einstein Brownian diffusivity, optionally slip corrected [m2/s]. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sd_brownian_diffusivity (amrex::Real r, const SDWallLossParams& p)
{
    const amrex::Real r_safe = amrex::max(r, amrex::Real(1.0e-12));
    const amrex::Real Cc = p.diff_slip_correction
                         ? sd_cunningham_slip_correction(r_safe, p.diff_mean_free_path)
                         : amrex::Real(1.0);
    return SDWL_kB * p.diff_temperature * Cc
         / (amrex::Real(6.0) * PI * p.diff_viscosity * r_safe);
}

/*! \brief Turbulent wall deposition velocity [m/s] for one boundary face.
 *
 *  Dispatches on params.dep_vel_model: SDWL_Constant returns the per-face
 *  constant vd_const_face; SDWL_Diffusion returns the size-dependent
 *  slip-corrected Stokes-Einstein value D(r)/delta (same for all faces).
 *  The measured-rate model is handled separately because it supplies a
 *  whole-chamber first-order coefficient rather than a face velocity.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sd_vd_turbulent (amrex::Real r, const SDWallLossParams& p, amrex::Real vd_const_face)
{
    amrex::Real vd;
    if (p.dep_vel_model == SDWL_Diffusion) {
        const amrex::Real D = sd_brownian_diffusivity(r, p);
        vd = D / amrex::max(p.diff_delta, amrex::Real(1.0e-12));
    } else {
        vd = vd_const_face;
    }
    return amrex::max(vd, p.vd_floor);
}

/*! \brief Interpolate one measured k(Dp) profile linearly in log diameter.
 *
 *  Values outside the supplied diameter range use the nearest endpoint.  This
 *  avoids unsupported extrapolation beyond the chamber characterization data.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sd_wall_loss_profile_rate (amrex::Real diameter,
                                       const SDWallLossParams& p,
                                       bool use_end_profile)
{
    const int n = (p.profile_size < 1) ? 0
                : ((p.profile_size > SDWL_MAX_PROFILE_POINTS)
                    ? SDWL_MAX_PROFILE_POINTS : p.profile_size);
    if (n == 0) { return amrex::Real(0.0); }

    const amrex::Real d = amrex::max(diameter, amrex::Real(1.0e-12));
    if (n == 1 || d <= p.profile_diameter[0]) {
        const amrex::Real value = use_end_profile ? p.profile_rate_end[0]
                                                  : p.profile_rate_start[0];
        return amrex::max(value, amrex::Real(0.0));
    }
    if (d >= p.profile_diameter[n-1]) {
        const amrex::Real value = use_end_profile ? p.profile_rate_end[n-1]
                                                  : p.profile_rate_start[n-1];
        return amrex::max(value, amrex::Real(0.0));
    }

    int hi = 1;
    while (hi < n && d > p.profile_diameter[hi]) { ++hi; }
    const int lo = hi - 1;
    const amrex::Real log_dlo = std::log(p.profile_diameter[lo]);
    const amrex::Real fraction = (std::log(d) - log_dlo)
                               / (std::log(p.profile_diameter[hi]) - log_dlo);
    const amrex::Real rate_lo = use_end_profile ? p.profile_rate_end[lo]
                                                : p.profile_rate_start[lo];
    const amrex::Real rate_hi = use_end_profile ? p.profile_rate_end[hi]
                                                : p.profile_rate_start[hi];
    const amrex::Real value = rate_lo + fraction * (rate_hi - rate_lo);
    return amrex::max(value, amrex::Real(0.0));
}

/*! \brief Measured whole-chamber first-order loss coefficient k(Dp,t) [1/s].
 *
 *  Diameter interpolation is performed first.  When a second profile is
 *  configured, rates are then interpolated linearly in time and clamped to the
 *  start/end profiles.  This represents the paired initial/final seed-loss
 *  characterizations recommended for chambers whose wall loss changes in time.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sd_wall_loss_measured_rate (amrex::Real radius, amrex::Real time,
                                        const SDWallLossParams& p)
{
    const amrex::Real diameter = amrex::Real(2.0) * radius;
    const amrex::Real k_start = sd_wall_loss_profile_rate(diameter, p, false);
    if (!p.profile_time_varying) { return k_start; }

    const amrex::Real k_end = sd_wall_loss_profile_rate(diameter, p, true);
    const amrex::Real duration = p.profile_time_end - p.profile_time_start;
    if (duration <= amrex::Real(0.0)) { return k_start; }
    amrex::Real fraction = (time - p.profile_time_start) / duration;
    fraction = amrex::max(amrex::Real(0.0), amrex::min(amrex::Real(1.0), fraction));
    return k_start + fraction * (k_end - k_start);
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
