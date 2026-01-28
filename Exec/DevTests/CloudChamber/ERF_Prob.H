#ifndef ERF_PROB_H_
#define ERF_PROB_H_

#include <string>
#include "AMReX_REAL.H"
#include "ERF_ProbCommon.H"

/**
 * @brief Cloud Chamber Problem Parameters with Moist-Consistent Initialization
 *
 * This implementation prevents spurious temperature jumps in SatAdj by ensuring
 * thermodynamic consistency between initial conditions and SatAdj's calculations.
 * The approach computes (rho_0, theta_0, qv_0) from physical targets (T_0, p_0, RH_0) using
 * the same saturation function and EOS relations that SatAdj uses.
 */
struct ProbParm : ProbParmDefaults {
  // ========== THERMODYNAMIC TARGETS ==========
  // These are the physical conditions we want the chamber to have initially.
  // They are converted to thermodynamically consistent (rho, theta, qv) in the constructor.

  amrex::Real T_0   = 294.0;     // Target initial temperature [K] (21C room temp)
  amrex::Real p_0   = 1.0e5;     // Target initial pressure [Pa] (standard atmospheric)
  amrex::Real rh_0  = 1.00;      // Target initial relative humidity [0-1]; use <1.0 to avoid condensation

  // ========== VELOCITY FIELD ==========
  // Mean flow velocities and optional perturbations to seed turbulence
  amrex::Real U_0 = 0.0, V_0 = 0.0, W_0 = 0.0;                                    // Mean velocities [m/s]
  amrex::Real U_0_Pert_Mag = 0.0, V_0_Pert_Mag = 0.0, W_0_Pert_Mag = 0.0;        // Velocity perturbation magnitudes [m/s]

  // ========== DERIVED THERMODYNAMIC STATE ==========
  // These are computed in the constructor to ensure moist consistency.
  // They are the actual (rho, theta, qv) values used to initialize the simulation.
  amrex::Real theta_0 = 0.0;     // Derived potential temperature [K] consistent with (T_0, p_0)
  amrex::Real qv_0    = 0.0;     // Derived water vapor mixing ratio [kg/kg] = RH_0 * qsat(T_0, p_0)
};

/**
 * @brief Cloud Chamber Problem Class with Moist-Consistent Initialization
 *
 * This class implements a cloud chamber simulation that maintains exact thermodynamic
 * consistency with the SatAdj microphysics scheme. Key features:
 *
 * - Prevents spurious temperature jumps at simulation start
 * - Uses same saturation function (erf_qsatw) as SatAdj for qv computation
 * - Computes moist-aware density using getRhogivenThetaPress
 * - Maintains uniform temperature field (no thermal perturbations)
 * - Supports optional velocity perturbations for turbulence seeding
 * - Uses constant density HSE initialization for uniform chamber conditions
 */
class Problem : public ProblemBase {
public:
    Problem(const amrex::Real* problo, const amrex::Real* probhi);

    // Include constant density HSE initialization for uniform chamber conditions
#include "Prob/ERF_InitConstantDensityHSE.H"

    void init_custom_pert (
        const amrex::Box&  bx,
        const amrex::Box& xbx,
        const amrex::Box& ybx,
        const amrex::Box& zbx,
        amrex::Array4<amrex::Real const> const& state,
        amrex::Array4<amrex::Real      > const& state_pert,
        amrex::Array4<amrex::Real      > const& x_vel_pert,
        amrex::Array4<amrex::Real      > const& y_vel_pert,
        amrex::Array4<amrex::Real      > const& z_vel_pert,
        amrex::Array4<amrex::Real      > const& r_hse,
        amrex::Array4<amrex::Real      > const& p_hse,
        amrex::Array4<amrex::Real const> const& z_nd,
        amrex::Array4<amrex::Real const> const& z_cc,
        amrex::GeometryData const& geomdata,
        amrex::Array4<amrex::Real const> const& mf_m,
        amrex::Array4<amrex::Real const> const& mf_u,
        amrex::Array4<amrex::Real const> const& mf_v,
        const SolverChoice& sc,
        const int lev) override;

private:
    ProbParm parms;
};

#endif
