/**
 * @file ERF_Prob.cpp
 * @brief Cloud Chamber Problem Implementation with Moist-Consistent Initialization
 *
 * This implementation prevents the spurious temperature jumps that can occur when
 * SatAdj microphysics is first called. The key insight is that SatAdj reconstructs
 * pressure using getPgivenRTh(rho*theta, qv), so we must ensure our initial (rho, theta, qv)
 * yields exactly the target (T_0, p_0, RH_0) when processed by SatAdj.
 *
 * Method:
 * 1. Prescribe physical targets: T_0 [K], p_0 [Pa], RH_0 [0-1]
 * 2. Compute qv_0 using SatAdj's exact saturation function: erf_qsatw(T_0, p_0)
 * 3. Compute theta_0 from (T_0, p_0) using getThgivenTandP
 * 4. Compute rho_0 from (theta_0, p_0, qv_0) using getRhogivenThetaPress (moist-aware!)
 * 5. Verify: getPgivenRTh(rho_0*theta_0, qv_0) should exactly equal p_0
 *
 * This ensures SatAdj sees a perfectly consistent state and makes no adjustments.
 */

#include "ERF_Prob.H"
#include "AMReX_Random.H"
#include "ERF_MicrophysicsUtils.H"  // erf_qsatw - CRITICAL: must use same function as SatAdj
#include "ERF_EOS.H"                // EOS helpers for thermodynamic consistency
#include "ERF_Constants.H"

using namespace amrex;

std::unique_ptr<ProblemBase>
amrex_probinit(const amrex_real* problo, const amrex_real* probhi)
{
    return std::make_unique<Problem>(problo, probhi);
}

/**
 * @brief Cloud Chamber Constructor - Moist-Consistent Initialization
 *
 * This constructor implements the moist-consistent approach that prevents temperature
 * jumps in SatAdj. The key is to compute (rho_0, theta_0, qv_0) such that when SatAdj
 * reconstructs pressure via getPgivenRTh(rho*theta, qv), it gets exactly p_0.
 */
Problem::Problem(const amrex::Real* /*problo*/, const amrex::Real* /*probhi*/)
{
    // ========== READ INPUT PARAMETERS ==========
    ParmParse pp("prob");
    pp.query("T_0",  parms.T_0);   // Target temperature [K]
    pp.query("p_0",  parms.p_0);   // Target pressure [Pa]
    pp.query("rh_0", parms.rh_0);  // Target relative humidity [0-1]

    // Velocity field parameters
    pp.query("U_0", parms.U_0); pp.query("V_0", parms.V_0); pp.query("W_0", parms.W_0);
    pp.query("U_0_Pert_Mag", parms.U_0_Pert_Mag);
    pp.query("V_0_Pert_Mag", parms.V_0_Pert_Mag);
    pp.query("W_0_Pert_Mag", parms.W_0_Pert_Mag);

    // ========== DERIVE MOIST-CONSISTENT BASE STATE ==========
    // CRITICAL: This section ensures SatAdj will reproduce exactly (T_0, p_0, RH_0)
    const Real rdOcp = R_d / Cp_d;

    // Step 1: Compute qv_0 using EXACT same function as SatAdj
    // erf_qsatw expects: T [K], p [hPa], returns qsat [kg/kg]
    Real qsat0;
    erf_qsatw(parms.T_0, 0.01*parms.p_0, qsat0);  // Convert Pa to hPa for erf_qsatw
    parms.qv_0 = std::min(parms.rh_0 * qsat0, qsat0);  // Ensure qv <= qsat

    // Step 2: Compute theta_0 from target (T_0, p_0) - standard dry relation
    parms.theta_0 = getThgivenTandP(parms.T_0, parms.p_0, rdOcp);

    // Step 3: Compute rho_0 with MOIST AWARENESS - this is the key difference!
    // This function accounts for water vapor when computing density
    parms.rho_0 = getRhogivenThetaPress(parms.theta_0, parms.p_0, rdOcp, parms.qv_0);

    // Initialize base fields with these moist-consistent (rho, theta)
    // Moisture will be set uniformly in init_custom_pert below
    init_base_parms(parms.rho_0, parms.theta_0);

    // ========== VERIFICATION CHECK ==========
    // Verify that SatAdj's pressure reconstruction yields exactly p_0
    if (ParallelDescriptor::IOProcessor()) {
        Real p_check = getPgivenRTh(parms.rho_0 * parms.theta_0, parms.qv_0);
        Real T_check = getTgivenRandRTh(parms.rho_0, parms.rho_0*parms.theta_0, parms.qv_0);
        amrex::Print() << "[CloudChamber] IC targets: T0=" << parms.T_0
                       << " K, p0=" << parms.p_0 << " Pa, RH0=" << parms.rh_0
                       << " -> qv0=" << parms.qv_0
                       << " | derived rho0=" << parms.rho_0
                       << " theta0=" << parms.theta_0
                       << " | check p=" << p_check << " Pa, T=" << T_check << " K\n";
    }
}

/**
 * @brief Initialize Custom Perturbations for Cloud Chamber
 *
 * This function sets the actual state values that will be used in the simulation.
 * For a cloud chamber, we want:
 * - Uniform temperature (NO thermal perturbations)
 * - Uniform moisture at computed qv_0
 * - Optional velocity perturbations to seed turbulence
 *
 * CRITICAL: We set RhoTheta_comp = 0 to keep theta exactly uniform, preventing
 * any temperature variations that could trigger unwanted convection.
 */
void
Problem::init_custom_pert(
    const amrex::Box&  bx,
    const amrex::Box& xbx,
    const amrex::Box& ybx,
    const amrex::Box& zbx,
    [[maybe_unused]] amrex::Array4<amrex::Real const> const& state,
    amrex::Array4<amrex::Real      > const& state_pert,
    amrex::Array4<amrex::Real      > const& x_vel_pert,
    amrex::Array4<amrex::Real      > const& y_vel_pert,
    amrex::Array4<amrex::Real      > const& z_vel_pert,
    [[maybe_unused]] amrex::Array4<amrex::Real      > const& r_hse,
    [[maybe_unused]] amrex::Array4<amrex::Real      > const& p_hse,
    [[maybe_unused]] amrex::Array4<amrex::Real const> const& z_nd,
    [[maybe_unused]] amrex::Array4<amrex::Real const> const& z_cc,
    [[maybe_unused]] amrex::GeometryData const& geomdata,
    amrex::Array4<amrex::Real const> const& /*mf_m*/,
    amrex::Array4<amrex::Real const> const& /*mf_u*/,
    amrex::Array4<amrex::Real const> const& /*mf_v*/,
    const SolverChoice& sc,
    [[maybe_unused]] const int lev)
{
    const bool use_moisture = (sc.moisture_type != MoistureType::None);

    // ========== TEMPERATURE FIELD: KEEP STRICTLY UNIFORM ==========
    // CRITICAL: NO temperature perturbations for cloud chamber!
    // This ensures we maintain the exact thermodynamic consistency established above.
    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
        state_pert(i,j,k,RhoTheta_comp) = 0.0;  // No perturbations to rho*theta
    });

    // ========== MOISTURE FIELD: SET UNIFORM qv_0 ==========
    // Set water vapor uniformly to the moist-consistent value computed above.
    // Start with zero cloud water - any condensation will be handled by SatAdj.
    if (use_moisture) {
        const Real rho0   = parms.rho_0;   // Moist-consistent density
        const Real qv0    = parms.qv_0;    // qv = RH_0 * qsat(T_0, p_0)

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            state_pert(i,j,k,RhoQ1_comp) = rho0 * qv0;  // Water vapor: rho*qv
            state_pert(i,j,k,RhoQ2_comp) = 0.0;         // Cloud liquid: rho*qc = 0
            // Note: If carrying more moisture species (ice, rain, etc.), set their rho*q = 0 here
        });
    }

    // ========== VELOCITY FIELD: OPTIONAL TURBULENCE SEEDING ==========
    // Set mean flow + optional random perturbations to seed turbulence.
    // This is the only "noise" we add - no thermal perturbations!

    // X-velocity: mean + random perturbations
    if (parms.U_0_Pert_Mag != 0.0) {
        ParallelForRNG(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE (int i,int j,int k, const amrex::RandomEngine& engine) noexcept {
            Real r = amrex::Random(engine); // [0,1] uniform random
            x_vel_pert(i,j,k) = parms_d.U_0 + (r*2.0 - 1.0)*parms_d.U_0_Pert_Mag;  // Mean + [-mag, +mag]
        });
    } else {
        ParallelFor(xbx, [=, U0=parms.U_0] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            x_vel_pert(i,j,k) = U0;  // Just mean flow, no perturbations
        });
    }

    // Y-velocity: mean + random perturbations
    if (parms.V_0_Pert_Mag != 0.0) {
        ParallelForRNG(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE (int i,int j,int k, const amrex::RandomEngine& engine) noexcept {
            Real r = amrex::Random(engine);
            y_vel_pert(i,j,k) = parms_d.V_0 + (r*2.0 - 1.0)*parms_d.V_0_Pert_Mag;
        });
    } else {
        ParallelFor(ybx, [=, V0=parms.V_0] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            y_vel_pert(i,j,k) = V0;
        });
    }

    // Z-velocity: mean + random perturbations
    if (parms.W_0_Pert_Mag != 0.0) {
        ParallelForRNG(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE (int i,int j,int k, const amrex::RandomEngine& engine) noexcept {
            Real r = amrex::Random(engine);
            z_vel_pert(i,j,k) = parms_d.W_0 + (r*2.0 - 1.0)*parms_d.W_0_Pert_Mag;
        });
    } else {
        ParallelFor(zbx, [=, W0=parms.W_0] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            z_vel_pert(i,j,k) = W0;
        });
    }
}
