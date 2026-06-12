/**
 * @file ERF_WallFluxesAllFaces.cpp
 * @brief Implementation of wall flux boundary conditions for all domain faces
 * 
 * This module provides wall flux boundary conditions suitable for cloud chamber
 * simulations and other applications requiring wall heat/moisture exchange.
 * 
 * Key features:
 * - Bulk exchange formulas for momentum, heat, and moisture
 * - Independent coefficients per face with proper face orientation
 * - Support for free-slip and no-slip momentum conditions
 * - Water-coated wall support for moisture exchange
 * - Automatic momentum model detection from boundary conditions
 * - 4-point velocity averaging for robust tangential velocity calculation
 * 
 * Physics (as implemented):
 * The wall fluxes are computed using face-oriented bulk exchange formulas:
 * - τ_tn = -s * ρ * Cm * |U_t| * u_t             (tangential shear on face with normal n)
 * - H_n  =  s * ρ * Ch * |U_t| * (θ_wall - θ_air) (ρ·θ flux, not energy)
 * - Q_n  =  s * ρ * Cq * |U_t| * (q_wall - q_air) (ρ·qv flux)
 * 
 * where:
 * - s = +1 for lo faces (wall-to-air flux positive into domain)
 * - s = -1 for hi faces (wall-to-air flux negative into domain)
 * - Cm, Ch, Cq are dimensionless exchange coefficients
 * - |U_t| is tangential velocity magnitude from 4-point stencil averaging
 * - θ_wall is computed from user-specified T_wall using inverse Exner function
 * - Heat flux is computed as potential temperature flux for ERF's θ-based formulation
 * - q_wall is computed from T_wall using saturation vapor pressure for water-coated walls
 * 
 * Temperature handling:
 * Users specify actual thermodynamic temperature T_wall [K] in input files,
 * which is physically intuitive and directly used for saturation calculations.
 * For heat flux computations, T_wall is converted to potential temperature
 * using the local pressure to maintain consistency with ERF's θ-based formulation.
 */

#include "ERF_WallFluxesAllFaces.H"
#include "ERF_PAFNN_WallModel.H"
#include "ERF_IndexDefines.H"
#include "ERF_Utils.H"
#include "ERF_EOS.H"
#include "ERF_MicrophysicsUtils.H"
#include "ERF_Constants.H"
#include "ERF_DataStruct.H"
#include <AMReX_BCRec.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_Math.H>
#include "AMReX_PhysBCFunct.H"

using namespace amrex;

/**
 * Determine if momentum drag should be applied on a face
 * 
 * @param face_idx Face index (0=xlo, 1=xhi, 2=ylo, 3=yhi, 4=zlo, 5=zhi)
 * @param face_params Face-specific wall flux parameters
 * @param domain_bcs_type Vector of BCRec objects for velocity components
 * @param var_idx Variable index (unused, kept for interface compatibility)
 * @return True if momentum drag should be applied, false for free-slip
 * 
 * For Auto mode, infers momentum model from velocity boundary conditions:
 * - Checks tangential velocity components for the face
 * - Free-slip if both tangential components have reflect_even BC
 * - No-slip (drag) if either tangential component has reflect_odd BC
 */
bool should_apply_momentum_drag(
    int face_idx,
    const WallFaceParams& face_params,
    const amrex::Vector<amrex::BCRec>& domain_bcs_type,
    int var_idx [[maybe_unused]])
{
    switch (face_params.model) {
        case WallFaceParams::MomModel::Drag:
            return true;
        case WallFaceParams::MomModel::FreeSlip:
            return false;
        case WallFaceParams::MomModel::Auto:
        default:
            // Infer from AMReX velocity BC types for tangential components
            // face_idx: 0=xlo, 1=xhi, 2=ylo, 3=yhi, 4=zlo, 5=zhi
            int dir = face_idx / 2;  // 0=x, 1=y, 2=z
            int side = face_idx % 2; // 0=lo, 1=hi
            
            if (domain_bcs_type.size() >= 3) {
                // Check tangential components for this face
                bool slip1 = false, slip2 = false;
                
                if (dir == 0) {
                    // X-face: check v (component 1) and w (component 2) BCs
                    int bc_v = (side == 0) ? domain_bcs_type[1].lo(dir) : domain_bcs_type[1].hi(dir);
                    int bc_w = (side == 0) ? domain_bcs_type[2].lo(dir) : domain_bcs_type[2].hi(dir);
                    slip1 = (bc_v == amrex::BCType::reflect_even);
                    slip2 = (bc_w == amrex::BCType::reflect_even);
                } else if (dir == 1) {
                    // Y-face: check u (component 0) and w (component 2) BCs  
                    int bc_u = (side == 0) ? domain_bcs_type[0].lo(dir) : domain_bcs_type[0].hi(dir);
                    int bc_w = (side == 0) ? domain_bcs_type[2].lo(dir) : domain_bcs_type[2].hi(dir);
                    slip1 = (bc_u == amrex::BCType::reflect_even);
                    slip2 = (bc_w == amrex::BCType::reflect_even);
                } else {
                    // Z-face: check u (component 0) and v (component 1) BCs
                    int bc_u = (side == 0) ? domain_bcs_type[0].lo(dir) : domain_bcs_type[0].hi(dir);
                    int bc_v = (side == 0) ? domain_bcs_type[1].lo(dir) : domain_bcs_type[1].hi(dir);
                    slip1 = (bc_u == amrex::BCType::reflect_even);
                    slip2 = (bc_v == amrex::BCType::reflect_even);
                }
                
                // If both tangential components are slip (reflect_even), use free slip
                // If either is no-slip (reflect_odd), apply drag
                return !(slip1 && slip2);
            }
            return true; // default to drag if BC types unavailable
    }
}

/**
 * Compute oriented scalar flux using bulk exchange formula
 * 
 * @param s Face orientation factor (+1 for lo faces, -1 for hi faces)
 * @param rho Density [kg/m³]
 * @param C Exchange coefficient [dimensionless]
 * @param Ut Tangential velocity magnitude [m/s]
 * @param wall_value Wall-side scalar value
 * @param air_value Air-side scalar value
 * @return Oriented scalar flux - units depend on scalar; for θ: K·kg·m⁻²·s⁻¹, for qv: kg·m⁻²·s⁻¹
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real oriented_flux(
    amrex::Real s, 
    amrex::Real rho, 
    amrex::Real C, 
    amrex::Real Ut, 
    amrex::Real wall_value, 
    amrex::Real air_value)
{
    return s * rho * C * Ut * (wall_value - air_value);
}

/**
 * Compute saturation mixing ratio at wall
 *
 * @param T_wall Wall thermodynamic temperature [K]
 * @param pressure_pa Pressure in Pascals
 * @param wetness Wall wetness (0.0=dry, 1.0=saturated) [dimensionless]
 * @return Saturation mixing ratio scaled by wetness [kg/kg]
 *
 * For wet walls (wetness > 0), calculates saturation vapor pressure and
 * scales it by the wetness factor. For completely dry walls (wetness = 0),
 * returns 0.0. Intermediate values (0 < wetness < 1) represent partially
 * wet surfaces with reduced moisture availability.
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_qsat_wall(
    amrex::Real T_wall,
    amrex::Real pressure_pa,
    amrex::Real wetness)
{
    if (wetness <= 0.0) {
        return 0.0; // dry wall
    }

    // Use the provided thermodynamic temperature directly
    Real T_wall_use = T_wall;

    // Convert pressure from Pa to hPa for erf_qsatw
    Real pressure_hpa = 0.01 * pressure_pa;

    // Use ERF's saturation vapor pressure function
    Real qsat;
    erf_qsatw(T_wall_use, pressure_hpa, qsat);

    // Scale by wetness: 0.0 = dry, 1.0 = fully saturated
    return wetness * qsat;
}

/**
 * Main wall flux operator - applies wall fluxes to all non-periodic domain faces
 *
 * @param lev AMR level index
 * @param mfs Vector of MultiFabs [cons, xvel, yvel, zvel]
 * @param Tau_lev Vector of stress tensor MultiFabs (indexed by TauType enum)
 * @param Hfx1 Heat flux MultiFab for x-faces (ρ·θ flux)
 * @param Hfx2 Heat flux MultiFab for y-faces (ρ·θ flux)
 * @param Hfx3 Heat flux MultiFab for z-faces (ρ·θ flux)
 * @param Q1fx1 Moisture flux MultiFab for x-faces (ρ·qv flux)
 * @param Q1fx2 Moisture flux MultiFab for y-faces (ρ·qv flux)
 * @param Q1fx3 Moisture flux MultiFab for z-faces (ρ·qv flux)
 * @param geom Geometry object for domain information
 * @param params Wall flux parameters (enable flag, per-face settings)
 * @param domain_bcs_type Velocity boundary condition types for auto momentum detection
 * @param alpha_T Thermal diffusivity for molecular heat diffusion [m²/s]
 * @param alpha_C Moisture diffusivity for molecular moisture diffusion [m²/s]
 *
 * Implementation details:
 * - Uses nodal tile boxes with domain slabbing for face-centered operations
 * - Applies face orientation factor s = ±1 for proper flux direction
 * - Employs 4-point stencil averaging for robust tangential velocity calculation
 * - Supports both momentum drag and free-slip boundary conditions
 * - Handles water-coated walls with saturation vapor pressure calculations
 * - Combines turbulent exchange (C * Ut) with molecular diffusion (α/Δx)
 */
void impose_WallFluxes_all_faces(
    [[maybe_unused]] int lev,
    amrex::Vector<const amrex::MultiFab*> mfs,
    amrex::Vector<std::unique_ptr<amrex::MultiFab>>& Tau_lev,
    amrex::MultiFab* Hfx1, amrex::MultiFab* Hfx2, amrex::MultiFab* Hfx3,
    amrex::MultiFab* Q1fx1, amrex::MultiFab* Q1fx2, amrex::MultiFab* Q1fx3,
    const amrex::Geometry& geom,
    const WallFluxParams& params,
    const amrex::Vector<amrex::BCRec>& domain_bcs_type,
    amrex::Real alpha_T,
    amrex::Real alpha_C)
{
    if (!params.enable) return;
    
    const auto& cons_mf = *mfs[0];
    const auto& xvel_mf = *mfs[1];
    const auto& yvel_mf = *mfs[2];
    const auto& zvel_mf = *mfs[3];
    
    const Box& domain = geom.Domain();
    
    // Precompute domain bounds for GPU kernels (Box methods aren't device functions)
    const int i_lo = domain.smallEnd(0);
    const int i_hi = domain.bigEnd(0) + 1;
    const int j_lo = domain.smallEnd(1);
    const int j_hi = domain.bigEnd(1) + 1;
    const int k_lo = domain.smallEnd(2);
    const int k_hi = domain.bigEnd(2) + 1;

    // Get grid spacing for molecular diffusion
    const Real* dx_ptr = geom.CellSize();
    const Real dx = dx_ptr[0];
    const Real dy = dx_ptr[1];
    const Real dz = dx_ptr[2];

    // Physical z-origin for two_layer sidewall pattern
    const Real prob_lo_z = geom.ProbLo(2);

    // Check which momentum drag to apply per face based on BC types and face settings
    bool apply_momentum_drag[6];
    for (int face = 0; face < 6; ++face) {
        apply_momentum_drag[face] = should_apply_momentum_drag(
            face, params.face[face], domain_bcs_type, 1); // use xvel index for BC check
    }
    
    // Pre-compute has_qv flag for device kernels (moisture transport enabled?)
    bool has_qv = (cons_mf.nComp() > RhoQ1_comp);
    const bool use_pafnn = params.use_pafnn;
    
    // Capture boolean flags for Array4 nullptr checks (Array4 has no boolean conversion)
    const bool do_hfx1 = (Hfx1 != nullptr);
    const bool do_hfx2 = (Hfx2 != nullptr);
    const bool do_hfx3 = (Hfx3 != nullptr);
    const bool do_qfx1 = (Q1fx1 != nullptr);
    const bool do_qfx2 = (Q1fx2 != nullptr);
    const bool do_qfx3 = (Q1fx3 != nullptr);
    
    // Size-aware Tau indexing to avoid UB on short vectors
    const auto nTau = Tau_lev.size();
    const bool have_tau12 = (nTau > TauType::tau12) && (Tau_lev[TauType::tau12] != nullptr);
    const bool have_tau13 = (nTau > TauType::tau13) && (Tau_lev[TauType::tau13] != nullptr);
    const bool have_tau21 = (nTau > TauType::tau21) && (Tau_lev[TauType::tau21] != nullptr);
    const bool have_tau23 = (nTau > TauType::tau23) && (Tau_lev[TauType::tau23] != nullptr);
    const bool have_tau31 = (nTau > TauType::tau31) && (Tau_lev[TauType::tau31] != nullptr);
    const bool have_tau32 = (nTau > TauType::tau32) && (Tau_lev[TauType::tau32] != nullptr);
    
    for (MFIter mfi(cons_mf); mfi.isValid(); ++mfi) {
        
        const auto& cons_arr = cons_mf.const_array(mfi);
        const auto& xvel_arr = xvel_mf.const_array(mfi);
        const auto& yvel_arr = yvel_mf.const_array(mfi);
        const auto& zvel_arr = zvel_mf.const_array(mfi);
        
        // Get flux arrays using captured boolean flags
        Array4<Real> hfx1_arr, hfx2_arr, hfx3_arr;
        Array4<Real> q1fx1_arr, q1fx2_arr, q1fx3_arr;
        
        if (do_hfx1) hfx1_arr = Hfx1->array(mfi);
        if (do_hfx2) hfx2_arr = Hfx2->array(mfi);
        if (do_hfx3) hfx3_arr = Hfx3->array(mfi);
        if (do_qfx1) q1fx1_arr = Q1fx1->array(mfi);
        if (do_qfx2) q1fx2_arr = Q1fx2->array(mfi);
        if (do_qfx3) q1fx3_arr = Q1fx3->array(mfi);
        
        // Get stress arrays using size-safe boolean flags
        Array4<Real> tau12_arr, tau13_arr, tau21_arr, tau23_arr, tau31_arr, tau32_arr;
        
        if (have_tau12) tau12_arr = Tau_lev[TauType::tau12]->array(mfi);
        if (have_tau13) tau13_arr = Tau_lev[TauType::tau13]->array(mfi);
        if (have_tau21) tau21_arr = Tau_lev[TauType::tau21]->array(mfi);
        if (have_tau23) tau23_arr = Tau_lev[TauType::tau23]->array(mfi);
        if (have_tau31) tau31_arr = Tau_lev[TauType::tau31]->array(mfi);
        if (have_tau32) tau32_arr = Tau_lev[TauType::tau32]->array(mfi);
        
        // X-faces (0=xlo, 1=xhi) - process only if domain is not periodic in x
        if (!geom.isPeriodic(0)) {
            auto tbx = mfi.nodaltilebox(0);
            
            // X-lo face (face = 0) - left boundary wall
            if (tbx.smallEnd(0) == i_lo) {
                auto slab = tbx; slab.setSmall(0, i_lo); slab.setBig(0, i_lo);
                const WallFaceParams& face_params = params.face[0];
                bool apply_mom_drag = apply_momentum_drag[0];

                ParallelFor(slab, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    // Face orientation factor: +1 for lo faces, -1 for hi faces
                    const Real s = Real(1.0);

                    // Interior cell index and bounds clamping (avoid Box method calls)
                    const int ii = i_lo;
                    const int jj = amrex::max(j_lo, amrex::min(j, j_hi-1));
                    const int kk = amrex::max(k_lo, amrex::min(k, k_hi-1));
                    const int ii_p = amrex::min(ii+1, i_hi-1);
                    const int jj_p = amrex::min(jj+1, j_hi-1);
                    const int kk_p = amrex::min(kk+1, k_hi-1);

                    // Read state from interior cell
                    Real rho = cons_arr(ii, jj, kk, Rho_comp);
                    Real rhotheta = cons_arr(ii, jj, kk, RhoTheta_comp);
                    Real rhoqv = has_qv ? cons_arr(ii, jj, kk, RhoQ1_comp) : Real(0.0);
                    Real qv = rhoqv / rho;
                    Real theta = rhotheta / rho;

                    // 4-point average for tangential velocities on the x-face (inward bias for x-lo)
                    Real vt = Real(0.25) * ( yvel_arr(ii,   jj,   kk) + yvel_arr(ii,   jj_p, kk) +
                                             yvel_arr(ii_p, jj,   kk) + yvel_arr(ii_p, jj_p, kk) );
                    Real wt = Real(0.25) * ( zvel_arr(ii,   jj,   kk) + zvel_arr(ii,   jj,   kk_p) +
                                             zvel_arr(ii_p, jj,   kk) + zvel_arr(ii_p, jj,   kk_p) );
                    Real Ut = std::sqrt(vt*vt + wt*wt);

                    // Determine effective T_wall and wetness (two_layer overrides face values)
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
                    }

                    // Shared thermodynamics: compute once for both heat and moisture
                    Real p_pa  = getPgivenRTh(rhotheta, qv);
                    Real exner = getExnergivenP(p_pa, R_d / Cp_d);
                    Real theta_wall = T_wall_eff / exner;  // Convert T to θ

                    // Wall-normal velocity for PAFNN
                    Real un_pafnn = xvel_arr(i, j, k);

                    if (use_pafnn) {
                        Real T_air = theta * exner;
                        Real DT_pafnn = T_wall_eff - T_air;
                        Real q_wall_pafnn = compute_qsat_wall(T_wall_eff, p_pa, wetness_eff);
                        Real DQ_pafnn = q_wall_pafnn - qv;
                        Real h_pafnn = Real(0.5) * dx;

                        amrex::Real tau_kin, H_kin, E_kin;
                        PAFNN::pafnn_sw_fluxes(Ut, un_pafnn, DT_pafnn, DQ_pafnn, h_pafnn,
                                               tau_kin, H_kin, E_kin);

                        if (apply_mom_drag && Ut > Real(1e-10)) {
                            Real tau_rho = s * rho * tau_kin;
                            if (have_tau21) tau21_arr(i, j, k) = -tau_rho * vt / Ut;
                            if (have_tau31) tau31_arr(i, j, k) = -tau_rho * wt / Ut;
                        } else {
                            if (have_tau21) tau21_arr(i, j, k) = Real(0.0);
                            if (have_tau31) tau31_arr(i, j, k) = Real(0.0);
                        }
                        if (do_hfx1) hfx1_arr(i, j, k) = s * rho * H_kin / exner;
                        if (do_qfx1) q1fx1_arr(i, j, k) = s * rho * E_kin;
                    } else {


                    // Momentum fluxes (tangential stresses)
                    if (apply_mom_drag && Ut > Real(1e-10)) {
                        Real tau_base = s * rho * face_params.Cm * Ut;
                        if (have_tau21) tau21_arr(i, j, k) = -tau_base * vt; // τ_yx
                        if (have_tau31) tau31_arr(i, j, k) = -tau_base * wt; // τ_zx
                    } else {
                        if (have_tau21) tau21_arr(i, j, k) = Real(0.0);
                        if (have_tau31) tau31_arr(i, j, k) = Real(0.0);
                    }

                    // Scalar fluxes (heat and moisture)
                    // Combined turbulent + molecular: Flux = s * ρ * (Δφ) * (C*Ut + α/Δx)
                    if (do_hfx1) {
                        Real turb_coeff = face_params.Ch * Ut;
                        Real molec_coeff = alpha_T / (Real(0.5) * dx); // wall->first cell-center distance is dx/2
                        Real H = s * rho * (theta_wall - theta) * (turb_coeff + molec_coeff);
                        hfx1_arr(i, j, k) = H;
                    }
                    if (do_qfx1) {
                        Real q_wall = compute_qsat_wall(T_wall_eff, p_pa, wetness_eff);
                        Real turb_coeff = face_params.Cq * Ut;
                        Real molec_coeff = alpha_C / (Real(0.5) * dx); // wall->first cell-center distance is dx/2
                        Real Q = s * rho * (q_wall - qv) * (turb_coeff + molec_coeff);
                        q1fx1_arr(i, j, k) = Q;
                    }
                    } // end else (bulk coefficients)
                });
            }

            // X-hi face (face = 1) - right boundary wall
            if (tbx.bigEnd(0) == i_hi) {
                auto slab = tbx; slab.setSmall(0, i_hi); slab.setBig(0, i_hi);
                const WallFaceParams& face_params = params.face[1];
                bool apply_mom_drag = apply_momentum_drag[1];

                ParallelFor(slab, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    // Face orientation factor: +1 for lo faces, -1 for hi faces
                    const Real s = Real(-1.0);

                    // Interior cell index and bounds clamping (avoid Box method calls)
                    const int ii = i_hi - 1;
                    const int jj = amrex::max(j_lo, amrex::min(j, j_hi-1));
                    const int kk = amrex::max(k_lo, amrex::min(k, k_hi-1));
                    const int ii_m = amrex::max(ii-1, i_lo);
                    const int jj_p = amrex::min(jj+1, j_hi-1);
                    const int kk_p = amrex::min(kk+1, k_hi-1);

                    // Read state from interior cell
                    Real rho = cons_arr(ii, jj, kk, Rho_comp);
                    Real rhotheta = cons_arr(ii, jj, kk, RhoTheta_comp);
                    Real rhoqv = has_qv ? cons_arr(ii, jj, kk, RhoQ1_comp) : Real(0.0);
                    Real qv = rhoqv / rho;
                    Real theta = rhotheta / rho;

                    // 4-point average for tangential velocities on the x-face
                    Real vt = Real(0.25) * ( yvel_arr(ii,   jj,   kk) + yvel_arr(ii,   jj_p, kk) +
                                             yvel_arr(ii_m, jj,   kk) + yvel_arr(ii_m, jj_p, kk) );
                    Real wt = Real(0.25) * ( zvel_arr(ii,   jj,   kk) + zvel_arr(ii,   jj,   kk_p) +
                                             zvel_arr(ii_m, jj,   kk) + zvel_arr(ii_m, jj,   kk_p) );
                    Real Ut = std::sqrt(vt*vt + wt*wt);

                    // Determine effective T_wall and wetness (two_layer overrides face values)
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
                    }

                    // Shared thermodynamics: compute once for both heat and moisture
                    Real p_pa  = getPgivenRTh(rhotheta, qv);
                    Real exner = getExnergivenP(p_pa, R_d / Cp_d);
                    Real theta_wall = T_wall_eff / exner;  // Convert T to θ

                    // Wall-normal velocity for PAFNN
                    Real un_pafnn = xvel_arr(i, j, k);

                    if (use_pafnn) {
                        Real T_air = theta * exner;
                        Real DT_pafnn = T_wall_eff - T_air;
                        Real q_wall_pafnn = compute_qsat_wall(T_wall_eff, p_pa, wetness_eff);
                        Real DQ_pafnn = q_wall_pafnn - qv;
                        Real h_pafnn = Real(0.5) * dx;

                        amrex::Real tau_kin, H_kin, E_kin;
                        PAFNN::pafnn_sw_fluxes(Ut, un_pafnn, DT_pafnn, DQ_pafnn, h_pafnn,
                                               tau_kin, H_kin, E_kin);

                        if (apply_mom_drag && Ut > Real(1e-10)) {
                            Real tau_rho = s * rho * tau_kin;
                            if (have_tau21) tau21_arr(i, j, k) = -tau_rho * vt / Ut;
                            if (have_tau31) tau31_arr(i, j, k) = -tau_rho * wt / Ut;
                        } else {
                            if (have_tau21) tau21_arr(i, j, k) = Real(0.0);
                            if (have_tau31) tau31_arr(i, j, k) = Real(0.0);
                        }
                        if (do_hfx1) hfx1_arr(i, j, k) = s * rho * H_kin / exner;
                        if (do_qfx1) q1fx1_arr(i, j, k) = s * rho * E_kin;
                    } else {


                    // Momentum fluxes (tangential stresses)
                    if (apply_mom_drag && Ut > Real(1e-10)) {
                        Real tau_base = s * rho * face_params.Cm * Ut;
                        if (have_tau21) tau21_arr(i, j, k) = -tau_base * vt; // τ_yx
                        if (have_tau31) tau31_arr(i, j, k) = -tau_base * wt; // τ_zx
                    } else {
                        if (have_tau21) tau21_arr(i, j, k) = Real(0.0);
                        if (have_tau31) tau31_arr(i, j, k) = Real(0.0);
                    }

                    // Scalar fluxes (heat and moisture)
                    // Combined turbulent + molecular: Flux = s * ρ * (Δφ) * (C*Ut + α/Δx)
                    if (do_hfx1) {
                        Real turb_coeff = face_params.Ch * Ut;
                        Real molec_coeff = alpha_T / (Real(0.5) * dx); // wall->first cell-center distance is dx/2
                        Real H = s * rho * (theta_wall - theta) * (turb_coeff + molec_coeff);
                        hfx1_arr(i, j, k) = H;
                    }
                    if (do_qfx1) {
                        Real q_wall = compute_qsat_wall(T_wall_eff, p_pa, wetness_eff);
                        Real turb_coeff = face_params.Cq * Ut;
                        Real molec_coeff = alpha_C / (Real(0.5) * dx); // wall->first cell-center distance is dx/2
                        Real Q = s * rho * (q_wall - qv) * (turb_coeff + molec_coeff);
                        q1fx1_arr(i, j, k) = Q;
                    }
                    } // end else (bulk coefficients)
                });
            }
        }

        // Y-faces (2=ylo, 3=yhi) - process only if domain is not periodic in y
        if (!geom.isPeriodic(1)) {
            auto tby = mfi.nodaltilebox(1);
            
                    // Y-lo face (face = 2) - front/south boundary wall
            if (tby.smallEnd(1) == j_lo) {
                auto slab = tby; slab.setSmall(1, j_lo); slab.setBig(1, j_lo);
                const WallFaceParams& face_params = params.face[2];
                bool apply_mom_drag = apply_momentum_drag[2];
                
                ParallelFor(slab, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    // Face orientation factor: +1 for lo faces, -1 for hi faces
                    const Real s = Real(1.0);
                    
                    // Interior cell index and bounds clamping (avoid Box method calls)
                    const int jj = j_lo;
                    const int ii = amrex::max(i_lo, amrex::min(i, i_hi-1));
                    const int kk = amrex::max(k_lo, amrex::min(k, k_hi-1));
                    const int ii_p = amrex::min(ii+1, i_hi-1);
                    const int kk_p = amrex::min(kk+1, k_hi-1);
                    const int jj_p = amrex::min(jj+1, j_hi-1);  // Use inward bias for y-lo
                    
                    // Read state from interior cell
                    Real rho = cons_arr(ii, jj, kk, Rho_comp);
                    Real rhotheta = cons_arr(ii, jj, kk, RhoTheta_comp);
                    Real rhoqv = has_qv ? cons_arr(ii, jj, kk, RhoQ1_comp) : Real(0.0);
                    Real qv = rhoqv / rho;
                    Real theta = rhotheta / rho;
                    
                    
                    // 4-point average for tangential velocities on the y-face (inward bias for y-lo)
                    Real ut = Real(0.25) * ( xvel_arr(ii,   jj,   kk) + xvel_arr(ii_p, jj,   kk) +
                                             xvel_arr(ii,   jj_p, kk) + xvel_arr(ii_p, jj_p, kk) );
                    Real wt = Real(0.25) * ( zvel_arr(ii,   jj,   kk) + zvel_arr(ii,   jj,   kk_p) +
                                             zvel_arr(ii,   jj_p, kk) + zvel_arr(ii,   jj_p, kk_p) );
                    Real Ut = std::sqrt(ut*ut + wt*wt);

                    // Determine effective T_wall and wetness (two_layer overrides face values)
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
                    }

                    // Shared thermodynamics: compute once for both heat and moisture
                    Real p_pa  = getPgivenRTh(rhotheta, qv);
                    Real exner = getExnergivenP(p_pa, R_d / Cp_d);
                    Real theta_wall = T_wall_eff / exner;  // Convert T to θ

                    // Wall-normal velocity for PAFNN
                    Real un_pafnn = yvel_arr(i, j, k);

                    if (use_pafnn) {
                        Real T_air = theta * exner;
                        Real DT_pafnn = T_wall_eff - T_air;
                        Real q_wall_pafnn = compute_qsat_wall(T_wall_eff, p_pa, wetness_eff);
                        Real DQ_pafnn = q_wall_pafnn - qv;
                        Real h_pafnn = Real(0.5) * dy;

                        amrex::Real tau_kin, H_kin, E_kin;
                        PAFNN::pafnn_sw_fluxes(Ut, un_pafnn, DT_pafnn, DQ_pafnn, h_pafnn,
                                               tau_kin, H_kin, E_kin);

                        if (apply_mom_drag && Ut > Real(1e-10)) {
                            Real tau_rho = s * rho * tau_kin;
                            if (have_tau12) tau12_arr(i, j, k) = -tau_rho * ut / Ut;
                            if (have_tau32) tau32_arr(i, j, k) = -tau_rho * wt / Ut;
                        } else {
                            if (have_tau12) tau12_arr(i, j, k) = Real(0.0);
                            if (have_tau32) tau32_arr(i, j, k) = Real(0.0);
                        }
                        if (do_hfx2) hfx2_arr(i, j, k) = s * rho * H_kin / exner;
                        if (do_qfx2) q1fx2_arr(i, j, k) = s * rho * E_kin;
                    } else {


                    // Momentum fluxes (tangential stresses)
                    if (apply_mom_drag && Ut > Real(1e-10)) {
                        Real tau_base = s * rho * face_params.Cm * Ut;
                        if (have_tau12) tau12_arr(i, j, k) = -tau_base * ut; // τ_xy
                        if (have_tau32) tau32_arr(i, j, k) = -tau_base * wt; // τ_zy
                    } else {
                        if (have_tau12) tau12_arr(i, j, k) = Real(0.0);
                        if (have_tau32) tau32_arr(i, j, k) = Real(0.0);
                    }

                    // Scalar fluxes (heat and moisture)
                    // Combined turbulent + molecular: Flux = s * ρ * (Δφ) * (C*Ut + α/Δy)
                    if (do_hfx2) {
                        Real turb_coeff = face_params.Ch * Ut;
                        Real molec_coeff = alpha_T / (Real(0.5) * dy); // wall->first cell-center distance is dy/2
                        Real H = s * rho * (theta_wall - theta) * (turb_coeff + molec_coeff);
                        hfx2_arr(i, j, k) = H;
                    }
                    if (do_qfx2) {
                        Real q_wall = compute_qsat_wall(T_wall_eff, p_pa, wetness_eff);
                        Real turb_coeff = face_params.Cq * Ut;
                        Real molec_coeff = alpha_C / (Real(0.5) * dy); // wall->first cell-center distance is dy/2
                        Real Q = s * rho * (q_wall - qv) * (turb_coeff + molec_coeff);
                        q1fx2_arr(i, j, k) = Q;
                    }
                    } // end else (bulk coefficients)
                });
            }

            // Y-hi face (face = 3) - back/north boundary wall
            if (tby.bigEnd(1) == j_hi) {
                auto slab = tby; slab.setSmall(1, j_hi); slab.setBig(1, j_hi);
                const WallFaceParams& face_params = params.face[3];
                bool apply_mom_drag = apply_momentum_drag[3];
                
                ParallelFor(slab, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    // Face orientation factor: +1 for lo faces, -1 for hi faces
                    const Real s = Real(-1.0);
                    
                    // Interior cell index and bounds clamping (avoid Box method calls)
                    const int jj = j_hi - 1;
                    const int ii = amrex::max(i_lo, amrex::min(i, i_hi-1));
                    const int kk = amrex::max(k_lo, amrex::min(k, k_hi-1));
                    const int jj_m = amrex::max(jj-1, j_lo);
                    const int ii_p = amrex::min(ii+1, i_hi-1);
                    const int kk_p = amrex::min(kk+1, k_hi-1);
                    
                    // Read state from interior cell
                    Real rho = cons_arr(ii, jj, kk, Rho_comp);
                    Real rhotheta = cons_arr(ii, jj, kk, RhoTheta_comp);
                    Real rhoqv = has_qv ? cons_arr(ii, jj, kk, RhoQ1_comp) : Real(0.0);
                    Real qv = rhoqv / rho;
                    Real theta = rhotheta / rho;
                    
                    
                    // 4-point average for tangential velocities on the y-face
                    Real ut = Real(0.25) * ( xvel_arr(ii,   jj,   kk) + xvel_arr(ii_p, jj,   kk) +
                                             xvel_arr(ii,   jj_m, kk) + xvel_arr(ii_p, jj_m, kk) );
                    Real wt = Real(0.25) * ( zvel_arr(ii,   jj,   kk) + zvel_arr(ii,   jj,   kk_p) +
                                             zvel_arr(ii,   jj_m, kk) + zvel_arr(ii,   jj_m, kk_p) );
                    Real Ut = std::sqrt(ut*ut + wt*wt);

                    // Determine effective T_wall and wetness (two_layer overrides face values)
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
                    }

                    // Shared thermodynamics: compute once for both heat and moisture
                    Real p_pa  = getPgivenRTh(rhotheta, qv);
                    Real exner = getExnergivenP(p_pa, R_d / Cp_d);
                    Real theta_wall = T_wall_eff / exner;  // Convert T to θ

                    // Wall-normal velocity for PAFNN
                    Real un_pafnn = yvel_arr(i, j, k);

                    if (use_pafnn) {
                        Real T_air = theta * exner;
                        Real DT_pafnn = T_wall_eff - T_air;
                        Real q_wall_pafnn = compute_qsat_wall(T_wall_eff, p_pa, wetness_eff);
                        Real DQ_pafnn = q_wall_pafnn - qv;
                        Real h_pafnn = Real(0.5) * dy;

                        amrex::Real tau_kin, H_kin, E_kin;
                        PAFNN::pafnn_sw_fluxes(Ut, un_pafnn, DT_pafnn, DQ_pafnn, h_pafnn,
                                               tau_kin, H_kin, E_kin);

                        if (apply_mom_drag && Ut > Real(1e-10)) {
                            Real tau_rho = s * rho * tau_kin;
                            if (have_tau12) tau12_arr(i, j, k) = -tau_rho * ut / Ut;
                            if (have_tau32) tau32_arr(i, j, k) = -tau_rho * wt / Ut;
                        } else {
                            if (have_tau12) tau12_arr(i, j, k) = Real(0.0);
                            if (have_tau32) tau32_arr(i, j, k) = Real(0.0);
                        }
                        if (do_hfx2) hfx2_arr(i, j, k) = s * rho * H_kin / exner;
                        if (do_qfx2) q1fx2_arr(i, j, k) = s * rho * E_kin;
                    } else {


                    // Momentum fluxes (tangential stresses)
                    if (apply_mom_drag && Ut > Real(1e-10)) {
                        Real tau_base = s * rho * face_params.Cm * Ut;
                        if (have_tau12) tau12_arr(i, j, k) = -tau_base * ut; // τ_xy
                        if (have_tau32) tau32_arr(i, j, k) = -tau_base * wt; // τ_zy
                    } else {
                        if (have_tau12) tau12_arr(i, j, k) = Real(0.0);
                        if (have_tau32) tau32_arr(i, j, k) = Real(0.0);
                    }

                    // Scalar fluxes (heat and moisture)
                    // Combined turbulent + molecular: Flux = s * ρ * (Δφ) * (C*Ut + α/Δy)
                    if (do_hfx2) {
                        Real turb_coeff = face_params.Ch * Ut;
                        Real molec_coeff = alpha_T / (Real(0.5) * dy); // wall->first cell-center distance is dy/2
                        Real H = s * rho * (theta_wall - theta) * (turb_coeff + molec_coeff);
                        hfx2_arr(i, j, k) = H;
                    }
                    if (do_qfx2) {
                        Real q_wall = compute_qsat_wall(T_wall_eff, p_pa, wetness_eff);
                        Real turb_coeff = face_params.Cq * Ut;
                        Real molec_coeff = alpha_C / (Real(0.5) * dy); // wall->first cell-center distance is dy/2
                        Real Q = s * rho * (q_wall - qv) * (turb_coeff + molec_coeff);
                        q1fx2_arr(i, j, k) = Q;
                    }
                    } // end else (bulk coefficients)
                });
            }
        }
        
        // Z-faces (4=zlo, 5=zhi) - process only if domain is not periodic in z
        if (!geom.isPeriodic(2)) {
            auto tbz = mfi.nodaltilebox(2);
            
            // Z-lo face (face = 4) - bottom boundary wall
            if (tbz.smallEnd(2) == k_lo) {
                auto slab = tbz; slab.setSmall(2, k_lo); slab.setBig(2, k_lo);
                const WallFaceParams& face_params = params.face[4];
                bool apply_mom_drag = apply_momentum_drag[4];
                
                ParallelFor(slab, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    // Face orientation factor: +1 for lo faces, -1 for hi faces
                    const Real s = Real(1.0);
                    
                    // Interior cell index and bounds clamping (avoid Box method calls)
                    const int kk = k_lo;
                    const int ii = amrex::max(i_lo, amrex::min(i, i_hi-1));
                    const int jj = amrex::max(j_lo, amrex::min(j, j_hi-1));
                    const int ii_p = amrex::min(ii+1, i_hi-1);
                    const int jj_p = amrex::min(jj+1, j_hi-1);
                    const int kk_p = amrex::min(kk+1, k_hi-1);  // Use inward bias for z-lo
                    
                    // Read state from interior cell
                    Real rho = cons_arr(ii, jj, kk, Rho_comp);
                    Real rhotheta = cons_arr(ii, jj, kk, RhoTheta_comp);
                    Real rhoqv = has_qv ? cons_arr(ii, jj, kk, RhoQ1_comp) : Real(0.0);
                    Real qv = rhoqv / rho;
                    Real theta = rhotheta / rho;
                    
                    
                    // 4-point average for tangential velocities on the z-face (inward bias for z-lo)
                    Real ut = Real(0.25) * ( xvel_arr(ii,   jj,   kk) + xvel_arr(ii_p, jj,   kk) +
                                             xvel_arr(ii,   jj,   kk_p) + xvel_arr(ii_p, jj,   kk_p) );
                    Real vt = Real(0.25) * ( yvel_arr(ii,   jj,   kk) + yvel_arr(ii,   jj_p, kk) +
                                             yvel_arr(ii,   jj,   kk_p) + yvel_arr(ii,   jj_p, kk_p) );
                    Real Ut = std::sqrt(ut*ut + vt*vt);
                    
                    // Shared thermodynamics: compute once for both heat and moisture
                    Real p_pa  = getPgivenRTh(rhotheta, qv);
                    Real exner = getExnergivenP(p_pa, R_d / Cp_d);
                    Real theta_wall = face_params.T_wall / exner;  // Convert T to θ

                    // Wall-normal velocity for PAFNN
                    Real un_pafnn = zvel_arr(i, j, k);

                    if (use_pafnn) {
                        Real T_air = theta * exner;
                        Real DT_pafnn = T_air < face_params.T_wall ? face_params.T_wall - T_air : T_air - face_params.T_wall;
                        Real q_wall_pafnn = compute_qsat_wall(face_params.T_wall, p_pa, face_params.wetness);
                        Real DQ_pafnn = qv < q_wall_pafnn ? q_wall_pafnn - qv : qv - q_wall_pafnn;
                        Real h_pafnn = Real(0.5) * dz;

                        DT_pafnn = amrex::max(DT_pafnn, Real(1e-5));
                        DQ_pafnn = amrex::max(DQ_pafnn, Real(1e-12));

                        amrex::Real tau_kin, H_kin, E_kin;
                        PAFNN::pafnn_bt_fluxes(Ut, un_pafnn, DT_pafnn, DQ_pafnn, h_pafnn,
                                               tau_kin, H_kin, E_kin);

                        Real sign_DT = (face_params.T_wall >= T_air) ? Real(1.0) : Real(-1.0);
                        Real sign_DQ = (q_wall_pafnn >= qv) ? Real(1.0) : Real(-1.0);

                        if (apply_mom_drag && Ut > Real(1e-10)) {
                            Real tau_rho = s * rho * tau_kin;
                            if (have_tau13) tau13_arr(i, j, k) = -tau_rho * ut / Ut;
                            if (have_tau23) tau23_arr(i, j, k) = -tau_rho * vt / Ut;
                        } else {
                            if (have_tau13) tau13_arr(i, j, k) = Real(0.0);
                            if (have_tau23) tau23_arr(i, j, k) = Real(0.0);
                        }
                        if (do_hfx3) hfx3_arr(i, j, k) = s * rho * sign_DT * H_kin / exner;
                        if (do_qfx3) q1fx3_arr(i, j, k) = s * rho * sign_DQ * E_kin;
                    } else {

                    
                    // Momentum fluxes (tangential stresses)
                    if (apply_mom_drag && Ut > Real(1e-10)) {
                        Real tau_base = s * rho * face_params.Cm * Ut;
                        if (have_tau13) tau13_arr(i, j, k) = -tau_base * ut; // τ_xz
                        if (have_tau23) tau23_arr(i, j, k) = -tau_base * vt; // τ_yz
                    } else {
                        if (have_tau13) tau13_arr(i, j, k) = Real(0.0);
                        if (have_tau23) tau23_arr(i, j, k) = Real(0.0);
                    }
                    
                    // Scalar fluxes (heat and moisture)
                    // Combined turbulent + molecular: Flux = s * ρ * (Δφ) * (C*Ut + α/Δz)
                    if (do_hfx3) {
                        Real turb_coeff = face_params.Ch * Ut;
                        Real molec_coeff = alpha_T / (Real(0.5) * dz); // wall->first cell-center distance is dz/2
                        Real H = s * rho * (theta_wall - theta) * (turb_coeff + molec_coeff);
                        hfx3_arr(i, j, k) = H;
                    }
                    if (do_qfx3) {
                        Real q_wall = compute_qsat_wall(face_params.T_wall, p_pa, face_params.wetness);
                        Real turb_coeff = face_params.Cq * Ut;
                        Real molec_coeff = alpha_C / (Real(0.5) * dz); // wall->first cell-center distance is dz/2
                        Real Q = s * rho * (q_wall - qv) * (turb_coeff + molec_coeff);
                        q1fx3_arr(i, j, k) = Q;
                    }
                    } // end else (bulk coefficients)
                });
            }
            
            // Z-hi face (face = 5) - top boundary wall
            if (tbz.bigEnd(2) == k_hi) {
                auto slab = tbz; slab.setSmall(2, k_hi); slab.setBig(2, k_hi);
                const WallFaceParams& face_params = params.face[5];
                bool apply_mom_drag = apply_momentum_drag[5];
                
                ParallelFor(slab, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    // Face orientation factor: +1 for lo faces, -1 for hi faces
                    const Real s = Real(-1.0);
                    
                    // Interior cell index and bounds clamping (avoid Box method calls)
                    const int kk = k_hi - 1;
                    const int ii = amrex::max(i_lo, amrex::min(i, i_hi-1));
                    const int jj = amrex::max(j_lo, amrex::min(j, j_hi-1));
                    const int kk_m = amrex::max(kk-1, k_lo);
                    const int ii_p = amrex::min(ii+1, i_hi-1);
                    const int jj_p = amrex::min(jj+1, j_hi-1);
                    
                    // Read state from interior cell
                    Real rho = cons_arr(ii, jj, kk, Rho_comp);
                    Real rhotheta = cons_arr(ii, jj, kk, RhoTheta_comp);
                    Real rhoqv = has_qv ? cons_arr(ii, jj, kk, RhoQ1_comp) : Real(0.0);
                    Real qv = rhoqv / rho;
                    Real theta = rhotheta / rho;
                    
                    
                    // 4-point average for tangential velocities on the z-face
                    Real ut = Real(0.25) * ( xvel_arr(ii,   jj,   kk) + xvel_arr(ii_p, jj,   kk) +
                                             xvel_arr(ii,   jj,   kk_m) + xvel_arr(ii_p, jj,   kk_m) );
                    Real vt = Real(0.25) * ( yvel_arr(ii,   jj,   kk) + yvel_arr(ii,   jj_p, kk) +
                                             yvel_arr(ii,   jj,   kk_m) + yvel_arr(ii,   jj_p, kk_m) );
                    Real Ut = std::sqrt(ut*ut + vt*vt);
                    
                    // Shared thermodynamics: compute once for both heat and moisture
                    Real p_pa  = getPgivenRTh(rhotheta, qv);
                    Real exner = getExnergivenP(p_pa, R_d / Cp_d);
                    Real theta_wall = face_params.T_wall / exner;  // Convert T to θ

                    // Wall-normal velocity for PAFNN
                    Real un_pafnn = zvel_arr(i, j, k);

                    if (use_pafnn) {
                        Real T_air = theta * exner;
                        Real DT_pafnn = T_air < face_params.T_wall ? face_params.T_wall - T_air : T_air - face_params.T_wall;
                        Real q_wall_pafnn = compute_qsat_wall(face_params.T_wall, p_pa, face_params.wetness);
                        Real DQ_pafnn = qv < q_wall_pafnn ? q_wall_pafnn - qv : qv - q_wall_pafnn;
                        Real h_pafnn = Real(0.5) * dz;

                        DT_pafnn = amrex::max(DT_pafnn, Real(1e-5));
                        DQ_pafnn = amrex::max(DQ_pafnn, Real(1e-12));

                        amrex::Real tau_kin, H_kin, E_kin;
                        PAFNN::pafnn_bt_fluxes(Ut, un_pafnn, DT_pafnn, DQ_pafnn, h_pafnn,
                                               tau_kin, H_kin, E_kin);

                        Real sign_DT = (face_params.T_wall >= T_air) ? Real(1.0) : Real(-1.0);
                        Real sign_DQ = (q_wall_pafnn >= qv) ? Real(1.0) : Real(-1.0);

                        if (apply_mom_drag && Ut > Real(1e-10)) {
                            Real tau_rho = s * rho * tau_kin;
                            if (have_tau13) tau13_arr(i, j, k) = -tau_rho * ut / Ut;
                            if (have_tau23) tau23_arr(i, j, k) = -tau_rho * vt / Ut;
                        } else {
                            if (have_tau13) tau13_arr(i, j, k) = Real(0.0);
                            if (have_tau23) tau23_arr(i, j, k) = Real(0.0);
                        }
                        if (do_hfx3) hfx3_arr(i, j, k) = s * rho * sign_DT * H_kin / exner;
                        if (do_qfx3) q1fx3_arr(i, j, k) = s * rho * sign_DQ * E_kin;
                    } else {

                    
                    // Momentum fluxes (tangential stresses)
                    if (apply_mom_drag && Ut > Real(1e-10)) {
                        Real tau_base = s * rho * face_params.Cm * Ut;
                        if (have_tau13) tau13_arr(i, j, k) = -tau_base * ut; // τ_xz
                        if (have_tau23) tau23_arr(i, j, k) = -tau_base * vt; // τ_yz
                    } else {
                        if (have_tau13) tau13_arr(i, j, k) = Real(0.0);
                        if (have_tau23) tau23_arr(i, j, k) = Real(0.0);
                    }
                    
                    // Scalar fluxes (heat and moisture)
                    // Combined turbulent + molecular: Flux = s * ρ * (Δφ) * (C*Ut + α/Δz)
                    if (do_hfx3) {
                        Real turb_coeff = face_params.Ch * Ut;
                        Real molec_coeff = alpha_T / (Real(0.5) * dz); // wall->first cell-center distance is dz/2
                        Real H = s * rho * (theta_wall - theta) * (turb_coeff + molec_coeff);
                        hfx3_arr(i, j, k) = H;
                    }
                    if (do_qfx3) {
                        Real q_wall = compute_qsat_wall(face_params.T_wall, p_pa, face_params.wetness);
                        Real turb_coeff = face_params.Cq * Ut;
                        Real molec_coeff = alpha_C / (Real(0.5) * dz); // wall->first cell-center distance is dz/2
                        Real Q = s * rho * (q_wall - qv) * (turb_coeff + molec_coeff);
                        q1fx3_arr(i, j, k) = Q;
                    }
                    } // end else (bulk coefficients)
                });
            }
        }
    }
}
