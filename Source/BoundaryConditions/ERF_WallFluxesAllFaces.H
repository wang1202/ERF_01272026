#ifndef ERF_WALLFLUXESALLFACES_H_
#define ERF_WALLFLUXESALLFACES_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_BCRec.H>

/**
 * @file ERF_WallFluxesAllFaces.H
 * @brief Wall flux boundary conditions for all domain faces
 *
 * This module implements wall flux boundary conditions that can be applied
 * to all six faces of a rectangular domain. It supports:
 * - Independent drag/exchange coefficients per face
 * - Free-slip or drag momentum models
 * - Heat and moisture exchange with wall surfaces
 * - Water-coated walls for cloud chamber applications
 *
 * The implementation uses bulk exchange formulas:
 * - Momentum flux: tau = rho * Cm * |U| * U_component
 * - Heat flux: H = rho * cp * Ch * |U| * (T_wall - T_air)
 * - Moisture flux: Q = rho * Cq * |U| * (q_wall - q_air)
 *
 * Note: Wall temperature is specified as thermodynamic temperature T_wall [K],
 * which is converted internally to potential temperature for heat flux calculations.
 */

/**
 * @struct WallFaceParams
 * @brief Parameters for a single wall face
 *
 * Contains all physical parameters needed to compute wall fluxes
 * for one face of the domain boundary.
 */
struct WallFaceParams {
    amrex::Real Cm{1.5e-3};        ///< Momentum drag coefficient (dimensionless)
    amrex::Real Ch{1.5e-3};        ///< Heat exchange coefficient (dimensionless)
    amrex::Real Cq{1.5e-3};        ///< Moisture exchange coefficient (dimensionless)
    amrex::Real T_wall{293.0};     ///< Wall thermodynamic temperature [K]
    amrex::Real wetness{1.0};      ///< Wall wetness (0.0=dry, 1.0=saturated) [dimensionless]

    /// Momentum model for this face
    enum class MomModel {
        Auto,     ///< Automatically detect from boundary conditions
        Drag,     ///< Apply momentum drag (no-slip wall)
        FreeSlip  ///< Free-slip wall (no momentum drag)
    } model{MomModel::Auto};
};

/**
 * @struct WallFluxParams
 * @brief Global wall flux configuration
 *
 * Contains enable flags and parameters for all six domain faces.
 * Face ordering: [xlo, xhi, ylo, yhi, zlo, zhi]
 */
struct WallFluxParams {
    bool enable{false};      ///< Enable wall flux application
    WallFaceParams face[6];  ///< Parameters for each face: xlo,xhi,ylo,yhi,zlo,zhi

    // Two-layer sidewall pattern (applies to xlo, xhi, ylo, yhi only)
    bool two_layer_sidewall{false};                    ///< Use height-dependent T_wall/wetness on side walls
    amrex::Real sidewall_separation_height{0.0};       ///< Height separating upper/lower regions [m]
    amrex::Real T_wall_upper{293.0};                   ///< Wall temperature for upper region [K]
    amrex::Real wetness_upper{1.0};                    ///< Wall wetness for upper region [dimensionless]
    amrex::Real T_wall_lower{293.0};                   ///< Wall temperature for lower region [K]
    amrex::Real wetness_lower{1.0};                    ///< Wall wetness for lower region [dimensionless]
};

/**
 * @brief Main function to impose wall fluxes on all domain faces
 *
 * Applies wall flux boundary conditions to momentum, heat, and moisture
 * on all six faces of the computational domain using bulk exchange formulas
 * combined with molecular diffusion.
 *
 * @param lev           AMR level
 * @param mfs           Vector of MultiFabs: {cons, xvel, yvel, zvel}
 * @param Tau_lev       Vector of stress tensor MultiFabs (tau_ij components)
 * @param Hfx1,Hfx2,Hfx3  Heat flux MultiFabs in x,y,z directions
 * @param Q1fx1,Q1fx2,Q1fx3  Moisture flux MultiFabs in x,y,z directions
 * @param geom          Geometry object for domain information
 * @param params        Wall flux parameters for all faces
 * @param domain_bcs_type  Boundary condition types for momentum detection
 * @param alpha_T       Thermal diffusivity [m^2/s] (0.0 = no molecular diffusion)
 * @param alpha_C       Moisture diffusivity [m^2/s] (0.0 = no molecular diffusion)
 */
void impose_WallFluxes_all_faces(
    int lev,
    amrex::Vector<const amrex::MultiFab*> mfs,
    amrex::Vector<std::unique_ptr<amrex::MultiFab>>& Tau_lev,
    amrex::MultiFab* Hfx1, amrex::MultiFab* Hfx2, amrex::MultiFab* Hfx3,
    amrex::MultiFab* Q1fx1, amrex::MultiFab* Q1fx2, amrex::MultiFab* Q1fx3,
    const amrex::Geometry& geom,
    const WallFluxParams& params,
    const amrex::Vector<amrex::BCRec>& domain_bcs_type,
    amrex::Real alpha_T = 0.0,
    amrex::Real alpha_C = 0.0);

/**
 * @brief Determine if momentum drag should be applied on a face
 *
 * Checks the momentum model setting and boundary conditions to decide
 * whether to apply momentum drag (no-slip) or free-slip conditions.
 *
 * @param face_idx      Face index (0=xlo, 1=xhi, 2=ylo, 3=yhi, 4=zlo, 5=zhi)
 * @param face_params   Parameters for this face
 * @param domain_bcs_type  Boundary condition types for auto-detection
 * @param var_idx       Velocity component index for BC lookup
 * @return true if momentum drag should be applied, false for free-slip
 */
bool should_apply_momentum_drag(
    int face_idx,
    const WallFaceParams& face_params,
    const amrex::Vector<amrex::BCRec>& domain_bcs_type,
    int var_idx);

/**
 * @brief Compute saturation mixing ratio at wall surface
 *
 * Calculates the saturation mixing ratio for moisture exchange
 * at the wall boundary, accounting for wall wetness.
 *
 * @param T_wall        Wall thermodynamic temperature [K]
 * @param pressure      Pressure at wall [Pa]
 * @param wetness       Wall wetness (0.0=dry, 1.0=saturated) [dimensionless]
 * @return Saturation mixing ratio scaled by wetness [kg/kg]
 */
amrex::Real compute_qsat_wall(
    amrex::Real T_wall,
    amrex::Real pressure,
    amrex::Real wetness);

#endif
