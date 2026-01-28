#ifndef ERF_WALLFLUXDIAGNOSTICS_H_
#define ERF_WALLFLUXDIAGNOSTICS_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <string>
#include <fstream>

/**
 * @file ERF_WallFluxDiagnostics.H
 * @brief Diagnostic output for wall fluxes on all domain faces
 *
 * Computes and outputs time series of mean wall fluxes on all six domain faces:
 * - Mean momentum flux (shear stress) [Pa = N/m²]
 * - Mean heat flux [W/m²]
 * - Mean moisture flux [kg/(m²·s)]
 */

/**
 * @struct WallFluxDiagnostics
 * @brief Container for wall flux diagnostic data and output
 */
struct WallFluxDiagnostics {
    bool enabled{false};               ///< Enable wall flux diagnostics
    int output_interval{1};            ///< Output every N timesteps
    std::string output_dir{"."};       ///< Output directory
    std::string filename_prefix{"wallflux"}; ///< Filename prefix

    // Output file streams (one per level)
    std::vector<std::ofstream> output_files;

    // Face names for output headers
    const std::vector<std::string> face_names = {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"};

    /**
     * @brief Initialize diagnostic output files
     * @param nlevels Number of AMR levels
     * @param geom Geometry for each level
     */
    void initialize(int nlevels, const amrex::Vector<amrex::Geometry>& geom);

    /**
     * @brief Compute and output wall flux diagnostics
     * @param level AMR level
     * @param time Current simulation time
     * @param timestep Current timestep number
     * @param xvel x-velocity MultiFab
     * @param yvel y-velocity MultiFab
     * @param zvel z-velocity MultiFab
     * @param cons Conservative variables MultiFab
     * @param Tau_lev Stress tensor components
     * @param Hfx1 Heat flux in x-direction
     * @param Hfx2 Heat flux in y-direction
     * @param Hfx3 Heat flux in z-direction
     * @param Q1fx1 Moisture flux in x-direction (can be nullptr)
     * @param Q1fx2 Moisture flux in y-direction (can be nullptr)
     * @param Q1fx3 Moisture flux in z-direction (can be nullptr)
     * @param geom Geometry object
     */
    void output(int level, amrex::Real time, int timestep,
                const amrex::MultiFab& xvel,
                const amrex::MultiFab& yvel,
                const amrex::MultiFab& zvel,
                const amrex::MultiFab& cons,
                const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& Tau_lev,
                amrex::MultiFab* Hfx1, amrex::MultiFab* Hfx2, amrex::MultiFab* Hfx3,
                amrex::MultiFab* Q1fx1, amrex::MultiFab* Q1fx2, amrex::MultiFab* Q1fx3,
                const amrex::Geometry& geom);

    /**
     * @brief Close all output files
     */
    void finalize();
};

/**
 * @brief Compute mean flux on a domain face
 * @param mf MultiFab containing flux data
 * @param face Face index (0=xlo, 1=xhi, 2=ylo, 3=yhi, 4=zlo, 5=zhi)
 * @param geom Geometry object
 * @return Mean flux value on the face
 */
amrex::Real compute_mean_face_flux(const amrex::MultiFab& mf, int face,
                                    const amrex::Geometry& geom);

/**
 * @brief Convert potential temperature flux to sensible heat flux
 * @param rho_theta_flux Potential temperature flux [kg·K/(m²·s)]
 * @param pressure Pressure [Pa]
 * @param cp Specific heat at constant pressure [J/(kg·K)]
 * @return Sensible heat flux [W/m²]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real convert_theta_flux_to_heat_flux(amrex::Real rho_theta_flux,
                                             amrex::Real pressure,
                                             amrex::Real cp = 1004.0) {
    // Exner function: π = (p/p0)^(R/cp)
    constexpr amrex::Real p0 = 1.0e5;  // Reference pressure [Pa]
    constexpr amrex::Real rdOcp = 0.287 / 1.004;  // R_d/cp
    amrex::Real exner = std::pow(pressure / p0, rdOcp);

    // H[W/m²] = cp * π * (ρθ flux)
    return cp * exner * rho_theta_flux;
}

#endif
