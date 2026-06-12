#include "ERF_SuperDropletsMoist.H"
#include "ERF_MaterialProperties.H"
#include "ERF_SuperDropletPC.H"
#include "ERF_SuperDropletPCDefinitions.H"
#include <AMReX_ParReduce.H>
#include <AMReX_Reduce.H>
#include <iomanip>

#ifdef ERF_USE_PARTICLES

using namespace amrex;
using namespace SDPCDefn;

/*! Compute and write domain-averaged time series of thermodynamic and microphysical properties.
 *  Outputs:
 *  - Temperature (domain mean)
 *  - Relative humidity (domain mean)
 *  - Droplet radius statistics (mean, std, skewness, kurtosis) for droplets > cutoff radius
 *  - Droplet number concentration (only for droplets > cutoff radius)
 *  - Droplet liquid water content (only for droplets > cutoff radius)
 *  - Droplet number concentration (only for droplets <= cutoff radius)
 *  - Super droplet count (computational particles > cutoff radius)
 *  - Super droplet count (computational particles <= cutoff radius)
 *
 *  Performance note: All 9 particle statistics are computed in a single pass over
 *  particles to minimize GPU kernel launches and memory bandwidth usage.
 */
void SuperDropletsMoist::WriteDomainTimeSeries(const Real& a_time,
                                               const int& a_iter)
{
    BL_PROFILE("SuperDropletsMoist::WriteDomainTimeSeries()");

    // Domain-averaged diagnostic: m_mic_fab_vars is indexed [lev][var] after the
    // SDM refactor, and m_geom is the level-0 (full domain) geometry.
    const int lev = 0;

    // Get domain volume
    const Geometry& geom = m_geom;
    const auto dx = geom.CellSizeArray();
    const Real cell_volume = dx[0] * dx[1] * dx[2];
    const BoxArray& ba = m_mic_fab_vars[lev][MicVar_SD::temperature]->boxArray();
    const DistributionMapping& dm = m_mic_fab_vars[lev][MicVar_SD::temperature]->DistributionMap();

    // Count total cells
    Long total_cells = ba.numPts();

    // Compute domain-averaged Eulerian quantities
    IntVect nghost(0, 0, 0);

    auto const& temp_arrs = m_mic_fab_vars[lev][MicVar_SD::temperature]->const_arrays();
    auto const& rh_arrs = m_mic_fab_vars[lev][MicVar_SD::rh]->const_arrays();

    Real temp_sum = ParReduce(TypeList<ReduceOpSum>{},
                              TypeList<Real>{},
                              *m_mic_fab_vars[lev][MicVar_SD::temperature], nghost,
                              [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept -> GpuTuple<Real> {
                                  return { temp_arrs[box_no](i,j,k) };
                              });

    Real rh_sum = ParReduce(TypeList<ReduceOpSum>{},
                            TypeList<Real>{},
                            *m_mic_fab_vars[lev][MicVar_SD::rh], nghost,
                            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept -> GpuTuple<Real> {
                                return { rh_arrs[box_no](i,j,k) };
                            });

    // ParReduce only sums over each rank's local boxes; sum across ranks before
    // dividing by the global cell count (total_cells = ba.numPts()).
    ParallelDescriptor::ReduceRealSum(temp_sum);
    ParallelDescriptor::ReduceRealSum(rh_sum);

    Real temp_mean = temp_sum / total_cells;
    Real rh_mean = rh_sum / total_cells;

    // Compute droplet properties for droplets > cutoff radius
    const auto& species_mat = m_super_droplets->getSpeciesMaterial(Species::Name::H2O);
    const auto rho_w = species_mat.m_density;
    const auto na = m_num_aerosols;
    const auto ns = m_num_species;
    const auto idx_w = m_idx_w;
    const Real r_cutoff = m_domain_ts_radius_cutoff;
    const Real domain_volume = geom.ProbDomain().volume();

    // =========================================================================
    // BATCHED REDUCTION: Compute all 9 particle statistics in a single pass
    // This reduces GPU kernel launches from 9 to 1 and minimizes memory bandwidth
    // =========================================================================

    // Define reduction operations for 9 sums
    ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum,
              ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_op;
    ReduceData<Real, Real, Real, Real, Real, Real, Real, Real, Real> reduce_data(reduce_op);

    // Single pass over all particles computing all statistics
    for (int lev = 0; lev < m_super_droplets->finestLevel()+1; ++lev) {
        for (SuperDropletPC::ParIterType pti(*m_super_droplets, lev); pti.isValid(); ++pti) {
            const auto& ptd = pti.GetParticleTile().getConstParticleTileData();
            const int np = pti.numParticles();

            reduce_op.eval(np, reduce_data,
                [=] AMREX_GPU_DEVICE (int i) -> GpuTuple<Real, Real, Real, Real, Real, Real, Real, Real, Real>
                {
                    // Read particle data once
                    auto ni = ptd.m_runtime_rdata[SuperDropletsRealIdxSoA_RT::multiplicity][i];
                    auto mi = ptd.m_runtime_rdata[ridx_s(idx_w,na,ns)][i];

                    // Compute radius once (avoid redundant cbrt calls)
                    auto ri = std::cbrt( mi / ((4.0/3.0)*PI*rho_w) );

                    // Determine cutoff condition once
                    bool above = (ri > r_cutoff);

                    // Compute powers of radius
                    Real ri2 = ri * ri;
                    Real ri3 = ri2 * ri;
                    Real ri4 = ri2 * ri2;

                    // Return all 9 values in single tuple:
                    // 0: num_droplets (above cutoff)
                    // 1: radius_weighted_sum (above cutoff)
                    // 2: radius2_weighted_sum (above cutoff)
                    // 3: radius3_weighted_sum (above cutoff)
                    // 4: radius4_weighted_sum (above cutoff)
                    // 5: lwc_sum (above cutoff)
                    // 6: num_droplets_below (below cutoff)
                    // 7: num_SD_above (count of super droplets above cutoff)
                    // 8: num_SD_below (count of super droplets below cutoff)
                    return {
                        above ? ni : 0.0,           // num_droplets
                        above ? ni*ri : 0.0,        // radius_weighted_sum
                        above ? ni*ri2 : 0.0,       // radius2_weighted_sum
                        above ? ni*ri3 : 0.0,       // radius3_weighted_sum
                        above ? ni*ri4 : 0.0,       // radius4_weighted_sum
                        above ? ni*mi : 0.0,        // lwc_sum
                        above ? 0.0 : ni,           // num_droplets_below
                        above ? 1.0 : 0.0,          // num_SD_above
                        above ? 0.0 : 1.0           // num_SD_below
                    };
                });
        }
    }

    // Extract results from batched reduction
    auto hv = reduce_data.value(reduce_op);
    Real num_droplets = amrex::get<0>(hv);
    Real radius_weighted_sum = amrex::get<1>(hv);
    Real radius2_weighted_sum = amrex::get<2>(hv);
    Real radius3_weighted_sum = amrex::get<3>(hv);
    Real radius4_weighted_sum = amrex::get<4>(hv);
    Real lwc_sum = amrex::get<5>(hv);
    Real num_droplets_below = amrex::get<6>(hv);
    Real num_SD_above = amrex::get<7>(hv);
    Real num_SD_below = amrex::get<8>(hv);

    // Single MPI reduction for all 9 values (instead of 9 separate calls)
    Real reduce_vals[9] = {num_droplets, radius_weighted_sum, radius2_weighted_sum,
                           radius3_weighted_sum, radius4_weighted_sum, lwc_sum,
                           num_droplets_below, num_SD_above, num_SD_below};
    ParallelDescriptor::ReduceRealSum(reduce_vals, 9);

    // Unpack reduced values
    num_droplets = reduce_vals[0];
    radius_weighted_sum = reduce_vals[1];
    radius2_weighted_sum = reduce_vals[2];
    radius3_weighted_sum = reduce_vals[3];
    radius4_weighted_sum = reduce_vals[4];
    lwc_sum = reduce_vals[5];
    num_droplets_below = reduce_vals[6];
    num_SD_above = reduce_vals[7];
    num_SD_below = reduce_vals[8];

    // Compute domain-averaged quantities
    Real mean_radius = (num_droplets > 0) ? radius_weighted_sum / num_droplets : 0.0;

    // Compute std, skewness, and kurtosis using raw moments
    // Variance = E[r^2] - E[r]^2
    // Skewness = E[(r-mu)^3] / sigma^3 = (E[r^3] - 3*mu*E[r^2] + 2*mu^3) / sigma^3
    // Kurtosis = E[(r-mu)^4] / sigma^4 = (E[r^4] - 4*mu*E[r^3] + 6*mu^2*E[r^2] - 3*mu^4) / sigma^4
    Real std_radius = 0.0;
    Real skewness_radius = 0.0;
    Real kurtosis_radius = 0.0;

    if (num_droplets > 0) {
        Real mu = mean_radius;
        Real mu2 = mu * mu;
        Real mu3 = mu2 * mu;
        Real mu4 = mu3 * mu;

        Real E_r2 = radius2_weighted_sum / num_droplets;
        Real E_r3 = radius3_weighted_sum / num_droplets;
        Real E_r4 = radius4_weighted_sum / num_droplets;

        Real variance = E_r2 - mu2;
        if (variance > 0) {
            std_radius = std::sqrt(variance);
            Real sigma3 = std_radius * std_radius * std_radius;
            Real sigma4 = sigma3 * std_radius;

            // Third central moment: E[(r-mu)^3] = E[r^3] - 3*mu*E[r^2] + 2*mu^3
            Real central_moment_3 = E_r3 - 3.0*mu*E_r2 + 2.0*mu3;
            skewness_radius = central_moment_3 / sigma3;

            // Fourth central moment: E[(r-mu)^4] = E[r^4] - 4*mu*E[r^3] + 6*mu^2*E[r^2] - 3*mu^4
            Real central_moment_4 = E_r4 - 4.0*mu*E_r3 + 6.0*mu2*E_r2 - 3.0*mu4;
            kurtosis_radius = central_moment_4 / sigma4;
        }
    }
    Real number_conc = num_droplets / domain_volume;  // #/m³
    Real lwc = lwc_sum / domain_volume;  // kg/m³
    Real number_conc_below = num_droplets_below / domain_volume;  // #/m³

    // Write to file
    if (ParallelDescriptor::IOProcessor()) {
        std::string filename = m_name + "_domain_timeseries.txt";
        std::ofstream outfile;

        // Write header on first call
        if (a_iter == 0) {
            outfile.open(filename.c_str(), std::ios::out | std::ios::trunc);
            if (!outfile.good()) { amrex::FileOpenFailed(filename); }

            outfile << "# Domain-averaged time series\n";
            outfile << "# Cutoff radius: " << r_cutoff << " m\n";
            outfile << "# Columns:\n";
            outfile << "# 1: Time (s)\n";
            outfile << "# 2: Temperature (K)\n";
            outfile << "# 3: Relative humidity (0-1)\n";
            outfile << "# 4: Mean droplet radius (m) for r > r_cutoff\n";
            outfile << "# 5: Std dev of droplet radius (m) for r > r_cutoff\n";
            outfile << "# 6: Skewness of droplet radius for r > r_cutoff\n";
            outfile << "# 7: Kurtosis of droplet radius for r > r_cutoff\n";
            outfile << "# 8: Droplet number concentration (#/m³) for r > r_cutoff\n";
            outfile << "# 9: Liquid water content (kg/m³) for r > r_cutoff\n";
            outfile << "# 10: Droplet number concentration (#/m³) for r <= r_cutoff\n";
            outfile << "# 11: Super droplet count for r > r_cutoff\n";
            outfile << "# 12: Super droplet count for r <= r_cutoff\n";
            outfile << "# Time    Temperature    RH    MeanRadius    StdRadius    Skewness    Kurtosis    NumConc    LWC    NumConcBelow    NumSD_above    NumSD_below\n";
        } else {
            outfile.open(filename.c_str(), std::ios::out | std::ios::app);
            if (!outfile.good()) { amrex::FileOpenFailed(filename); }
        }

        outfile << std::scientific << std::setprecision(8)
                << a_time << "  "
                << temp_mean << "  "
                << rh_mean << "  "
                << mean_radius << "  "
                << std_radius << "  "
                << skewness_radius << "  "
                << kurtosis_radius << "  "
                << number_conc << "  "
                << lwc << "  "
                << number_conc_below << "  "
                << num_SD_above << "  "
                << num_SD_below << "\n";

        outfile.flush();
        outfile.close();
        if (!outfile.good()) { amrex::Abort("problem writing domain time series file"); }
    }
}

#endif
