#include "ERF_SuperDropletPC.H"

#ifdef ERF_USE_PARTICLES

#include <AMReX_GpuAtomic.H>
#include <AMReX_Random.H>
#include <AMReX_TracerParticle_mod_K.H>
#include <AMReX_ParticleUtil.H>
#include "ERF_Constants.H"
#include "ERF_SDWallLoss.H"

#include <fstream>
#include <iomanip>

using namespace amrex;
using namespace SDPCDefn;

/*! \brief Apply the boundary-surface deposition sink to all chamber boundaries.
 *
 *  For every active super-droplet in a cell adjacent to an absorbing chamber
 *  boundary, the multiplicity is reduced by the total first-order loss across
 *  all boundary faces the cell touches:
 *
 *      k_total(r) = sum_b vd_b(r) * A_b / V_cell
 *      P_loss(r)  = 1 - exp(-k_total(r) * dt)
 *
 *  with boundary-specific deposition velocities (side / top / bottom), an
 *  optional gravitational settling correction on the top/bottom faces, and the
 *  removal applied either deterministically (Option A) or by stochastic thinning
 *  (Option B).  Optionally deactivates super-droplets whose multiplicity decays
 *  below a threshold (their remaining multiplicity is counted as deposited).
 *
 *  See ERF_SDWallLoss.H for the formulation and references.
 */
void SuperDropletPC::WallLoss ( const int        a_lev,
                                const Real       a_dt,
                                const Real       a_time,
                                const int        a_iter,
                                const BCTypeArr& a_bctypes )
{
    if (!m_wall_loss.enable) { return; }

    BL_PROFILE("SuperDropletPC::WallLoss()");

    const auto ctx = buildProcessContext(a_lev);

    // POD copy captured by value into the device kernel.
    const SDWallLossParams wl = m_wall_loss;

    // A boundary face is absorbing only if non-periodic with a wall-type BC.
    auto is_wall_bc = [] (ERF_BC bc) -> bool {
        return (bc == ERF_BC::no_slip_wall) ||
               (bc == ERF_BC::slip_wall)    ||
               (bc == ERF_BC::surface_layer);
    };
    const bool wall_xlo = (!ctx.is_periodic[0]) && is_wall_bc(a_bctypes[Orientation(0,Orientation::low)]);
    const bool wall_xhi = (!ctx.is_periodic[0]) && is_wall_bc(a_bctypes[Orientation(0,Orientation::high)]);
    const bool wall_ylo = (!ctx.is_periodic[1]) && is_wall_bc(a_bctypes[Orientation(1,Orientation::low)]);
    const bool wall_yhi = (!ctx.is_periodic[1]) && is_wall_bc(a_bctypes[Orientation(1,Orientation::high)]);
    const bool wall_zlo = (!ctx.is_periodic[2]) && is_wall_bc(a_bctypes[Orientation(2,Orientation::low)]);  // bottom
    const bool wall_zhi = (!ctx.is_periodic[2]) && is_wall_bc(a_bctypes[Orientation(2,Orientation::high)]); // top

    if (!(wall_xlo || wall_xhi || wall_ylo || wall_yhi || wall_zlo || wall_zhi)) { return; }

    const Box& domain = ctx.domain;
    const int domlo0 = domain.smallEnd(0); const int domhi0 = domain.bigEnd(0);
    const int domlo1 = domain.smallEnd(1); const int domhi1 = domain.bigEnd(1);
    const int domlo2 = domain.smallEnd(2); const int domhi2 = domain.bigEnd(2);

    // Cell face areas (normal to each boundary) and cell volume.
    const Real A_xface = ctx.dx[1] * ctx.dx[2]; // x-normal (side)
    const Real A_yface = ctx.dx[0] * ctx.dx[2]; // y-normal (side)
    const Real A_zface = ctx.dx[0] * ctx.dx[1]; // z-normal (top/bottom)
    const Real V_cell  = ctx.cell_volume;

    const int nbins = (wl.num_bins < 1) ? 1
                    : ((wl.num_bins > SDWL_MAX_BINS) ? SDWL_MAX_BINS : wl.num_bins);

    // Device accumulators:
    //   [0,1,2] multiplicity removed to side / top / bottom
    //   [3,4,5] deposited mass to side / top / bottom
    //   [6 .. 6+nbins-1] multiplicity removed per size bin (all boundaries)
    const int nacc = 6 + nbins;
    Gpu::DeviceVector<Real> accum(nacc, Real(0.0));
    Real* accum_ptr = accum.dataPtr();

    forEachParticleTile(a_lev, ctx,
        [&](ParIterType& pti, int grid, ParticleType* p_pbox,
            const SDProcess::ParticlePointers& ptrs,
            const SDProcess::ProcessContext& ctx)
    {
        amrex::ignore_unused(pti, grid);

        ParallelForRNG(ptrs.num_particles,
            [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine)
        {
            ParticleType& p = p_pbox[i];
            if (p.id() <= 0)             { return; }
            if (ptrs.active_ptr[i] == 0) { return; }

            const IntVect iv = getParticleCell(p, ctx.plo, ctx.dxi, ctx.domain);

            // Side-wall face area adjacent to this cell (x and y faces).
            Real A_side = Real(0.0);
            if (wall_xlo && iv[0] == domlo0) { A_side += A_xface; }
            if (wall_xhi && iv[0] == domhi0) { A_side += A_xface; }
            if (wall_ylo && iv[1] == domlo1) { A_side += A_yface; }
            if (wall_yhi && iv[1] == domhi1) { A_side += A_yface; }
            // Top / bottom face area adjacent to this cell.
            const Real A_top = (wall_zhi && iv[2] == domhi2) ? A_zface : Real(0.0);
            const Real A_bot = (wall_zlo && iv[2] == domlo2) ? A_zface : Real(0.0);

            if (A_side == Real(0.0) && A_top == Real(0.0) && A_bot == Real(0.0)) { return; }

            const Real r = ptrs.radius_ptr[i];

            // Gravitational settling velocity (use the SD terminal velocity).
            const Real v_settle = wl.settling_correction
                                ? std::abs(ptrs.vterm_ptr[i]) : Real(0.0);

            // Per-boundary deposition velocities.
            const Real vd_side = sd_vd_turbulent(r, wl, wl.vd_side);
            const Real vd_top  = amrex::max(sd_vd_turbulent(r, wl, wl.vd_top) - v_settle, Real(0.0));
            const Real vd_bot  = sd_vd_turbulent(r, wl, wl.vd_bottom) + v_settle;

            // First-order loss coefficients per boundary group.
            const Real k_side = sd_wall_loss_coefficient(vd_side, A_side, V_cell);
            const Real k_top  = sd_wall_loss_coefficient(vd_top,  A_top,  V_cell);
            const Real k_bot  = sd_wall_loss_coefficient(vd_bot,  A_bot,  V_cell);
            const Real k_total = k_side + k_top + k_bot;
            if (k_total <= Real(0.0)) { return; }

            const Real xi_old = ptrs.mult_ptr[i];

            // Removed multiplicity: deterministic (A) or stochastic thinning (B).
            Real removed;
            if (wl.method == SDWL_Stochastic) {
                const Real P = sd_wall_loss_probability(k_total, a_dt);
                removed = sd_wall_loss_stochastic_removed(xi_old, P, engine);
            } else {
                removed = xi_old * (Real(1.0) - std::exp(-k_total * a_dt));
            }
            Real xi_new = xi_old - removed;

            // Deactivate depleted super-droplets; remaining xi is counted deposited.
            if (wl.deactivate_on_depletion && xi_new < wl.deactivate_threshold) {
                removed += xi_new;
                xi_new = Real(0.0);
                ptrs.active_ptr[i] = 0;
                p.id() = -1;
            }
            ptrs.mult_ptr[i] = xi_new;

            // Attribute the removal to each boundary by its share of k_total.
            const Real inv_k = Real(1.0) / k_total;
            const Real rem_side = removed * (k_side * inv_k);
            const Real rem_top  = removed * (k_top  * inv_k);
            const Real rem_bot  = removed * (k_bot  * inv_k);

            const Real mass = ptrs.mass_ptr[i];
            const int  b    = sd_wall_loss_bin_index(r, wl);

            Gpu::Atomic::Add(&accum_ptr[0], rem_side);
            Gpu::Atomic::Add(&accum_ptr[1], rem_top);
            Gpu::Atomic::Add(&accum_ptr[2], rem_bot);
            Gpu::Atomic::Add(&accum_ptr[3], rem_side * mass);
            Gpu::Atomic::Add(&accum_ptr[4], rem_top  * mass);
            Gpu::Atomic::Add(&accum_ptr[5], rem_bot  * mass);
            Gpu::Atomic::Add(&accum_ptr[6 + b], removed);
        });
    });
    Gpu::synchronize();

    // Reduce accumulators to host and across MPI ranks.
    Vector<Real> h(nacc, Real(0.0));
    Gpu::copy(Gpu::deviceToHost, accum.begin(), accum.end(), h.begin());
    ParallelDescriptor::ReduceRealSum(h.dataPtr(), nacc);

    const Real removed_side = h[0];
    const Real removed_top  = h[1];
    const Real removed_bot  = h[2];

    // Cumulative deposited mass by boundary type (accumulates every step).
    m_wall_loss_dep_mass_side   += h[3];
    m_wall_loss_dep_mass_top    += h[4];
    m_wall_loss_dep_mass_bottom += h[5];

    // Write diagnostics file on level 0 at the requested cadence.
    if (a_lev != 0) { return; }
    if (wl.diag_interval <= 0) { return; }
    if (((a_iter + 1) % wl.diag_interval) != 0) { return; }

    if (ParallelDescriptor::IOProcessor()) {
        const std::string filename = m_name + "_wall_loss.txt";
        std::ofstream outfile;

        if (!m_wall_loss_log_initialized) {
            outfile.open(filename.c_str(), std::ios::out | std::ios::trunc);
            if (!outfile.good()) { amrex::FileOpenFailed(filename); }
            outfile << "# Super-droplet boundary-deposition wall-loss diagnostics\n";
            outfile << "# Method: " << (wl.method == SDWL_Stochastic ? "stochastic" : "deterministic")
                    << ",  deposition-velocity model: "
                    << (wl.dep_vel_model == SDWL_Diffusion ? "diffusion" : "constant") << "\n";
            outfile << "# vd_side=" << wl.vd_side << " vd_top=" << wl.vd_top
                    << " vd_bottom=" << wl.vd_bottom << " m/s, settling_correction="
                    << (wl.settling_correction ? "on" : "off") << "\n";
            outfile << "# Size bins: " << nbins << " log-spaced on r in ["
                    << wl.bin_rmin << ", " << wl.bin_rmax << "] m\n";
            outfile << "# Columns:\n";
            outfile << "# 1: Time (s)\n";
            outfile << "# 2: dt (s)\n";
            outfile << "# 3: Multiplicity lost to side walls this step (#)\n";
            outfile << "# 4: Multiplicity lost to top boundary this step (#)\n";
            outfile << "# 5: Multiplicity lost to bottom boundary this step (#)\n";
            outfile << "# 6: Cumulative deposited mass on side walls (kg)\n";
            outfile << "# 7: Cumulative deposited mass on top boundary (kg)\n";
            outfile << "# 8: Cumulative deposited mass on bottom boundary (kg)\n";
            outfile << "# 9..: Size-resolved deposition rate (#/s), low-r to high-r\n";
            m_wall_loss_log_initialized = true;
        } else {
            outfile.open(filename.c_str(), std::ios::out | std::ios::app);
            if (!outfile.good()) { amrex::FileOpenFailed(filename); }
        }

        const Real inv_dt = (a_dt > Real(0.0)) ? (Real(1.0) / a_dt) : Real(0.0);

        outfile << std::scientific << std::setprecision(8)
                << a_time << "  "
                << a_dt << "  "
                << removed_side << "  "
                << removed_top << "  "
                << removed_bot << "  "
                << m_wall_loss_dep_mass_side << "  "
                << m_wall_loss_dep_mass_top << "  "
                << m_wall_loss_dep_mass_bottom;
        for (int b = 0; b < nbins; ++b) {
            outfile << "  " << h[6 + b] * inv_dt;
        }
        outfile << "\n";

        outfile.flush();
        outfile.close();
        if (!outfile.good()) { amrex::Abort("problem writing wall-loss diagnostics file"); }
    }
}

#endif
