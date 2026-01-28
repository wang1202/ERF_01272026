/**
 * @file ERF_WriteWallFluxes.cpp
 * @brief Simple wall flux diagnostic output
 */

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <AMReX_Print.H>
#include "ERF_IndexDefines.H"
#include "ERF_DataStruct.H"
#include "ERF_EOS.H"
#include "ERF_Constants.H"
#include <fstream>
#include <iomanip>

using namespace amrex;

/**
 * @brief Write mean wall fluxes to file
 */
void write_wallflux_output(
    int level,
    Real time,
    int timestep,
    const MultiFab& cons,
    const Vector<std::unique_ptr<MultiFab>>& Tau_lev,
    MultiFab* Hfx1, MultiFab* Hfx2, MultiFab* Hfx3,
    MultiFab* Q1fx1, MultiFab* Q1fx2, MultiFab* Q1fx3,
    const Geometry& geom,
    const std::string& filename)
{
    BL_PROFILE("write_wallflux_output");

    const Box& domain = geom.Domain();
    const int ilo = domain.smallEnd(0);
    const int ihi = domain.bigEnd(0) + 1;
    const int jlo = domain.smallEnd(1);
    const int jhi = domain.bigEnd(1) + 1;
    const int klo = domain.smallEnd(2);
    const int khi = domain.bigEnd(2) + 1;

    // Number of cells on each face type
    const int ncells_x = (domain.bigEnd(0) - domain.smallEnd(0) + 1);
    const int ncells_y = (domain.bigEnd(1) - domain.smallEnd(1) + 1);
    const int ncells_z = (domain.bigEnd(2) - domain.smallEnd(2) + 1);
    const Real num_cells_xface = Real(ncells_y * ncells_z);  // cells on x-normal face
    const Real num_cells_yface = Real(ncells_x * ncells_z);  // cells on y-normal face
    const Real num_cells_zface = Real(ncells_x * ncells_y);  // cells on z-normal face

    constexpr Real cp = 1004.0; // J/(kg·K)

    // Mean fluxes: [tau, H, E] x 6 faces = 18 values
    std::array<Real, 18> fluxes = {0.0};

    // Helper function to compute mean flux on a face
    auto compute_face_flux = [&](const MultiFab& mf, int face_idx) -> Real {
        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto const& arr = mf.const_array(mfi);

            reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                if (face_idx == 0 && i == ilo) return arr(i,j,k);
                if (face_idx == 1 && i == ihi) return arr(i,j,k);
                if (face_idx == 2 && j == jlo) return arr(i,j,k);
                if (face_idx == 3 && j == jhi) return arr(i,j,k);
                if (face_idx == 4 && k == klo) return arr(i,j,k);
                if (face_idx == 5 && k == khi) return arr(i,j,k);
                return 0.0;
            });
        }

        Real sum = get<0>(reduce_data.value());
        ParallelDescriptor::ReduceRealSum(sum);

        // Divide by number of cells on the face, not physical area
        Real num_cells = 1.0;
        if (face_idx < 2) num_cells = num_cells_xface;
        else if (face_idx < 4) num_cells = num_cells_yface;
        else num_cells = num_cells_zface;

        return sum / num_cells;
    };

    // X-faces (0=xlo, 1=xhi)
    if (!geom.isPeriodic(0)) {
        // X-face shear stress (tau): sqrt(tau_yx^2 + tau_zx^2)
        if (Tau_lev.size() > TauType::tau21 && Tau_lev[TauType::tau21] != nullptr &&
            Tau_lev.size() > TauType::tau31 && Tau_lev[TauType::tau31] != nullptr) {

            // xlo tau
            ReduceOps<ReduceOpSum> reduce_op;
            ReduceData<Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            for (MFIter mfi(*Tau_lev[TauType::tau21]); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto const& tau21_arr = Tau_lev[TauType::tau21]->const_array(mfi);
                auto const& tau31_arr = Tau_lev[TauType::tau31]->const_array(mfi);

                reduce_op.eval(bx, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (i == ilo) {
                        Real tau_yx = tau21_arr(i,j,k);
                        Real tau_zx = tau31_arr(i,j,k);
                        return std::sqrt(tau_yx*tau_yx + tau_zx*tau_zx);
                    }
                    return 0.0;
                });
            }
            Real tau_sum = get<0>(reduce_data.value());
            ParallelDescriptor::ReduceRealSum(tau_sum);
            fluxes[0] = tau_sum / num_cells_xface;

            // xhi tau
            reduce_data.value() = 0.0;
            for (MFIter mfi(*Tau_lev[TauType::tau21]); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto const& tau21_arr = Tau_lev[TauType::tau21]->const_array(mfi);
                auto const& tau31_arr = Tau_lev[TauType::tau31]->const_array(mfi);

                reduce_op.eval(bx, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (i == ihi) {
                        Real tau_yx = tau21_arr(i,j,k);
                        Real tau_zx = tau31_arr(i,j,k);
                        return std::sqrt(tau_yx*tau_yx + tau_zx*tau_zx);
                    }
                    return 0.0;
                });
            }
            tau_sum = get<0>(reduce_data.value());
            ParallelDescriptor::ReduceRealSum(tau_sum);
            fluxes[3] = tau_sum / num_cells_xface;
        }

        if (Hfx1) {
            // xlo heat flux
            ReduceOps<ReduceOpSum> reduce_op;
            ReduceData<Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            for (MFIter mfi(*Hfx1); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto const& hfx_arr = Hfx1->const_array(mfi);
                auto const& cons_arr = cons.const_array(mfi);

                reduce_op.eval(bx, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (i == ilo) {
                        Real rho_theta_flux = hfx_arr(i,j,k);
                        Real rhotheta = cons_arr(ilo,j,k,RhoTheta_comp);
                        Real p_pa = getPgivenRTh(rhotheta);
                        Real exner = getExnergivenP(p_pa, R_d/Cp_d);
                        return cp * exner * rho_theta_flux;
                    }
                    return 0.0;
                });
            }
            Real hfx_sum = get<0>(reduce_data.value());
            ParallelDescriptor::ReduceRealSum(hfx_sum);
            fluxes[1] = hfx_sum / num_cells_xface;

            // xhi heat flux
            reduce_data.value() = 0.0;
            for (MFIter mfi(*Hfx1); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto const& hfx_arr = Hfx1->const_array(mfi);
                auto const& cons_arr = cons.const_array(mfi);

                reduce_op.eval(bx, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (i == ihi) {
                        Real rho_theta_flux = hfx_arr(i,j,k);
                        Real rhotheta = cons_arr(ihi-1,j,k,RhoTheta_comp);
                        Real p_pa = getPgivenRTh(rhotheta);
                        Real exner = getExnergivenP(p_pa, R_d/Cp_d);
                        return cp * exner * rho_theta_flux;
                    }
                    return 0.0;
                });
            }
            hfx_sum = get<0>(reduce_data.value());
            ParallelDescriptor::ReduceRealSum(hfx_sum);
            fluxes[4] = hfx_sum / num_cells_xface;
        }
        if (Q1fx1) {
            fluxes[2] = compute_face_flux(*Q1fx1, 0);
            fluxes[5] = compute_face_flux(*Q1fx1, 1);
        }
    }

    // Y-faces (2=ylo, 3=yhi)
    if (!geom.isPeriodic(1)) {
        // Y-face shear stress (tau): sqrt(tau_xy^2 + tau_zy^2)
        if (Tau_lev.size() > TauType::tau12 && Tau_lev[TauType::tau12] != nullptr &&
            Tau_lev.size() > TauType::tau32 && Tau_lev[TauType::tau32] != nullptr) {

            // ylo tau
            ReduceOps<ReduceOpSum> reduce_op;
            ReduceData<Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            for (MFIter mfi(*Tau_lev[TauType::tau12]); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto const& tau12_arr = Tau_lev[TauType::tau12]->const_array(mfi);
                auto const& tau32_arr = Tau_lev[TauType::tau32]->const_array(mfi);

                reduce_op.eval(bx, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (j == jlo) {
                        Real tau_xy = tau12_arr(i,j,k);
                        Real tau_zy = tau32_arr(i,j,k);
                        return std::sqrt(tau_xy*tau_xy + tau_zy*tau_zy);
                    }
                    return 0.0;
                });
            }
            Real tau_sum = get<0>(reduce_data.value());
            ParallelDescriptor::ReduceRealSum(tau_sum);
            fluxes[6] = tau_sum / num_cells_yface;

            // yhi tau
            reduce_data.value() = 0.0;
            for (MFIter mfi(*Tau_lev[TauType::tau12]); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto const& tau12_arr = Tau_lev[TauType::tau12]->const_array(mfi);
                auto const& tau32_arr = Tau_lev[TauType::tau32]->const_array(mfi);

                reduce_op.eval(bx, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (j == jhi) {
                        Real tau_xy = tau12_arr(i,j,k);
                        Real tau_zy = tau32_arr(i,j,k);
                        return std::sqrt(tau_xy*tau_xy + tau_zy*tau_zy);
                    }
                    return 0.0;
                });
            }
            tau_sum = get<0>(reduce_data.value());
            ParallelDescriptor::ReduceRealSum(tau_sum);
            fluxes[9] = tau_sum / num_cells_yface;
        }

        if (Hfx2) {
            // ylo heat flux
            ReduceOps<ReduceOpSum> reduce_op;
            ReduceData<Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            for (MFIter mfi(*Hfx2); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto const& hfx_arr = Hfx2->const_array(mfi);
                auto const& cons_arr = cons.const_array(mfi);

                reduce_op.eval(bx, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (j == jlo) {
                        Real rho_theta_flux = hfx_arr(i,j,k);
                        Real rhotheta = cons_arr(i,jlo,k,RhoTheta_comp);
                        Real p_pa = getPgivenRTh(rhotheta);
                        Real exner = getExnergivenP(p_pa, R_d/Cp_d);
                        return cp * exner * rho_theta_flux;
                    }
                    return 0.0;
                });
            }
            Real hfx_sum = get<0>(reduce_data.value());
            ParallelDescriptor::ReduceRealSum(hfx_sum);
            fluxes[7] = hfx_sum / num_cells_yface;

            // yhi heat flux
            reduce_data.value() = 0.0;
            for (MFIter mfi(*Hfx2); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto const& hfx_arr = Hfx2->const_array(mfi);
                auto const& cons_arr = cons.const_array(mfi);

                reduce_op.eval(bx, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (j == jhi) {
                        Real rho_theta_flux = hfx_arr(i,j,k);
                        Real rhotheta = cons_arr(i,jhi-1,k,RhoTheta_comp);
                        Real p_pa = getPgivenRTh(rhotheta);
                        Real exner = getExnergivenP(p_pa, R_d/Cp_d);
                        return cp * exner * rho_theta_flux;
                    }
                    return 0.0;
                });
            }
            hfx_sum = get<0>(reduce_data.value());
            ParallelDescriptor::ReduceRealSum(hfx_sum);
            fluxes[10] = hfx_sum / num_cells_yface;
        }
        if (Q1fx2) {
            fluxes[8] = compute_face_flux(*Q1fx2, 2);
            fluxes[11] = compute_face_flux(*Q1fx2, 3);
        }
    }

    // Z-faces (4=zlo, 5=zhi)
    if (!geom.isPeriodic(2)) {
        // Z-face shear stress (tau): sqrt(tau_xz^2 + tau_yz^2)
        if (Tau_lev.size() > TauType::tau13 && Tau_lev[TauType::tau13] != nullptr &&
            Tau_lev.size() > TauType::tau23 && Tau_lev[TauType::tau23] != nullptr) {

            // zlo tau
            ReduceOps<ReduceOpSum> reduce_op;
            ReduceData<Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            for (MFIter mfi(*Tau_lev[TauType::tau13]); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto const& tau13_arr = Tau_lev[TauType::tau13]->const_array(mfi);
                auto const& tau23_arr = Tau_lev[TauType::tau23]->const_array(mfi);

                reduce_op.eval(bx, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (k == klo) {
                        Real tau_xz = tau13_arr(i,j,k);
                        Real tau_yz = tau23_arr(i,j,k);
                        return std::sqrt(tau_xz*tau_xz + tau_yz*tau_yz);
                    }
                    return 0.0;
                });
            }
            Real tau_sum = get<0>(reduce_data.value());
            ParallelDescriptor::ReduceRealSum(tau_sum);
            fluxes[12] = tau_sum / num_cells_zface;

            // zhi tau
            reduce_data.value() = 0.0;
            for (MFIter mfi(*Tau_lev[TauType::tau13]); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto const& tau13_arr = Tau_lev[TauType::tau13]->const_array(mfi);
                auto const& tau23_arr = Tau_lev[TauType::tau23]->const_array(mfi);

                reduce_op.eval(bx, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (k == khi) {
                        Real tau_xz = tau13_arr(i,j,k);
                        Real tau_yz = tau23_arr(i,j,k);
                        return std::sqrt(tau_xz*tau_xz + tau_yz*tau_yz);
                    }
                    return 0.0;
                });
            }
            tau_sum = get<0>(reduce_data.value());
            ParallelDescriptor::ReduceRealSum(tau_sum);
            fluxes[15] = tau_sum / num_cells_zface;
        }

        if (Hfx3) {
            // zlo heat flux
            ReduceOps<ReduceOpSum> reduce_op;
            ReduceData<Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            for (MFIter mfi(*Hfx3); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto const& hfx_arr = Hfx3->const_array(mfi);
                auto const& cons_arr = cons.const_array(mfi);

                reduce_op.eval(bx, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (k == klo) {
                        Real rho_theta_flux = hfx_arr(i,j,k);
                        Real rhotheta = cons_arr(i,j,klo,RhoTheta_comp);
                        Real p_pa = getPgivenRTh(rhotheta);
                        Real exner = getExnergivenP(p_pa, R_d/Cp_d);
                        return cp * exner * rho_theta_flux;
                    }
                    return 0.0;
                });
            }
            Real hfx_sum = get<0>(reduce_data.value());
            ParallelDescriptor::ReduceRealSum(hfx_sum);
            fluxes[13] = hfx_sum / num_cells_zface;

            // zhi heat flux
            reduce_data.value() = 0.0;
            for (MFIter mfi(*Hfx3); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto const& hfx_arr = Hfx3->const_array(mfi);
                auto const& cons_arr = cons.const_array(mfi);

                reduce_op.eval(bx, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (k == khi) {
                        Real rho_theta_flux = hfx_arr(i,j,k);
                        Real rhotheta = cons_arr(i,j,khi-1,RhoTheta_comp);
                        Real p_pa = getPgivenRTh(rhotheta);
                        Real exner = getExnergivenP(p_pa, R_d/Cp_d);
                        return cp * exner * rho_theta_flux;
                    }
                    return 0.0;
                });
            }
            hfx_sum = get<0>(reduce_data.value());
            ParallelDescriptor::ReduceRealSum(hfx_sum);
            fluxes[16] = hfx_sum / num_cells_zface;
        }
        if (Q1fx3) {
            fluxes[14] = compute_face_flux(*Q1fx3, 4);
            fluxes[17] = compute_face_flux(*Q1fx3, 5);
        }
    }

    // Write to file (I/O processor only)
    if (ParallelDescriptor::IOProcessor()) {
        std::ofstream outfile;

        // Create/write header on first call
        bool file_exists = std::ifstream(filename).good();
        if (!file_exists) {
            outfile.open(filename, std::ios::out);
            outfile << std::setw(15) << "# time[s]";
            std::vector<std::string> faces = {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"};
            for (const auto& face : faces) {
                outfile << std::setw(18) << (face + "_tau[Pa]")
                       << std::setw(18) << (face + "_H[W/m2]")
                       << std::setw(18) << (face + "_E[kg/m2/s]");
            }
            outfile << "\n";
        } else {
            outfile.open(filename, std::ios::app);
        }

        outfile << std::scientific << std::setprecision(8) << std::setw(15) << time;
        for (int i = 0; i < 18; ++i) {
            outfile << std::setw(18) << fluxes[i];
        }
        outfile << "\n";
        outfile.close();
    }
}
