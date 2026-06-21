    ParmParse pp_prob("prob");
    Real U_0 = zero; pp_prob.query("U_0", U_0);
    Real V_0 = zero; pp_prob.query("V_0", V_0);
    Real W_0 = zero; pp_prob.query("W_0", W_0);

    Real U_0_Pert_Mag = zero; pp_prob.query("U_0_Pert_Mag", U_0_Pert_Mag);
    Real V_0_Pert_Mag = zero; pp_prob.query("V_0_Pert_Mag", V_0_Pert_Mag);
    Real W_0_Pert_Mag = zero; pp_prob.query("W_0_Pert_Mag", W_0_Pert_Mag);

    // Domain face indices: hold the wall-normal velocity at zero on each wall
    // (closed box -- no penetration), but seed random noise on the interior faces.
    const int dom_lo_x = geomdata.Domain().smallEnd()[0];
    const int dom_hi_x = geomdata.Domain().bigEnd()[0];
    const int dom_lo_y = geomdata.Domain().smallEnd()[1];
    const int dom_hi_y = geomdata.Domain().bigEnd()[1];
    const int dom_lo_z = geomdata.Domain().smallEnd()[2];
    const int dom_hi_z = geomdata.Domain().bigEnd()[2];

    // Set the x-velocity
    ParallelForRNG(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
    {
        x_vel_pert(i, j, k) = U_0;
        if (i == dom_lo_x || i == dom_hi_x+1) {
            x_vel_pert(i, j, k) = zero;
        } else if (U_0_Pert_Mag != zero) {
            Real rand_double = amrex::Random(engine); // Between zero and one
            x_vel_pert(i, j, k) += (rand_double*two - one)*U_0_Pert_Mag;
        }
    });

    // Set the y-velocity
    ParallelForRNG(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
    {
        y_vel_pert(i, j, k) = V_0;
        if (j == dom_lo_y || j == dom_hi_y+1) {
            y_vel_pert(i, j, k) = zero;
        } else if (V_0_Pert_Mag != zero) {
            Real rand_double = amrex::Random(engine); // Between zero and one
            y_vel_pert(i, j, k) += (rand_double*two - one)*V_0_Pert_Mag;
        }
    });

    // Set the z-velocity
    ParallelForRNG(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
    {
        z_vel_pert(i, j, k) = W_0;
        if (k == dom_lo_z || k == dom_hi_z+1) {
            z_vel_pert(i, j, k) = zero;
        } else if (W_0_Pert_Mag != zero) {
            Real rand_double = amrex::Random(engine); // Between zero and one
            z_vel_pert(i, j, k) += (rand_double*two - one)*W_0_Pert_Mag;
        }
    });
