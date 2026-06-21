    ParmParse pp_prob("prob");

    Real T_0_Pert_Mag  = zero; pp_prob.query("T_0_Pert_Mag",  T_0_Pert_Mag);
    Real qv_0_Pert_Mag = zero; pp_prob.query("qv_0_Pert_Mag", qv_0_Pert_Mag);

    const bool use_moisture = (sc.moisture_type != MoistureType::None);
    const Real rdOcp        = sc.rdOcp;

    ParallelForRNG(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
    {
        // Perturb the temperature at constant pressure to seed convection.
        // Holding pressure (and rho*theta) fixed, the perturbation is stored as
        // a density perturbation, matching the approach used by the other probs.
        if (T_0_Pert_Mag != zero)
        {
            Real rho      = state(i,j,k,Rho_comp);
            Real rhotheta = state(i,j,k,RhoTheta_comp);
            Real qv       = use_moisture ? state(i,j,k,RhoQ1_comp) / rho : zero;
            Real Told     = getTgivenRandRTh(rho, rhotheta, qv);
            Real P        = getPgivenRTh(rhotheta, qv);

            Real rand_double = amrex::Random(engine); // Between zero and one
            Real Tnew      = Told + (rand_double*two - one)*T_0_Pert_Mag;

            Real theta_new = getThgivenTandP(Tnew, P, rdOcp);
            Real rhonew    = getRhogivenThetaPress(theta_new, P, rdOcp, qv);

            state_pert(i, j, k, Rho_comp)      = rhonew - rho;
            state_pert(i, j, k, RhoTheta_comp) = zero;
        }

        if (use_moisture && (qv_0_Pert_Mag != zero))
        {
            Real rhoold = state(i,j,k,Rho_comp);
            Real rhonew = rhoold + state_pert(i,j,k,Rho_comp);
            Real qvold  = state(i,j,k,RhoQ1_comp) / rhoold;

            Real rand_double = amrex::Random(engine); // Between zero and one
            Real qvnew  = qvold + (rand_double*two - one)*qv_0_Pert_Mag;

            state_pert(i, j, k, RhoQ1_comp) = rhonew*qvnew - rhoold*qvold;
        }
    });
