    // CloudChamber state initialization.
    //
    // Inputs (prob.*):
    //   rho_0, T_0, T_0_is_temperature  -> base state (handled in init_base_parms / InitDensityHSE;
    //                                      with rho_0 = p_0/(R_d*T_0) the base is self-consistent:
    //                                      p_hse = p_0, pi0 = 1, so T = theta).
    //   p_0            reference pressure used for the saturation (qsat) calculation [Pa].
    //   rh_0           target relative humidity in [0,1]; rh_0 = 1 -> exactly saturated, qc = 0.
    //   T_0_Pert_Mag   amplitude of the random temperature perturbation [K] used to seed convection.
    //
    // The temperature perturbation is applied at constant pressure and stored as a density
    // perturbation (rho*theta held fixed), matching the UserDefined / other closed-box probs.
    // Moisture is then initialized to rh_0 at the local (perturbed) temperature and p_0. For the
    // SuperDroplets model RhoQ1_comp is total water q_t, so rh_0 = 1 sets saturated vapor (qc = 0).

    ParmParse pp_prob("prob");

    Real T_0_Pert_Mag = zero; pp_prob.query("T_0_Pert_Mag", T_0_Pert_Mag);
    Real rh_0         = zero; pp_prob.query("rh_0",         rh_0);
    Real prob_p_0     = p_0;  pp_prob.query("p_0",          prob_p_0);   // p_0 = reference pressure (ERF_Constants)

    const bool use_moisture = (sc.moisture_type != MoistureType::None);
    const Real rdOcp        = sc.rdOcp;

    ParallelForRNG(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
    {
        const Real rho      = state(i,j,k,Rho_comp);
        const Real rhotheta = state(i,j,k,RhoTheta_comp);
        const Real P        = getPgivenRTh(rhotheta, zero);          // base state is dry (qv = 0)
        const Real Told     = getTgivenRandRTh(rho, rhotheta, zero); // for the chamber pi0 = 1 -> T = theta

        // Perturb the temperature at constant pressure; store as a density perturbation.
        Real Tnew = Told;
        if (T_0_Pert_Mag != zero)
        {
            const Real rand_double = amrex::Random(engine); // between zero and one
            Tnew = Told + (rand_double*two - one)*T_0_Pert_Mag;
            const Real theta_new = getThgivenTandP(Tnew, P, rdOcp);
            const Real rhonew    = getRhogivenThetaPress(theta_new, P, rdOcp, zero);
            state_pert(i, j, k, Rho_comp)      = rhonew - rho;
            state_pert(i, j, k, RhoTheta_comp) = zero;
        }

        // Initialize moisture to the target relative humidity rh_0 at (Tnew, p_0).
        // Base RhoQ1 (q_t) is zero, so the perturbation sets the full field.
        if (use_moisture && (rh_0 > zero))
        {
            const Real rhonew  = rho + state_pert(i, j, k, Rho_comp);
            const Real qv_init = vapor_mixing_ratio(prob_p_0, Tnew, rh_0);
            state_pert(i, j, k, RhoQ1_comp) = rhonew * qv_init;
        }
    });
