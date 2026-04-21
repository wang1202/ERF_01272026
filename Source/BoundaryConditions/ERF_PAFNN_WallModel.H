#ifndef ERF_PAFNN_WALLMODEL_H_
#define ERF_PAFNN_WALLMODEL_H_

/**
 * @file ERF_PAFNN_WallModel.H
 * @brief Physics-Aware Feedforward Neural Network (PAFNN) wall model
 *
 * DNS-trained neural network wall model for enclosed cloud chamber simulations.
 * Two separate networks:
 *   - BT: Bottom & Top (horizontal) walls — log inputs for Ra, DQ
 *   - SW: Side (vertical) walls — signed-log inputs for Ra, DQ
 *
 * Network architecture: 4 → [12] → [6] → 3, ReLU activation
 *
 * Inputs (4):
 *   x[0] = log(Re_h) = log(Up * h / nu)
 *   x[1] = log(Ra_h) or signed-log(Ra_h)
 *   x[2] = Re_w      = Un * h / nu
 *   x[3] = log(DQ*h/D_mol) or signed-log
 *
 * Outputs (3):
 *   y[0] = log(C_D)  → wall shear stress
 *   y[1] = log(Nu_h) → sensible heat flux
 *   y[2] = log(Sh_h) → moisture flux
 *
 * Trained on Pi Chamber DNS Cases 2-4 (512^3).
 * Reference: Wang et al. (2026)
 */

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <cmath>

namespace PAFNN {

// =====================================================================
// Fluid properties (DNS training conditions) — scalar constexpr OK for GPU
// =====================================================================
static constexpr amrex::Real pafnn_mu    = 1.717e-5;
static constexpr amrex::Real pafnn_rho   = 1.2096;
static constexpr amrex::Real pafnn_nu    = pafnn_mu / pafnn_rho;
static constexpr amrex::Real pafnn_Pr    = 0.72;
static constexpr amrex::Real pafnn_Sc    = 0.62;
static constexpr amrex::Real pafnn_cp    = 1004.0;
static constexpr amrex::Real pafnn_alpha = pafnn_mu / (pafnn_rho * pafnn_Pr);
static constexpr amrex::Real pafnn_Dmol  = pafnn_nu / pafnn_Sc;
static constexpr amrex::Real pafnn_kth   = pafnn_alpha * pafnn_rho * pafnn_cp;
static constexpr amrex::Real pafnn_beta  = 3.5e-3;
static constexpr amrex::Real pafnn_g     = 9.81;

// Network dimensions
static constexpr int N_IN  = 4;
static constexpr int N_H1  = 12;
static constexpr int N_H2  = 6;
static constexpr int N_OUT = 3;

// =====================================================================
// GPU-compatible NN forward pass: 4 → [12,ReLU] → [6,ReLU] → 3
// Takes pointers to weight arrays (must be device-accessible, e.g. local constexpr)
// =====================================================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void pafnn_forward (const amrex::Real x_sc[N_IN],
                    amrex::Real y_out[N_OUT],
                    const amrex::Real* w1, const amrex::Real* b1,
                    const amrex::Real* w2, const amrex::Real* b2,
                    const amrex::Real* w3, const amrex::Real* b3,
                    const amrex::Real* ymean, const amrex::Real* ystd)
{
    // Layer 1: h1 = ReLU(W1^T * x + b1)
    amrex::Real h1[N_H1];
    for (int j = 0; j < N_H1; ++j) {
        amrex::Real sum = b1[j];
        for (int i = 0; i < N_IN; ++i) {
            sum += x_sc[i] * w1[i * N_H1 + j];
        }
        h1[j] = amrex::max(sum, amrex::Real(0.0));
    }

    // Layer 2: h2 = ReLU(W2^T * h1 + b2)
    amrex::Real h2[N_H2];
    for (int j = 0; j < N_H2; ++j) {
        amrex::Real sum = b2[j];
        for (int i = 0; i < N_H1; ++i) {
            sum += h1[i] * w2[i * N_H2 + j];
        }
        h2[j] = amrex::max(sum, amrex::Real(0.0));
    }

    // Layer 3: y = W3^T * h2 + b3  (linear output)
    for (int j = 0; j < N_OUT; ++j) {
        amrex::Real sum = b3[j];
        for (int i = 0; i < N_H2; ++i) {
            sum += h2[i] * w3[i * N_OUT + j];
        }
        y_out[j] = sum * ystd[j] + ymean[j];
    }
}

/**
 * @brief Signed-log transform: sgn(x) * log(|x|)
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real signed_log (amrex::Real x)
{
    if (x >= amrex::Real(0.0)) {
        return std::log(amrex::max(x, amrex::Real(1e-30)));
    } else {
        return -std::log(amrex::max(-x, amrex::Real(1e-30)));
    }
}

/**
 * @brief Compute PAFNN wall fluxes for a Bottom/Top wall point.
 *
 * @param Up      Wall-parallel velocity magnitude [m/s]
 * @param Un      Wall-normal velocity [m/s]
 * @param DT      |T_wall - T_air| [K], must be > 0
 * @param DQ      |q_wall - q_air| [kg/kg], must be > 0
 * @param h       Distance from wall to cell center [m]
 * @param tau_out [out] Kinematic wall stress tau/rho [m^2/s^2]
 * @param H_out   [out] Kinematic heat flux Q [K m/s]  (= H/(rho cp))
 * @param E_out   [out] Kinematic moisture flux E [m/s] (= E_q/rho)
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void pafnn_bt_fluxes (amrex::Real Up, amrex::Real Un,
                      amrex::Real DT, amrex::Real DQ,
                      amrex::Real h,
                      amrex::Real& tau_out,
                      amrex::Real& H_out,
                      amrex::Real& E_out)
{
    using amrex::Real;

    // --- BT model weights as local constexpr (GPU-accessible) ---
    constexpr Real w1[] = {
        3.064370082150245e-03, 5.275858523009069e-01, 3.340063643099085e-01, 1.385221450657108e-01, -7.233589231382295e-01, -7.703795930802252e-01, -1.932941365027754e-01, 5.634271020986720e-01, -1.018974213363632e-01, 4.459624688873944e-01, -5.855206234233327e-01, 7.949415364150912e-01,
        4.872853784203369e-01, -2.330488212312515e-01, -3.483009574098287e-01, -3.713790800285017e-01, -1.141116807210688e-01, -2.000747484588393e-02, -4.060762439411007e-02, -4.219648077954705e-01, 1.563094637487377e-01, -1.434606261644996e-01, -5.936332974021383e-02, -2.739098995064753e-02,
        -1.878213188558225e-01, 2.547452037966070e-01, -2.626991870827479e-01, 1.194440517576759e-01, 1.980401087062194e-01, -4.523085531690223e-01, -8.520940151023192e-02, -2.424090376458532e-01, -4.293043629308991e-01, 4.779371595729128e-01, 4.318635843884992e-01, 3.849276723746892e-02,
        -1.692858124057451e-01, -3.724768378160869e-01, 2.503549119585053e-01, -1.068441841597742e-01, -3.203399233298992e-01, -4.204977921054329e-02, -5.341936226512450e-01, 2.455223095712599e-01, -2.626685757906163e-01, 4.818502325103078e-01, -5.873346266746663e-02, 8.456196391593664e-02
    };
    constexpr Real b1[] = {
        4.064152842081473e-02, -4.542181670232892e-01, 5.361746033974895e-01, 5.839989218105086e-01, 4.348125392787738e-01, 4.080237709832499e-01, 2.349401466414940e-01, 8.002569588947731e-01, -3.458632943477821e-01, -2.961969411785798e-01, -6.555609565436613e-01, 6.525838353065300e-03
    };
    constexpr Real w2[] = {
        4.151197095916561e-01, -3.350797045764056e-01, 3.497294508072759e-01, -1.588317623043658e-01, -2.316041445649140e-01, 7.172764376993218e-02,
        -7.410767566879581e-02, 4.937064259591245e-01, -3.434940472494146e-01, 5.229719276781490e-01, 2.325392774649768e-01, -2.701799867083549e-01,
        -3.370234564825723e-01, 6.202632943002576e-01, 2.048224657243134e-01, 2.258650235039156e-01, 2.069912760157727e-01, -3.831319651015045e-01,
        -2.028044602367896e-01, -1.525449347670293e-01, 4.285569942539250e-01, 1.020715536480779e-01, -3.251657213659521e-01, -1.958034115003523e-01,
        1.310943597101857e-01, -2.020841081012177e-01, 1.218797526567845e-01, 1.239422813756528e-01, 5.740341520005645e-01, -8.486499536332229e-02,
        5.181333684499149e-02, 4.012638184446991e-02, 1.650677297459696e-01, 4.367682938479693e-02, 4.548356422829332e-01, 1.346906219975064e-01,
        2.450667833924589e-01, -7.694312030145282e-02, -5.936095503039310e-01, -4.894243753875836e-01, -4.223478925249446e-01, 3.193153029907929e-02,
        -9.734190552463381e-02, 2.516348159543028e-01, 4.735077243963247e-01, -3.318713680966155e-01, -2.058943338144942e-01, 3.743960245650683e-01,
        -6.850592956802347e-03, 1.357150930691914e-04, -2.336290203404446e-01, -4.337397651043819e-01, 3.788433632964849e-01, 4.019646376715073e-01,
        3.068799764708822e-02, 4.524545491564801e-01, 5.044773188728948e-01, -4.016513143144322e-01, 4.032646790652865e-01, -7.884244363427818e-02,
        6.579581466765655e-01, 3.190393413567965e-01, -4.456516260290561e-01, -4.794095930454922e-01, 7.151380128517572e-02, -1.418481272656029e-01,
        2.992922901337887e-01, 6.531675921688461e-01, -3.966723112623936e-01, -3.084444464356425e-02, -1.634328857021826e-01, -4.736384068439699e-01
    };
    constexpr Real b2[] = {
        -1.648792415442921e-01, 1.463118404424152e-02, 4.643466480809131e-01, -2.417200982255847e-01, -3.607404116219796e-02, 2.639670476967915e-01
    };
    constexpr Real w3[] = {
        3.307188541042613e-01, 2.269274502538183e-01, 7.995768962144115e-01,
        -6.019275683953419e-01, 1.907022273598074e-01, -1.520149116365349e-01,
        -4.346154818034769e-01, -6.334312927475130e-01, 3.475490755104627e-01,
        -2.910918217207209e-02, -7.030953058183215e-01, -3.294003141859341e-01,
        9.604305619635520e-01, -3.651092836280315e-01, -2.985922703574140e-01,
        3.187627016250898e-01, 8.940484202260556e-01, -1.423595168334290e-01
    };
    constexpr Real b3[] = {
        4.970304148212690e-01, 5.272753795948020e-01, -2.357439381173649e-01
    };
    constexpr Real xmean[] = { 3.142493029960626e+00, 6.128987724171129e+00, 7.262499871173722e-03, -7.475879819824058e-01 };
    constexpr Real xstd[]  = { 7.495774070912368e-01, 4.648965985209141e-01, 9.827993540703270e-01, 4.655298906357982e-01 };
    constexpr Real ymean[] = { -2.375534974767009e-01, 2.772588866422343e+00, 2.772588653789040e+00 };
    constexpr Real ystd[]  = { 9.401706943850836e-01, 2.491388545919637e-05, 2.542723694625104e-06 };

    // Clamp inputs
    Up = amrex::max(Up, Real(1e-5));
    DT = amrex::max(DT, Real(1e-5));
    DQ = amrex::max(DQ, Real(1e-12));

    // Build raw features
    Real x_raw[N_IN];
    x_raw[0] = std::log(Up * h / pafnn_nu);
    x_raw[1] = std::log(pafnn_g * pafnn_beta * DT * h * h * h / (pafnn_nu * pafnn_alpha));
    x_raw[2] = Un * h / pafnn_nu;
    x_raw[3] = std::log(DQ * h / pafnn_Dmol);

    // Standardize
    Real x_sc[N_IN];
    for (int i = 0; i < N_IN; ++i) {
        x_sc[i] = (x_raw[i] - xmean[i]) / xstd[i];
    }

    // Forward pass
    Real y_out[N_OUT];
    pafnn_forward(x_sc, y_out, w1, b1, w2, b2, w3, b3, ymean, ystd);

    Real C_D  = std::exp(y_out[0]);
    Real Nu_h = std::exp(y_out[1]);
    Real Sh_h = std::exp(y_out[2]);

    tau_out = C_D * Up * Up;
    H_out   = Nu_h * pafnn_kth * DT / (pafnn_rho * pafnn_cp * h);
    E_out   = Sh_h * pafnn_Dmol * DQ / h;
}

/**
 * @brief Compute PAFNN wall fluxes for a Side wall point.
 *
 * Same as BT but uses signed-log for Ra and DQ inputs.
 * DT and DQ may be negative on side walls.
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void pafnn_sw_fluxes (amrex::Real Up, amrex::Real Un,
                      amrex::Real DT, amrex::Real DQ,
                      amrex::Real h,
                      amrex::Real& tau_out,
                      amrex::Real& H_out,
                      amrex::Real& E_out)
{
    using amrex::Real;

    // --- SW model weights as local constexpr (GPU-accessible) ---
    constexpr Real w1[] = {
        9.902369825936196e-02, 4.595979222021430e-01, 2.710657127205685e-01, 1.914352657311553e-01, -6.611654428145259e-01, -7.612575606926185e-01, -4.302893988373693e-01, 6.322917990539517e-01, -3.978959583744893e-02, 2.678549061117731e-01, -5.077236040008231e-01, 5.471206650978032e-01,
        5.161514015305122e-01, -4.135890388561100e-01, -2.848403539089102e-01, -3.299791402812937e-01, -8.649296589182033e-02, 2.199024625966109e-01, -1.327128350973563e-01, -1.183096090996787e-01, 8.484405202285582e-02, -3.897545703378836e-01, -1.565181173041005e-01, -3.369048865826946e-01,
        -8.186946552136511e-02, 1.981193917756105e-01, -2.338795325714443e-01, 1.088725202061429e-01, 2.009495410074003e-01, -4.125084566939105e-01, -8.950087915843666e-02, -2.660138879081451e-01, -4.127956869877953e-01, 4.642289396628373e-01, 4.639435131142532e-01, 1.978409246917551e-01,
        -3.946624383126098e-01, -3.021399821244913e-01, 1.546520306443263e-01, -1.708836245141596e-01, -3.211767217961137e-01, -4.348419546752885e-02, -3.565347993310268e-01, 2.557069419174113e-01, -1.700559721804331e-01, 1.689584395785009e-01, -2.161201957055508e-01, 1.480185098682962e-01
    };
    constexpr Real b1[] = {
        2.017486080311496e-01, -4.806565104217038e-01, 4.971651538121114e-01, 5.691808907718451e-01, 5.097033892538295e-01, 4.342696058848838e-01, 2.925994075620448e-01, 6.021608017224747e-01, -5.561827099133475e-01, -4.013503141000935e-01, -7.424571378083511e-01, -1.821577861805830e-01
    };
    constexpr Real w2[] = {
        -1.395926764954169e-01, -2.748291804603334e-01, 5.980377117192142e-01, -1.585452990715531e-01, -1.901553164960207e-01, 1.914143913992266e-01,
        -2.725730947111722e-01, 5.780871013170888e-01, -3.005194338501448e-01, 5.223010681641281e-01, 2.203826262285511e-01, -1.737412073008551e-01,
        -2.489386859067065e-02, 3.806741976845398e-01, 1.654967901913844e-01, 2.292463618587522e-01, 2.198561628045456e-01, -3.276869755135980e-01,
        -1.588258567466992e-01, -2.373171252099811e-01, 4.393387035437517e-01, 1.044469036386260e-01, -3.047824860256077e-01, -6.925153621152512e-02,
        1.027141847705065e-01, -2.911809089175659e-01, 1.560380766206270e-01, 1.223256118338023e-01, 6.312524664809190e-01, -6.039906445863375e-02,
        1.828203133830990e-01, 8.457766609300078e-02, 1.702542764761946e-01, 4.650572953618191e-02, 4.683999389204994e-01, -1.031792655968448e-01,
        1.953223337680964e-01, -1.772387604417399e-01, -6.883986539019226e-01, -4.787947173103718e-01, -2.435133926667996e-01, 9.767696707975501e-02,
        3.989770854564710e-02, 5.969466195701286e-02, 4.311101124783032e-01, -3.248417389560884e-01, -1.827882137661129e-01, 4.634923182359284e-01,
        2.366804266640746e-02, -5.090979342686907e-01, -1.297813628767099e-01, -4.114638049328245e-01, 3.789228290913645e-01, 3.450481991108746e-01,
        1.343669912666231e-01, 5.280289939436906e-01, 4.396644057696119e-01, -4.020816230270192e-01, 3.661657265109526e-01, 1.632534815796690e-01,
        2.467395233291432e-01, 3.734502209554538e-01, -2.876945112119108e-01, -4.832815034347078e-01, 7.735464312931073e-02, -1.541324132530154e-01,
        3.771504137251011e-01, 6.563989185792624e-01, -3.422531297286652e-01, -2.938638161708055e-02, -1.940219851115967e-01, -1.826485638016105e-01
    };
    constexpr Real b2[] = {
        -2.928038659349550e-02, -3.308695684164767e-01, 4.566541546733890e-01, -2.409064505471370e-01, -1.355471583236867e-02, 3.021880186131364e-01
    };
    constexpr Real w3[] = {
        2.732550045892231e-01, 3.605964651334414e-01, 7.504010598116170e-01,
        -6.010336882601595e-01, 1.367387154134883e-01, -1.682813898334427e-01,
        -6.336701882783417e-01, -6.480485999250709e-01, 3.122895471307376e-01,
        -3.189992931466407e-02, -7.046399600473704e-01, -3.288839670091275e-01,
        1.010584430909863e+00, -3.185032293792148e-01, -2.580004095497616e-01,
        -2.995331863256885e-01, 7.921421519250114e-01, -3.886592333217267e-01
    };
    constexpr Real b3[] = {
        5.210728121379822e-01, 5.521254601636131e-01, -2.155617965460525e-01
    };
    constexpr Real xmean[] = { 3.259314815449139e+00, -1.135742014303550e+00, 1.720120377010675e-05, 1.339044686192340e+00 };
    constexpr Real xstd[]  = { 8.294020512765707e-01, 4.318255041724622e+00, 7.265706656869069e-01, 2.730603038876582e+00 };
    constexpr Real ymean[] = { -4.093424787665559e-01, 2.772524578377258e+00, 2.772544893202265e+00 };
    constexpr Real ystd[]  = { 9.582870179500659e-01, 1.155128945044768e-02, 4.318372973514951e-03 };

    Up = amrex::max(Up, Real(1e-5));

    Real absDT = amrex::max(std::abs(DT), Real(1e-5));
    Real absDQ = amrex::max(std::abs(DQ), Real(1e-12));
    Real sign_DT = (DT >= Real(0.0)) ? Real(1.0) : Real(-1.0);
    Real sign_DQ = (DQ >= Real(0.0)) ? Real(1.0) : Real(-1.0);

    // Build raw features with signed-log for Ra and DQ
    Real x_raw[N_IN];
    x_raw[0] = std::log(Up * h / pafnn_nu);
    x_raw[1] = signed_log(sign_DT * pafnn_g * pafnn_beta * absDT * h * h * h / (pafnn_nu * pafnn_alpha));
    x_raw[2] = Un * h / pafnn_nu;
    x_raw[3] = signed_log(sign_DQ * absDQ * h / pafnn_Dmol);

    // Standardize
    Real x_sc[N_IN];
    for (int i = 0; i < N_IN; ++i) {
        x_sc[i] = (x_raw[i] - xmean[i]) / xstd[i];
    }

    // Forward pass
    Real y_out[N_OUT];
    pafnn_forward(x_sc, y_out, w1, b1, w2, b2, w3, b3, ymean, ystd);

    Real C_D  = std::exp(y_out[0]);
    Real Nu_h = std::exp(y_out[1]);
    Real Sh_h = std::exp(y_out[2]);

    tau_out = C_D * Up * Up;
    H_out   = sign_DT * Nu_h * pafnn_kth * absDT / (pafnn_rho * pafnn_cp * h);
    E_out   = sign_DQ * Sh_h * pafnn_Dmol * absDQ / h;
}

} // namespace PAFNN

#endif // ERF_PAFNN_WALLMODEL_H_
