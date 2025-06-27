#include "utility/FieldGenerator.h"
#include "utility/Output.h"
#include "utility/Parameters.h"
#include "utility/Parser.h"
#include "utility/ParticleGenerator.h"
#include "utility/Random.h"
#include "utility/Timer.h"

#include "pica/math/Dimension.h"
#include "pica/math/Vectors.h"
#include "pica/particles/Particle.h"
#include "pica/particles/ParticleArray.h"
#include "pica/particlePush/BorisPusher.h"
#include "pica/threading/OpenMPHelper.h"

#include <algorithm>
#include <memory>

#include <riscv_vector.h>

using pica::FP;
using pica::FP3;

struct Particle
{
    FP3 position;
    FP3 p;
    uint64_t typeIndex;
};

template<class ParticleArray, class FieldValue>
void runBenchmark(ParticleArray& particles,
    const FieldValue& electricFieldValues,
    const FieldValue& magneticFieldValues,
    const utility::PusherParameters& parameters);

int main(int argc, char* argv[])
{
    utility::PusherParameters parameters = utility::readPusherParameters(argc, argv);
    utility::printHeader("pusher-precomputing benchmark: using optimized 3D Boris particle pusher implementation with precomputing of inverse gamma and AoS particle representation",
        parameters);

    // Generate particles randomly,
    // particular coordinates and other data are not important for this benchmark
    std::vector<Particle> particles(parameters.numParticles);
    utility::Random random;
    utility::detail::initParticleTypes(parameters.numParticleTypes);
    for (int i = 0; i < parameters.numParticles; i++) {
        pica::Particle3d particle;
        utility::detail::generateParticle(particle, random, parameters.numParticleTypes);
        particles[i].position = particle.getPosition();
        particles[i].p = particle.getP();
        particles[i].typeIndex = particle.getType();
    }

    // Generate random field values for each particle
    // to ensure there is no compile-time substitution of fields,
    // particular values of field are not important for this benchmark
    typedef FP3 FieldValue;
    FieldValue electricFieldValue = utility::generateField<double>();
    FieldValue magneticFieldValue = utility::generateField<double>();

    std::auto_ptr<utility::Stopwatch> timer(utility::createStopwatch());
    timer->start();
    runBenchmark(particles, electricFieldValue, magneticFieldValue, parameters);
    timer->stop();

    utility::printResult(parameters, timer->getElapsed());

    return 0;
}

double inv_one_add_norm2_m(const vbool64_t &redo_m, const vfloat64m1_t &one_1,const vfloat64m1_t &one, const vfloat64m1_t &val) {
	vfloat64m1_t v_val2  = __riscv_vfmul_vv_f64m1(val, val, 4);
	vfloat64m1_t v_norm2 = __riscv_vfredosum_vs_f64m1_f64m1_m(redo_m, v_val2, one, 4);
  v_norm2 = __riscv_vfdiv_vv_f64m1(one_1, v_norm2, 4);
  double res[4];
	__riscv_vse64_v_f64m1(res, v_norm2, 4);
	return res[0];
}


// Run the whole benchmark
template<class ParticleArray, class FieldValue>
void runBenchmark(ParticleArray& particles,
    const FieldValue& electricFieldValue,
    const FieldValue& magneticFieldValue,
    const utility::PusherParameters& parameters)
{
    omp_set_num_threads(parameters.numThreads);

    // value of time step does not matter for this benchmark, so make it
    // (somewhat) random to guarantee no compiler substitution is done for it
    utility::Random random;
    const double dt = random.getUniform() / pica::Constants<double>::c();

    std::vector<double> coeff(parameters.numParticleTypes);
    for (int i = 0; i < parameters.numParticleTypes; i++)
    {
        coeff[i] = pica::ParticleTypes::typesVector[i].charge * dt /
            (2.0 * pica::ParticleTypes::typesVector[i].mass * pica::Constants<FP>::c());
    }

    for (int i = 0; i < parameters.numIterations; i++) {
        // Each thread processes some particles
        const int numParticles = particles.size();
        const int numThreads = pica::getNumThreads();
        const int particlesPerThread = (numParticles + numThreads - 1) / numThreads;

    //    #pragma omp parallel for
        for (int idx = 0; idx < numThreads; idx++) {
            const int beginIdx = idx * particlesPerThread;
            const int endIdx = std::min(beginIdx + particlesPerThread, numParticles);

			vfloat64m1_t v_electricFieldValue = __riscv_vle64_v_f64m1(&(electricFieldValue.x), 3);
			vfloat64m1_t v_magneticFieldValue = __riscv_vle64_v_f64m1(&(magneticFieldValue.x), 3);

			vfloat64m1_t v_two  = __riscv_vfmv_v_f_f64m1(2.0, 4);
			vfloat64m1_t v_one  = __riscv_vfmv_v_f_f64m1(1.0, 4);
			vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, 4);
      uint64_t mask_a[] = {1,1,1,0};
      vuint64m1_t v_mask_ui64 = __riscv_vle64_v_u64m1(mask_a, 4);
      vbool64_t redo_m = __riscv_vmseq_vx_u64m1_b64(v_mask_ui64, 1, 4);

            for (int i = beginIdx; i < endIdx; i+=8) {
                // The code below uses precomputed coefficient:
                // eCoeff = q * dt / (2 * m * c)
                FP eCoeff0 = coeff[particles[i+0].typeIndex];
                FP eCoeff1 = coeff[particles[i+1].typeIndex];
                FP eCoeff2 = coeff[particles[i+2].typeIndex];
                FP eCoeff3 = coeff[particles[i+3].typeIndex];
                FP eCoeff4 = coeff[particles[i+4].typeIndex];
                FP eCoeff5 = coeff[particles[i+5].typeIndex];
                FP eCoeff6 = coeff[particles[i+6].typeIndex];
                FP eCoeff7 = coeff[particles[i+7].typeIndex];

//                FP3 eMomentum = electricFieldValue * eCoeff;
                vfloat64m1_t v_eMomentum0 = __riscv_vfmul_vf_f64m1(v_electricFieldValue, eCoeff0, 4);
                vfloat64m1_t v_eMomentum1 = __riscv_vfmul_vf_f64m1(v_electricFieldValue, eCoeff1, 4);
                vfloat64m1_t v_eMomentum2 = __riscv_vfmul_vf_f64m1(v_electricFieldValue, eCoeff2, 4);
                vfloat64m1_t v_eMomentum3 = __riscv_vfmul_vf_f64m1(v_electricFieldValue, eCoeff3, 4);
                vfloat64m1_t v_eMomentum4 = __riscv_vfmul_vf_f64m1(v_electricFieldValue, eCoeff4, 4);
                vfloat64m1_t v_eMomentum5 = __riscv_vfmul_vf_f64m1(v_electricFieldValue, eCoeff5, 4);
                vfloat64m1_t v_eMomentum6 = __riscv_vfmul_vf_f64m1(v_electricFieldValue, eCoeff6, 4);
                vfloat64m1_t v_eMomentum7 = __riscv_vfmul_vf_f64m1(v_electricFieldValue, eCoeff7, 4);

//                FP3 um = particles[i].p + eMomentum;
                vfloat64m1_t v_p0  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+0].p.x), 4);
                vfloat64m1_t v_p1  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+1].p.x), 4);
                vfloat64m1_t v_p2  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+2].p.x), 4);
                vfloat64m1_t v_p3  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+3].p.x), 4);
                vfloat64m1_t v_p4  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+4].p.x), 4);
                vfloat64m1_t v_p5  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+5].p.x), 4);
                vfloat64m1_t v_p6  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+6].p.x), 4);
                vfloat64m1_t v_p7  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+7].p.x), 4);
                
                vfloat64m1_t v_um0 = __riscv_vfadd_vv_f64m1(v_p0, v_eMomentum0, 4);
                vfloat64m1_t v_um1 = __riscv_vfadd_vv_f64m1(v_p1, v_eMomentum1, 4);
                vfloat64m1_t v_um2 = __riscv_vfadd_vv_f64m1(v_p2, v_eMomentum2, 4);
                vfloat64m1_t v_um3 = __riscv_vfadd_vv_f64m1(v_p3, v_eMomentum3, 4);
                vfloat64m1_t v_um4 = __riscv_vfadd_vv_f64m1(v_p4, v_eMomentum4, 4);
                vfloat64m1_t v_um5 = __riscv_vfadd_vv_f64m1(v_p5, v_eMomentum5, 4);
                vfloat64m1_t v_um6 = __riscv_vfadd_vv_f64m1(v_p6, v_eMomentum6, 4);
                vfloat64m1_t v_um7 = __riscv_vfadd_vv_f64m1(v_p7, v_eMomentum7, 4);

//                FP3 t = magneticFieldValue * eCoeff;
                vfloat64m1_t v_t0 = __riscv_vfmul_vf_f64m1(v_magneticFieldValue, eCoeff0, 4);
                vfloat64m1_t v_t1 = __riscv_vfmul_vf_f64m1(v_magneticFieldValue, eCoeff1, 4);
                vfloat64m1_t v_t2 = __riscv_vfmul_vf_f64m1(v_magneticFieldValue, eCoeff2, 4);
                vfloat64m1_t v_t3 = __riscv_vfmul_vf_f64m1(v_magneticFieldValue, eCoeff3, 4);
                vfloat64m1_t v_t4 = __riscv_vfmul_vf_f64m1(v_magneticFieldValue, eCoeff4, 4);
                vfloat64m1_t v_t5 = __riscv_vfmul_vf_f64m1(v_magneticFieldValue, eCoeff5, 4);
                vfloat64m1_t v_t6 = __riscv_vfmul_vf_f64m1(v_magneticFieldValue, eCoeff6, 4);
                vfloat64m1_t v_t7 = __riscv_vfmul_vf_f64m1(v_magneticFieldValue, eCoeff7, 4);


//                FP3 uprime = um + pica::cross(um, t);
                vfloat64m1_t v_cross_um_t0 = __riscv_vfcross_vv_f64m1(v_um0, v_t0, 4);
                vfloat64m1_t v_cross_um_t1 = __riscv_vfcross_vv_f64m1(v_um1, v_t1, 4);
                vfloat64m1_t v_cross_um_t2 = __riscv_vfcross_vv_f64m1(v_um2, v_t2, 4);
                vfloat64m1_t v_cross_um_t3 = __riscv_vfcross_vv_f64m1(v_um3, v_t3, 4);
                vfloat64m1_t v_cross_um_t4 = __riscv_vfcross_vv_f64m1(v_um4, v_t4, 4);
                vfloat64m1_t v_cross_um_t5 = __riscv_vfcross_vv_f64m1(v_um5, v_t5, 4);
                vfloat64m1_t v_cross_um_t6 = __riscv_vfcross_vv_f64m1(v_um6, v_t6, 4);
                vfloat64m1_t v_cross_um_t7 = __riscv_vfcross_vv_f64m1(v_um7, v_t7, 4);

                vfloat64m1_t v_uprime0 = __riscv_vfadd_vv_f64m1(v_um0, v_cross_um_t0, 4);
                vfloat64m1_t v_uprime1 = __riscv_vfadd_vv_f64m1(v_um1, v_cross_um_t1, 4);
                vfloat64m1_t v_uprime2 = __riscv_vfadd_vv_f64m1(v_um2, v_cross_um_t2, 4);
                vfloat64m1_t v_uprime3 = __riscv_vfadd_vv_f64m1(v_um3, v_cross_um_t3, 4);
                vfloat64m1_t v_uprime4 = __riscv_vfadd_vv_f64m1(v_um4, v_cross_um_t4, 4);
                vfloat64m1_t v_uprime5 = __riscv_vfadd_vv_f64m1(v_um5, v_cross_um_t5, 4);
                vfloat64m1_t v_uprime6 = __riscv_vfadd_vv_f64m1(v_um6, v_cross_um_t6, 4);
                vfloat64m1_t v_uprime7 = __riscv_vfadd_vv_f64m1(v_um7, v_cross_um_t7, 4);


//                FP3 s = t * ((double)2.0 / ((double)1.0 + t.norm2()));
                double tt[8];
                tt[0] = inv_one_add_norm2_m(redo_m, v_two, v_one, v_t0);
                tt[1] = inv_one_add_norm2_m(redo_m, v_two, v_one, v_t1);
                tt[2] = inv_one_add_norm2_m(redo_m, v_two, v_one, v_t2);
                tt[3] = inv_one_add_norm2_m(redo_m, v_two, v_one, v_t3);
                tt[4] = inv_one_add_norm2_m(redo_m, v_two, v_one, v_t4);
                tt[5] = inv_one_add_norm2_m(redo_m, v_two, v_one, v_t5);
                tt[6] = inv_one_add_norm2_m(redo_m, v_two, v_one, v_t6);
                tt[7] = inv_one_add_norm2_m(redo_m, v_two, v_one, v_t7);
                
                vfloat64m1_t v_s0 = __riscv_vfmul_vf_f64m1(v_t0, tt[0], 4);
                vfloat64m1_t v_s1 = __riscv_vfmul_vf_f64m1(v_t1, tt[1], 4);
                vfloat64m1_t v_s2 = __riscv_vfmul_vf_f64m1(v_t2, tt[2], 4);
                vfloat64m1_t v_s3 = __riscv_vfmul_vf_f64m1(v_t3, tt[3], 4);
                vfloat64m1_t v_s4 = __riscv_vfmul_vf_f64m1(v_t4, tt[4], 4);
                vfloat64m1_t v_s5 = __riscv_vfmul_vf_f64m1(v_t5, tt[5], 4);
                vfloat64m1_t v_s6 = __riscv_vfmul_vf_f64m1(v_t6, tt[6], 4);
                vfloat64m1_t v_s7 = __riscv_vfmul_vf_f64m1(v_t7, tt[7], 4);

//              particles[i].p = um + pica::cross(uprime, s) + eMomentum;
                vfloat64m1_t v_cross_uprime_s0 = __riscv_vfcross_vv_f64m1(v_uprime0, v_s0, 4);
                vfloat64m1_t v_cross_uprime_s1 = __riscv_vfcross_vv_f64m1(v_uprime1, v_s1, 4);
                vfloat64m1_t v_cross_uprime_s2 = __riscv_vfcross_vv_f64m1(v_uprime2, v_s2, 4);
                vfloat64m1_t v_cross_uprime_s3 = __riscv_vfcross_vv_f64m1(v_uprime3, v_s3, 4);
                vfloat64m1_t v_cross_uprime_s4 = __riscv_vfcross_vv_f64m1(v_uprime4, v_s4, 4);
                vfloat64m1_t v_cross_uprime_s5 = __riscv_vfcross_vv_f64m1(v_uprime5, v_s5, 4);
                vfloat64m1_t v_cross_uprime_s6 = __riscv_vfcross_vv_f64m1(v_uprime6, v_s6, 4);
                vfloat64m1_t v_cross_uprime_s7 = __riscv_vfcross_vv_f64m1(v_uprime7, v_s7, 4);

                vfloat64m1_t v_tt0 = __riscv_vfadd_vv_f64m1(v_cross_uprime_s0, v_eMomentum0, 4);
                vfloat64m1_t v_tt1 = __riscv_vfadd_vv_f64m1(v_cross_uprime_s1, v_eMomentum1, 4);
                vfloat64m1_t v_tt2 = __riscv_vfadd_vv_f64m1(v_cross_uprime_s2, v_eMomentum2, 4);
                vfloat64m1_t v_tt3 = __riscv_vfadd_vv_f64m1(v_cross_uprime_s3, v_eMomentum3, 4);
                vfloat64m1_t v_tt4 = __riscv_vfadd_vv_f64m1(v_cross_uprime_s4, v_eMomentum4, 4);
                vfloat64m1_t v_tt5 = __riscv_vfadd_vv_f64m1(v_cross_uprime_s5, v_eMomentum5, 4);
                vfloat64m1_t v_tt6 = __riscv_vfadd_vv_f64m1(v_cross_uprime_s6, v_eMomentum6, 4);
                vfloat64m1_t v_tt7 = __riscv_vfadd_vv_f64m1(v_cross_uprime_s7, v_eMomentum7, 4);

                vfloat64m1_t v_res_p0 = __riscv_vfadd_vv_f64m1(v_um0, v_tt0, 4);
                vfloat64m1_t v_res_p1 = __riscv_vfadd_vv_f64m1(v_um1, v_tt1, 4);
                vfloat64m1_t v_res_p2 = __riscv_vfadd_vv_f64m1(v_um2, v_tt2, 4);
                vfloat64m1_t v_res_p3 = __riscv_vfadd_vv_f64m1(v_um3, v_tt3, 4);
                vfloat64m1_t v_res_p4 = __riscv_vfadd_vv_f64m1(v_um4, v_tt4, 4);
                vfloat64m1_t v_res_p5 = __riscv_vfadd_vv_f64m1(v_um5, v_tt5, 4);
                vfloat64m1_t v_res_p6 = __riscv_vfadd_vv_f64m1(v_um6, v_tt6, 4);
                vfloat64m1_t v_res_p7 = __riscv_vfadd_vv_f64m1(v_um7, v_tt7, 4);

                __riscv_vse64_v_f64m1_m(redo_m, &(particles[i+0].p.x), v_res_p0, 4);
                __riscv_vse64_v_f64m1_m(redo_m, &(particles[i+1].p.x), v_res_p1, 4);
                __riscv_vse64_v_f64m1_m(redo_m, &(particles[i+2].p.x), v_res_p2, 4);
                __riscv_vse64_v_f64m1_m(redo_m, &(particles[i+3].p.x), v_res_p3, 4);
                __riscv_vse64_v_f64m1_m(redo_m, &(particles[i+4].p.x), v_res_p4, 4);
                __riscv_vse64_v_f64m1_m(redo_m, &(particles[i+5].p.x), v_res_p5, 4);
                __riscv_vse64_v_f64m1_m(redo_m, &(particles[i+6].p.x), v_res_p6, 4);
                __riscv_vse64_v_f64m1_m(redo_m, &(particles[i+7].p.x), v_res_p7, 4);

//                particles[i].position += particles[i].p * dt;
                vfloat64m1_t v_position0  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+0].position.x), 4);
                vfloat64m1_t v_position1  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+1].position.x), 4);
                vfloat64m1_t v_position2  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+2].position.x), 4);
                vfloat64m1_t v_position3  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+3].position.x), 4);
                vfloat64m1_t v_position4  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+4].position.x), 4);
                vfloat64m1_t v_position5  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+5].position.x), 4);
                vfloat64m1_t v_position6  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+6].position.x), 4);
                vfloat64m1_t v_position7  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i+7].position.x), 4);

                v_position0 = __riscv_vfmacc_vf_f64m1(v_position0, dt, v_res_p0, 4);
                v_position1 = __riscv_vfmacc_vf_f64m1(v_position1, dt, v_res_p1, 4);
                v_position2 = __riscv_vfmacc_vf_f64m1(v_position2, dt, v_res_p2, 4);
                v_position3 = __riscv_vfmacc_vf_f64m1(v_position3, dt, v_res_p3, 4);
                v_position4 = __riscv_vfmacc_vf_f64m1(v_position4, dt, v_res_p4, 4);
                v_position5 = __riscv_vfmacc_vf_f64m1(v_position5, dt, v_res_p5, 4);
                v_position6 = __riscv_vfmacc_vf_f64m1(v_position6, dt, v_res_p6, 4);
                v_position7 = __riscv_vfmacc_vf_f64m1(v_position7, dt, v_res_p7, 4);

                __riscv_vse64_v_f64m1_m(redo_m ,&(particles[i+0].position.x), v_position0, 4);
                __riscv_vse64_v_f64m1_m(redo_m ,&(particles[i+1].position.x), v_position1, 4);
                __riscv_vse64_v_f64m1_m(redo_m ,&(particles[i+2].position.x), v_position2, 4);
                __riscv_vse64_v_f64m1_m(redo_m ,&(particles[i+3].position.x), v_position3, 4);
                __riscv_vse64_v_f64m1_m(redo_m ,&(particles[i+4].position.x), v_position4, 4);
                __riscv_vse64_v_f64m1_m(redo_m ,&(particles[i+5].position.x), v_position5, 4);
                __riscv_vse64_v_f64m1_m(redo_m ,&(particles[i+6].position.x), v_position6, 4);
                __riscv_vse64_v_f64m1_m(redo_m ,&(particles[i+7].position.x), v_position7, 4);


            }
        }
    }
}
