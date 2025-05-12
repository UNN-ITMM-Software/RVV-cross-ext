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


double inv_one_add_norm2_m(const vbool64_t &redo_m, 
                           const vfloat64m1_t &coeff,
                           const vfloat64m1_t &one, 
                           const vfloat64m1_t &val) {
    vfloat64m1_t v_val2  = __riscv_vfmul_vv_f64m1(val, val, 4);
    vfloat64m1_t v_norm2 = __riscv_vfredosum_vs_f64m1_f64m1_m(redo_m, v_val2, one, 4);
    v_norm2 = __riscv_vfdiv_vv_f64m1(coeff, v_norm2, 4);
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

            for (int i = beginIdx; i < endIdx; i++) {
                // The code below uses precomputed coefficient:
                // eCoeff = q * dt / (2 * m * c)
                FP eCoeff = coeff[particles[i].typeIndex];

//                FP3 eMomentum = electricFieldValue * eCoeff;
                vfloat64m1_t v_eMomentum = __riscv_vfmul_vf_f64m1(v_electricFieldValue, eCoeff, 4);

//                FP3 um = particles[i].p + eMomentum;
                vfloat64m1_t v_p  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i].p.x), 4);
                vfloat64m1_t v_um = __riscv_vfadd_vv_f64m1(v_p, v_eMomentum, 4);

//                FP3 t = magneticFieldValue * eCoeff;
                vfloat64m1_t v_t = __riscv_vfmul_vf_f64m1(v_magneticFieldValue, eCoeff, 4);


//                FP3 uprime = um + pica::cross(um, t);
                vfloat64m1_t v_cross_um_t = __riscv_vfcross_vv_f64m1(v_um, v_t, 4);
                vfloat64m1_t v_uprime     = __riscv_vfadd_vv_f64m1(v_um, v_cross_um_t, 4);


//                FP3 s = t * ((double)2.0 / ((double)1.0 + t.norm2()));
                double tt = inv_one_add_norm2_m(redo_m, v_two, v_one, v_t);
                vfloat64m1_t v_s = __riscv_vfmul_vf_f64m1(v_t, tt, 4);

//              particles[i].p = um + pica::cross(uprime, s) + eMomentum;
                vfloat64m1_t v_cross_uprime_s = __riscv_vfcross_vv_f64m1(v_uprime, v_s, 4);
                vfloat64m1_t v_tt = __riscv_vfadd_vv_f64m1(v_cross_uprime_s, v_eMomentum, 4);
                vfloat64m1_t v_res_p = __riscv_vfadd_vv_f64m1(v_um, v_tt, 4);
                __riscv_vse64_v_f64m1_m(redo_m, &(particles[i].p.x), v_res_p, 4);

//                particles[i].position += particles[i].p * dt;
                vfloat64m1_t v_position  = __riscv_vle64_v_f64m1_m(redo_m, &(particles[i].position.x), 4);
                v_position = __riscv_vfmacc_vf_f64m1(v_position, dt, v_res_p, 4);
                __riscv_vse64_v_f64m1_m(redo_m ,&(particles[i].position.x), v_position, 4);
            }
        }
    }
}
