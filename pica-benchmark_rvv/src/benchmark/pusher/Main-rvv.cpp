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


vfloat64m1_t one_add_norm2(const vfloat64m1_t &one,
                           const vfloat64m1_t &x,
                           const vfloat64m1_t &y,
                           const vfloat64m1_t &z,
                           int bs)
{
    vfloat64m1_t res;
    res = __riscv_vfmacc_vv_f64m1(one, x, x, bs);
    res = __riscv_vfmacc_vv_f64m1(res, y, y, bs);
    res = __riscv_vfmacc_vv_f64m1(res, z, z, bs);
    return res;
}

void cross(vfloat64m1_t &rx, vfloat64m1_t &ry, vfloat64m1_t &rz,
           const vfloat64m1_t &x1, const vfloat64m1_t &y1, const vfloat64m1_t &z1, 
           const vfloat64m1_t &x2, const vfloat64m1_t &y2, const vfloat64m1_t &z2, 
           int bs)
{
    rx = __riscv_vfmul_vv_f64m1(y1, z2, bs);
    rx = __riscv_vfmsac_vv_f64m1(rx, z1, y2, bs);
    
    ry = __riscv_vfmul_vv_f64m1(z1, x2, bs);
    ry = __riscv_vfmsac_vv_f64m1(ry, x1, z2, bs);

    rz = __riscv_vfmul_vv_f64m1(x1, y2, bs);
    rz = __riscv_vfmsac_vv_f64m1(rz, y1, x2,bs);
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

            int bs = __riscv_vsetvl_e64m1(endIdx);
            int type_stride = (uint64_t)(&(particles[1].typeIndex)) - (uint64_t)(&(particles[0].typeIndex));
            int p_stride = (uint64_t)(&(particles[1].p.x)) - (uint64_t)(&(particles[0].p.x));
            int position_stride = (uint64_t)(&(particles[1].position.x)) - (uint64_t)(&(particles[0].position.x));
			
            vfloat64m1_t v_electricFieldValue_x;
            vfloat64m1_t v_electricFieldValue_y;
            vfloat64m1_t v_electricFieldValue_z;
            v_electricFieldValue_x = __riscv_vfmv_v_f_f64m1(electricFieldValue.x, bs);
            v_electricFieldValue_y = __riscv_vfmv_v_f_f64m1(electricFieldValue.y, bs);
            v_electricFieldValue_z = __riscv_vfmv_v_f_f64m1(electricFieldValue.z, bs);
      
            vfloat64m1_t v_magneticFieldValue_x;
            vfloat64m1_t v_magneticFieldValue_y;
            vfloat64m1_t v_magneticFieldValue_z;
            v_magneticFieldValue_x = __riscv_vfmv_v_f_f64m1(magneticFieldValue.x, bs);
            v_magneticFieldValue_y = __riscv_vfmv_v_f_f64m1(magneticFieldValue.y, bs);
            v_magneticFieldValue_z = __riscv_vfmv_v_f_f64m1(magneticFieldValue.z, bs);
            
            vfloat64m1_t v_two  = __riscv_vfmv_v_f_f64m1(2.0, bs);
            vfloat64m1_t v_one  = __riscv_vfmv_v_f_f64m1(1.0, bs);
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, bs);
			
            for (int i = beginIdx; i < endIdx /*&& i < bs * 2*/; i+=bs) {
                // The code below uses precomputed coefficient:
                // eCoeff = q * dt / (2 * m * c)
                //FP eCoeff = coeff[particles[i].typeIndex];
                vfloat64m1_t v_coeff;
                vuint64m1_t index_coeff = __riscv_vlse64_v_u64m1(&(particles[i].typeIndex), type_stride, bs);
                index_coeff = __riscv_vsll_vx_u64m1(index_coeff, 3, bs); 
                v_coeff = __riscv_vluxei64_v_f64m1(coeff.data(), index_coeff, bs);
				
                //FP3 eMomentum = electricFieldValue * eCoeff;
                vfloat64m1_t v_eMomentum_x = __riscv_vfmul_vv_f64m1(v_electricFieldValue_x, v_coeff, bs);
                vfloat64m1_t v_eMomentum_y = __riscv_vfmul_vv_f64m1(v_electricFieldValue_y, v_coeff, bs);
                vfloat64m1_t v_eMomentum_z = __riscv_vfmul_vv_f64m1(v_electricFieldValue_z, v_coeff, bs);

                //FP3 um = particles[i].p + eMomentum;
                vfloat64m1_t v_p_x = __riscv_vlse64_v_f64m1(&(particles[i].p.x), p_stride, bs);
                vfloat64m1_t v_p_y = __riscv_vlse64_v_f64m1(&(particles[i].p.y), p_stride, bs);
                vfloat64m1_t v_p_z = __riscv_vlse64_v_f64m1(&(particles[i].p.z), p_stride, bs);

                vfloat64m1_t v_um_x = __riscv_vfadd_vv_f64m1(v_p_x, v_eMomentum_x, bs);
                vfloat64m1_t v_um_y = __riscv_vfadd_vv_f64m1(v_p_y, v_eMomentum_y, bs);
                vfloat64m1_t v_um_z = __riscv_vfadd_vv_f64m1(v_p_z, v_eMomentum_z, bs);


                //FP3 t = magneticFieldValue * eCoeff;
                vfloat64m1_t v_t_x = __riscv_vfmul_vv_f64m1(v_magneticFieldValue_x, v_coeff, bs);
                vfloat64m1_t v_t_y = __riscv_vfmul_vv_f64m1(v_magneticFieldValue_y, v_coeff, bs);
                vfloat64m1_t v_t_z = __riscv_vfmul_vv_f64m1(v_magneticFieldValue_z, v_coeff, bs);

                //FP3 uprime = um + pica::cross(um, t);
                vfloat64m1_t v_cross_um_t_x;
                vfloat64m1_t v_cross_um_t_y;
                vfloat64m1_t v_cross_um_t_z;

                cross(v_cross_um_t_x, v_cross_um_t_y, v_cross_um_t_z,
                      v_um_x, v_um_y, v_um_z, 
                      v_t_x, v_t_y, v_t_z, 
                      bs);
				
                vfloat64m1_t v_uprime_x = __riscv_vfadd_vv_f64m1(v_um_x, v_cross_um_t_x, bs);
                vfloat64m1_t v_uprime_y = __riscv_vfadd_vv_f64m1(v_um_y, v_cross_um_t_y, bs);
                vfloat64m1_t v_uprime_z = __riscv_vfadd_vv_f64m1(v_um_z, v_cross_um_t_z, bs);
				
                //FP3 s = t * ((double)2.0 / ((double)1.0 + t.norm2()));
                vfloat64m1_t v_t_norm2  = one_add_norm2(v_one, v_t_x, v_t_y, v_t_z, bs);
                vfloat64m1_t v_tt = __riscv_vfdiv_vv_f64m1(v_two, v_t_norm2, bs);
				
                vfloat64m1_t v_s_x = __riscv_vfmul_vv_f64m1(v_t_x, v_tt, bs);
                vfloat64m1_t v_s_y = __riscv_vfmul_vv_f64m1(v_t_y, v_tt, bs);
                vfloat64m1_t v_s_z = __riscv_vfmul_vv_f64m1(v_t_z, v_tt, bs);

                //particles[i].p = um + pica::cross(uprime, s) + eMomentum;
                vfloat64m1_t v_cross_uprime_s_x;
                vfloat64m1_t v_cross_uprime_s_y;
                vfloat64m1_t v_cross_uprime_s_z;

                cross(v_cross_uprime_s_x, v_cross_uprime_s_y, v_cross_uprime_s_z,
                      v_uprime_x, v_uprime_y, v_uprime_z, 
                      v_s_x, v_s_y, v_s_z, 
                      bs);
					  
                vfloat64m1_t v_res_p_x = __riscv_vfadd_vv_f64m1(v_um_x, v_cross_uprime_s_x, bs);
                vfloat64m1_t v_res_p_y = __riscv_vfadd_vv_f64m1(v_um_y, v_cross_uprime_s_y, bs);
                vfloat64m1_t v_res_p_z = __riscv_vfadd_vv_f64m1(v_um_z, v_cross_uprime_s_z, bs);
				
                v_res_p_x = __riscv_vfadd_vv_f64m1(v_res_p_x, v_eMomentum_x, bs);
                v_res_p_y = __riscv_vfadd_vv_f64m1(v_res_p_y, v_eMomentum_y, bs);
                v_res_p_z = __riscv_vfadd_vv_f64m1(v_res_p_z, v_eMomentum_z, bs);

                __riscv_vsse64_v_f64m1(&(particles[i].p.x), p_stride, v_res_p_x, bs);
                __riscv_vsse64_v_f64m1(&(particles[i].p.y), p_stride, v_res_p_y, bs);
                __riscv_vsse64_v_f64m1(&(particles[i].p.z), p_stride, v_res_p_z, bs);

                //particles[i].position += particles[i].p * dt;
                vfloat64m1_t v_position_x = __riscv_vlse64_v_f64m1(&(particles[i].position.x), position_stride, bs);
                vfloat64m1_t v_position_y = __riscv_vlse64_v_f64m1(&(particles[i].position.y), position_stride, bs);
                vfloat64m1_t v_position_z = __riscv_vlse64_v_f64m1(&(particles[i].position.z), position_stride, bs);

                v_position_x = __riscv_vfmacc_vf_f64m1(v_position_x, dt, v_res_p_x, bs);
                v_position_y = __riscv_vfmacc_vf_f64m1(v_position_y, dt, v_res_p_y, bs);
                v_position_z = __riscv_vfmacc_vf_f64m1(v_position_z, dt, v_res_p_z, bs);
				
                __riscv_vsse64_v_f64m1(&(particles[i].position.x), position_stride, v_position_x, bs);
                __riscv_vsse64_v_f64m1(&(particles[i].position.y), position_stride, v_position_y, bs);
                __riscv_vsse64_v_f64m1(&(particles[i].position.z), position_stride, v_position_z, bs);
            }
        }
    }
}
