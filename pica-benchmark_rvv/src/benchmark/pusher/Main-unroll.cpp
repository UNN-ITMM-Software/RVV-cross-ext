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

            for (int i = beginIdx; i < endIdx; i+=4) {
                // The code below uses precomputed coefficient:
                // eCoeff = q * dt / (2 * m * c)
                FP eCoeff0 = coeff[particles[i+0].typeIndex];
                FP eCoeff1 = coeff[particles[i+1].typeIndex];
                FP eCoeff2 = coeff[particles[i+2].typeIndex];
                FP eCoeff3 = coeff[particles[i+3].typeIndex];
 
                FP3 eMomentum0 = electricFieldValue * eCoeff0;
                FP3 eMomentum1 = electricFieldValue * eCoeff1;
                FP3 eMomentum2 = electricFieldValue * eCoeff2;
                FP3 eMomentum3 = electricFieldValue * eCoeff3;


                FP3 um0 = particles[i+0].p + eMomentum0;
                FP3 um1 = particles[i+1].p + eMomentum1;
                FP3 um2 = particles[i+2].p + eMomentum2;
                FP3 um3 = particles[i+3].p + eMomentum3;

                FP3 t0 = magneticFieldValue * eCoeff0;
                FP3 t1 = magneticFieldValue * eCoeff1;
                FP3 t2 = magneticFieldValue * eCoeff2;
                FP3 t3 = magneticFieldValue * eCoeff3;

                FP3 uprime0 = um0 + pica::cross(um0, t0);
                FP3 uprime1 = um1 + pica::cross(um1, t1);
                FP3 uprime2 = um2 + pica::cross(um2, t2);
                FP3 uprime3 = um3 + pica::cross(um3, t3);

                FP3 s0 = t0 * ((double)2.0 / ((double)1.0 + t0.norm2()));
                FP3 s1 = t1 * ((double)2.0 / ((double)1.0 + t1.norm2()));
                FP3 s2 = t2 * ((double)2.0 / ((double)1.0 + t2.norm2()));
                FP3 s3 = t3 * ((double)2.0 / ((double)1.0 + t3.norm2()));

                particles[i+0].p = um0 + pica::cross(uprime0, s0) + eMomentum0;
                particles[i+1].p = um1 + pica::cross(uprime1, s1) + eMomentum1;
                particles[i+2].p = um2 + pica::cross(uprime2, s2) + eMomentum2;
                particles[i+3].p = um3 + pica::cross(uprime3, s3) + eMomentum3;

                particles[i+0].position += particles[i+0].p * dt;
                particles[i+1].position += particles[i+1].p * dt;
                particles[i+2].position += particles[i+2].p * dt;
                particles[i+3].position += particles[i+3].p * dt;
            }
        }
    }
}
