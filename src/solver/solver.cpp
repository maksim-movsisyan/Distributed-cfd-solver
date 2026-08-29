#include "cfd/solver/solver.hpp"

#include "cfd/solver/config.hpp"
#include "cfd/solver/eos/ideal_gas.hpp"
#include "cfd/solver/limiter/limiters.hpp"
#include "cfd/solver/reconstruction/first_order.hpp"
#include "cfd/solver/reconstruction/muscl.hpp"
#include "cfd/solver/riemann/hllc.hpp"

namespace cfd::solver {

// Runtime selection -> compile-time policy instantiation. Adding a scheme
// (EOS, flux, reconstruction, limiter) = one policy header plus one case
// here; every other component stays untouched.
int run_solver(const SolverConfig& cfg,
               const BoundaryConfig& bcfg,
               const mesh::MeshPart& mp,
               const MPI_Comm comm) {
    if (cfg.flow.type == EqOfStateType::IdealGas) {
        const auto eos = cfg.flow.create_ideal_gas();

        switch (cfg.flux) {
            case FluxType::HLLC:
                switch (cfg.reconstruction) {
                    case ReconType::FirstOrder:
                        return Solver<eos::IdealGas, riemann::HllcFlux,
                                      recon::FirstOrder>(cfg, bcfg, eos, mp, comm).run();
                    case ReconType::Muscl:
                        switch (cfg.limiter) {
                            case LimiterType::Venkatakrishnan:
                                return Solver<eos::IdealGas, riemann::HllcFlux,
                                              recon::Muscl<limiter::Venkatakrishnan>>(
                                              cfg, bcfg, eos, mp, comm).run();
                            case LimiterType::BarthJespersen:
                                return Solver<eos::IdealGas, riemann::HllcFlux,
                                              recon::Muscl<limiter::BarthJespersen>>(
                                              cfg, bcfg, eos, mp, comm).run();
                            case LimiterType::VanAlbada:
                                return Solver<eos::IdealGas, riemann::HllcFlux,
                                              recon::Muscl<limiter::VanAlbada>>(
                                              cfg, bcfg, eos, mp, comm).run();
                        }
                        return 1;
                }
                return 1;
        }
    }

    return 1;
}

} // namespace cfd::solver
