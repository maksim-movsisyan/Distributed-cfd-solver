#include "cfd/solver/solver.hpp"

#include "cfd/solver/config.hpp"
#include "cfd/solver/eos/ideal_gas.hpp"
#include "cfd/solver/reiman/hllc.hpp"

namespace cfd::solver {

int run_solver(const SolverConfig& cfg,
               const BoundaryConfig& bcfg,
               const mesh::MeshPart& mp,
               const MPI_Comm comm) {
    if (cfg.flow.type == EqOfStateType::IdealGas) {
        const auto eos = cfg.flow.create_ideal_gas();

        switch (cfg.flux) {
            case FluxType::HLLC:
                return Solver<eos::IdealGas, riemann::HllcFlux>(cfg, bcfg, eos, mp, comm).run();
        }
    }

    return 1;
}

} // namespace cfd::solver