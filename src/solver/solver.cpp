#include "cfd/solver/solver.hpp"

#include "cfd/solver/config.hpp"
#include "cfd/solver/eos/ideal_gas.hpp"
#include "cfd/solver/limiter/limiters.hpp"
#include "cfd/solver/reconstruction/first_order.hpp"
#include "cfd/solver/reconstruction/muscl.hpp"
#include "cfd/solver/reconstruction/muscl_directional.hpp"
#include "cfd/solver/riemann/hllc.hpp"

namespace cfd::solver {

// dispatch multidim limiter type
template <typename EosType, typename FluxType, template <typename> class MultidimRecon>
int dispatch_multidim_limiter(const SolverConfig& cfg,
                              const BoundaryConfig& bcfg,
                              const mesh::MeshPart& mp,
                              const MPI_Comm comm,
                              const EosType& eos) {
    switch (cfg.limiter) {
        case LimiterType::Venkatakrishnan:
            return Solver<EosType, FluxType, MultidimRecon<limiter::Venkatakrishnan>>(
                cfg, bcfg, eos, mp, comm).run();

        case LimiterType::BarthJespersen:
            return Solver<EosType, FluxType, MultidimRecon<limiter::BarthJespersen>>(
                cfg, bcfg, eos, mp, comm).run();

        case LimiterType::VanAlbada:
            return Solver<EosType, FluxType, MultidimRecon<limiter::VanAlbada>>(
                cfg, bcfg, eos, mp, comm).run();

        default:
            mpi::fatal(comm, "dispatch: invalid multidimensional limiter type");
            return 1;
    }
}

// dispatch 1d limiter type
template <typename EosType, typename FluxType, template <typename> class DirectionalRecon>
int dispatch_directional_limiter(const SolverConfig& cfg,
                                const BoundaryConfig& bcfg,
                                const mesh::MeshPart& mp,
                                const MPI_Comm comm,
                                const EosType& eos) {
    switch (cfg.limiter) {
        case LimiterType::Minmod1D:
            return Solver<EosType, FluxType, DirectionalRecon<limiter::Minmod1D>>(
                cfg, bcfg, eos, mp, comm).run();

        case LimiterType::VanAlbada1D:
            return Solver<EosType, FluxType, DirectionalRecon<limiter::VanAlbada1D>>(
                cfg, bcfg, eos, mp, comm).run();

        default:
            mpi::fatal(comm, "dispatch: invalid directional 1D limiter type");
            return 1;
    }
}

// dispatch reconstruction type
template <typename EosType, typename FluxType>
int dispatch_reconstruction(const SolverConfig& cfg,
                            const BoundaryConfig& bcfg,
                            const mesh::MeshPart& mp,
                            const MPI_Comm comm,
                            const EosType& eos) {
    switch (cfg.reconstruction) {
        case ReconType::FirstOrder:
            return Solver<EosType, FluxType, recon::FirstOrder>(
                cfg, bcfg, eos, mp, comm).run();

        case ReconType::Muscl:
            return dispatch_multidim_limiter<EosType, FluxType, recon::Muscl>(
                cfg, bcfg, mp, comm, eos);

        case ReconType::MusclDirectional:
            return dispatch_directional_limiter<EosType, FluxType, recon::MusclDirectional>(
                cfg, bcfg, mp, comm, eos);

        default:
            mpi::fatal(comm, "dispatch: unknown reconstruction scheme");
            return 1;
    }
}

// dispatch riemann solver type
template <typename EosType>
int dispatch_flux(const SolverConfig& cfg,
                  const BoundaryConfig& bcfg,
                  const mesh::MeshPart& mp,
                  const MPI_Comm comm,
                  const EosType& eos) {
    switch (cfg.flux) {
        case FluxType::HLLC:
            return dispatch_reconstruction<EosType, riemann::HllcFlux>(
                cfg, bcfg, mp, comm, eos);

        // case FluxType::Roe:
        //     return dispatch_reconstruction<EosType, riemann::RoeFlux>(
        //         cfg, bcfg, mp, comm, eos);

        // case FluxType::Rusanov:
        //     return dispatch_reconstruction<EosType, riemann::RusanovFlux>(
        //         cfg, bcfg, mp, comm, eos);

        default:
            mpi::fatal(comm, "dispatch: unknown flux scheme");
            return 1;
    }
}

int run_solver(const SolverConfig& cfg,
               const BoundaryConfig& bcfg,
               const mesh::MeshPart& mp,
               const MPI_Comm comm) {
    switch (cfg.flow.type) {
        case EqOfStateType::IdealGas: {
            const auto eos = cfg.flow.create_ideal_gas();
            return dispatch_flux<eos::IdealGas>(cfg, bcfg, mp, comm, eos);
        }

        // case EqOfStateType::RealGas: {
        //     const auto eos = cfg.flow.create_real_gas();
        //     return dispatch_flux<eos::RealGas>(cfg, bcfg, mp, comm, eos);
        // }

        default:
            mpi::fatal(comm, "dispatch: unknown EOS model");
            return 1;
    }
}

} // namespace cfd::solver
