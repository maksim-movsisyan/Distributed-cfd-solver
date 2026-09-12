#include "cfd/solver/solver.hpp"

#include "cfd/numerics/limiter/slope_limiters.hpp"
#include "cfd/numerics/limiter/tvd_1d_limiters.hpp"
#include "cfd/numerics/reconstruction/first_order.hpp"
#include "cfd/numerics/reconstruction/muscl.hpp"
#include "cfd/numerics/reconstruction/muscl_directional.hpp"

#include "cfd/solver/config.hpp"
#include "cfd/solver/eos/ideal_gas.hpp"
#include "cfd/solver/physics/stack.hpp"
#include "cfd/solver/physics/viscous_flow.hpp"
#include "cfd/solver/physics/inviscid_flow.hpp"
#include "cfd/solver/fluxes/hllc.hpp"
#include "cfd/solver/time/forward_euler.hpp"
#include "cfd/solver/time/implicit_euler.hpp"
#include "cfd/solver/time/time_mode.hpp"
//#include "cfd/solver/turbulence/spalart_allmaras.hpp"

namespace cfd::solver {

// dispatch time integration scheme
template <typename EosType, typename FluxType, typename ReconType, typename PhysType, typename TimeModeType>
int dispatch_time_scheme(const SolverConfig& cfg,
                         const bc::BoundaryConfig& bcfg,
                         const mesh::MeshPart& mp,
                         const MPI_Comm comm,
                         const EosType& eos,
                         const PhysType& phys) {
    switch (cfg.scheme) {
        case TimeScheme::ForwardEuler:
            return Solver<EosType, FluxType, ReconType, PhysType, time::ForwardEuler, TimeModeType>(
                cfg, bcfg, eos, phys, mp, comm).run();

        case TimeScheme::BackwardEuler:
            return Solver<EosType, FluxType, ReconType, PhysType, time::BackwardEuler, TimeModeType>(
                cfg, bcfg, eos, phys, mp, comm).run();

        default:
            mpi::fatal(comm, "dispatch: unknown time scheme");
            return 1;
    }
}

// dispatch time mode 
template <typename EosType, typename FluxType, typename ReconType, typename PhysType>
int dispatch_time_mode(const SolverConfig& cfg,
                       const bc::BoundaryConfig& bcfg,
                       const mesh::MeshPart& mp,
                       const MPI_Comm comm,
                       const EosType& eos,
                       const PhysType& phys) {

    switch (cfg.time_mode) {
        case TimeMode::Steady:
            return dispatch_time_scheme<EosType, FluxType, ReconType, PhysType, time::SteadyMode>(
                cfg, bcfg, mp, comm, eos, phys);

        case TimeMode::DualTime:  
            if (cfg.bdf_order == 2) {
                using Mode = time::DualTimeMode<2>;
                return dispatch_time_scheme<EosType, FluxType, ReconType, PhysType, Mode>(
                    cfg, bcfg, mp, comm, eos, phys);
            } else if (cfg.bdf_order == 1) {
                using Mode = time::DualTimeMode<1>;
                return dispatch_time_scheme<EosType, FluxType, ReconType, PhysType, Mode>(
                    cfg, bcfg, mp, comm, eos, phys);
            } else {
                mpi::fatal(comm, "dispatch: invalid bdf_order (available: 1, 2)");
                return 1;
            }

        default:
            mpi::fatal(comm, "dispatch: unknown time mode");
            return 1;
    }
}

// dispatch multidim limiter type
template <typename EosType, typename FluxType, typename PhysType,
          template <typename> class MultidimRecon>
int dispatch_multidim_limiter(const SolverConfig& cfg,
                              const bc::BoundaryConfig& bcfg,
                              const mesh::MeshPart& mp,
                              const MPI_Comm comm,
                              const EosType& eos,
                              const PhysType& phys) {
    switch (cfg.limiter) {
        case LimiterType::Venkatakrishnan:
            return dispatch_time_mode<EosType, FluxType,
                                        MultidimRecon<numerics::limiter::Venkatakrishnan>, PhysType>(
                cfg, bcfg, mp, comm, eos, phys);

        case LimiterType::BarthJespersen:
            return dispatch_time_mode<EosType, FluxType,
                                        MultidimRecon<numerics::limiter::BarthJespersen>, PhysType>(
                cfg, bcfg, mp, comm, eos, phys);

        case LimiterType::VanAlbada:
            return dispatch_time_mode<EosType, FluxType,
                                        MultidimRecon<numerics::limiter::VanAlbada>, PhysType>(
                cfg, bcfg, mp, comm, eos, phys);

        default:
            mpi::fatal(comm, "dispatch: invalid multidimensional limiter type");
            return 1;
    }
}

// dispatch 1d limiter type
template <typename EosType, typename FluxType, typename PhysType,
          template <typename> class DirectionalRecon>
int dispatch_directional_limiter(const SolverConfig& cfg,
                                 const bc::BoundaryConfig& bcfg,
                                 const mesh::MeshPart& mp,
                                 const MPI_Comm comm,
                                 const EosType& eos,
                                 const PhysType& phys) {
    switch (cfg.limiter) {
        case LimiterType::Minmod1D:
            return dispatch_time_mode<EosType, FluxType,
                                        DirectionalRecon<numerics::limiter::tvd1d::Minmod>, PhysType>(
                cfg, bcfg, mp, comm, eos, phys);

        case LimiterType::VanAlbada1D:
            return dispatch_time_mode<EosType, FluxType,
                                        DirectionalRecon<numerics::limiter::tvd1d::VanAlbada>, PhysType>(
                cfg, bcfg, mp, comm, eos, phys);

        default:
            mpi::fatal(comm, "dispatch: invalid directional 1D limiter type");
            return 1;
    }
}

// dispatch reconstruction type
template <typename EosType, typename FluxType, typename PhysType>
int dispatch_reconstruction(const SolverConfig& cfg,
                            const bc::BoundaryConfig& bcfg,
                            const mesh::MeshPart& mp,
                            const MPI_Comm comm,
                            const EosType& eos,
                            const PhysType& phys) {
    switch (cfg.reconstruction) {
        case ReconType::FirstOrder:
            return dispatch_time_mode<EosType, FluxType, numerics::recon::FirstOrder, PhysType>(
                cfg, bcfg, mp, comm, eos, phys);

        case ReconType::Muscl:
            return dispatch_multidim_limiter<EosType, FluxType, PhysType, numerics::recon::Muscl>(
                cfg, bcfg, mp, comm, eos, phys);

        case ReconType::MusclDirectional:
            return dispatch_directional_limiter<EosType, FluxType, PhysType, numerics::recon::MusclDirectional>(
                cfg, bcfg, mp, comm, eos, phys);

        default:
            mpi::fatal(comm, "dispatch: unknown reconstruction scheme");
            return 1;
    }
}

// dispatch riemann solver type
template <typename EosType, typename PhysType>
int dispatch_flux(const SolverConfig& cfg,
                  const bc::BoundaryConfig& bcfg,
                  const mesh::MeshPart& mp,
                  const MPI_Comm comm,
                  const EosType& eos,
                  const PhysType& phys) {
    switch (cfg.flux) {
        case FluxType::HLLC:
            return dispatch_reconstruction<EosType, fluxes::HllcFlux, PhysType>(
                cfg, bcfg, mp, comm, eos, phys);

        // case FluxType::Roe:
        //     return dispatch_reconstruction<EosType, fluxes::RoeFlux, PhysType>(
        //         cfg, bcfg, mp, comm, eos, phys);

        // case FluxType::Rusanov:
        //     return dispatch_reconstruction<EosType, fluxes::RusanovFlux, PhysType>(
        //         cfg, bcfg, mp, comm, eos, phys);

        default:
            mpi::fatal(comm, "dispatch: unknown flux scheme");
            return 1;
    }
}

// dispatch physics: equation set + turbulence module (compile-time stack)
template <typename EosType>
int dispatch_physics(const SolverConfig& cfg,
                     const bc::BoundaryConfig& bcfg,
                     const mesh::MeshPart& mp,
                     const MPI_Comm comm,
                     const EosType& eos) {
    if (cfg.turbulence.enabled) {
        //using SaStack = physics::PhysicsStack<physics::ViscousFlow, turb::SpalartAllmaras>;

        //turb::SpalartAllmaras sa{};
        //sa.nu_inf_ratio = cfg.turbulence.nu_inf_ratio;
        //sa.max_distance_sweeps = cfg.turbulence.max_distance_sweeps;
        //sa.distance_tolerance = cfg.turbulence.distance_tolerance;

        //const SaStack phys{physics::ViscousFlow{cfg.prandtl}, std::tuple{sa}};
        //return dispatch_flux<EosType, SaStack>(cfg, bcfg, mp, comm, eos, phys);
        mpi::fatal(comm, "dispatch: turbulence not done yet :) ");
    }

    if (cfg.flow_model == FlowModel::ViscousFlow) {
        const physics::PhysicsStack<physics::ViscousFlow> phys{
            physics::ViscousFlow{cfg.prandtl}, {}
        };
        return dispatch_flux<EosType, physics::PhysicsStack<physics::ViscousFlow>>(
            cfg, bcfg, mp, comm, eos, phys);

    } else if (cfg.flow_model == FlowModel::InviscidFlow) {
        const physics::PhysicsStack<physics::InviscidFlow> phys{
            physics::InviscidFlow{}, {}
        };
        return dispatch_flux<EosType, physics::PhysicsStack<physics::InviscidFlow>>(
            cfg, bcfg, mp, comm, eos, phys);
    } else {
        mpi::fatal(comm, "dispatch: unknown Flow Model model");
        return 1;
    }
}

int run_solver(const SolverConfig& cfg,
               const bc::BoundaryConfig& bcfg,
               const mesh::MeshPart& mp,
               const MPI_Comm comm) {
    switch (cfg.flow.type) {
        case EqOfStateType::IdealGas: {
            const auto eos = cfg.flow.create_ideal_gas();
            return dispatch_physics<eos::IdealGas>(cfg, bcfg, mp, comm, eos);
        }

        // case EqOfStateType::RealGas: {
        //     const auto eos = cfg.flow.create_real_gas();
        //     return dispatch_physics<eos::RealGas>(cfg, bcfg, mp, comm, eos);
        // }

        default:
            mpi::fatal(comm, "dispatch: unknown EOS model");
            return 1;
    }
}

} // namespace cfd::solver
