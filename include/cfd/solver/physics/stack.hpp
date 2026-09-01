// Compile-time physics stack: mean-flow equation set + physics modules.
//
// Composed at compile time (design-01 Q1 decision): every module hook folds to
// a direct inlined call, and an EMPTY module list folds to nothing — the plain
// Euler / laminar Navier-Stokes stacks are behaviorally identical to running
// without any module machinery at all.
//
// The stack satisfies physics::FlowPhysics (it IS the solver's physics
// argument): mean-flow geometry and viscous fluxes delegate to the base, with
// the modules' eddy viscosity threaded through. Module orchestration hooks
// (fields / halo / BCs / kernels / clamps / output) are consumed by the Solver.
#pragma once

#include <mpi.h>

#include <cstddef>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "cfd/io/vtk/vtu.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/config.hpp"
#include "cfd/solver/eos/eos_concept.hpp"
#include "cfd/solver/fields/fields_manager.hpp"
#include "cfd/solver/fields/fields_view.hpp"
#include "cfd/solver/gradient/gradient.hpp"
#include "cfd/solver/halo.hpp"
#include "cfd/solver/physics/physics_concepts.hpp"
#include "cfd/solver/physics/inviscid_flow.hpp"
#include "cfd/solver/physics/viscous_flow.hpp"

namespace cfd::solver::physics {

/**
 * @class PhysicsStack
 * @brief Aggregates a mean-flow equation set (Base) with physics modules.
 *
 * @tparam BaseT    PhysicsGeneral policy: physics::InviscidFlow, physics::ViscousFlow.
 * @tparam Modules  PhysicsGeneral policy: physics modules (e.g. turb::SpalartAllmaras). May be empty.
 */
template <PhysicsGeneral BaseT, PhysicsGeneral... Modules>
class PhysicsStack {

public:
    using BaseType = BaseT;
    
    BaseT base{};
    std::tuple<Modules...> modules{};

    // --- Composition metadata (compile-time OR / sum over modules) -----------
    static constexpr std::size_t kNumVars = BaseT::kNumVars;             // meanflow num variables
    static constexpr bool kHasViscous = BaseT::kHasViscous;
    static constexpr bool kNeedsGradients = BaseT::kNeedsGradients
                                         || (Modules::kNeedsGradients || ...);
    static constexpr std::size_t kNumExtraVars = (Modules::kNumVars + ... + 0);
    static constexpr bool kHasEddyViscosity = (Modules::kHasEddyViscosity || ...);
    static constexpr bool kNeedsFaceMdot = (Modules::kNeedsFaceMdot || ...);
    static constexpr bool kNeedsWallDist = (Modules::kNeedsWallDist || ...);

    /**
     * @brief The mean-flow physics the ResidualKernel instantiates with: the
     *        stack BASE augmented with stack-level service flags (face
     *        mass-flux storage for module convection). The kernel never sees
     *        the full stack — only this base view, so it stays decoupled from
     *        module machinery while remaining compile-time specialized.
     */
    struct KernelPhysics : BaseT {
        static constexpr bool kNeedsFaceMdot = PhysicsStack::kNeedsFaceMdot;
    };

    static_assert(PhysicsGeneral<KernelPhysics>);

    /** @brief The kernel's physics instance (base values + service flags). */
    [[nodiscard]] KernelPhysics kernel_physics() const noexcept {
        return KernelPhysics{base};
    }

    // --- PhysicsGeneral interface: delegating to the mean-flow base -------------
    using Geometry = typename BaseT::Geometry;

    [[nodiscard]] static Geometry build_geometry(const mesh::MeshPart& mp) {
        return BaseT::build_geometry(mp);
    }

    static constexpr const char* name() noexcept { return BaseT::name(); }

    /** @brief Full name including modules (for logs). */
    [[nodiscard]] static std::string full_name() {
        if constexpr (sizeof...(Modules) == 0) {
            return BaseT::name();
        } else {
            std::string s = BaseT::name();
            ((s += '+', s += Modules::name()), ...);
            return s;
        }
    }

    // --- Module orchestration folds -------------------------------------------

    void register_fields(fields::FieldsManager& mgr,
                         const std::size_t n_total, const bool needs_prev_snapshot) {
        std::apply([&](auto&... m) {
            (m.register_fields(mgr, n_total, needs_prev_snapshot), ...);
        }, modules);
    }

    void append_update_slots(std::vector<double*>& u,
                             std::vector<double*>& prev,
                             std::vector<double*>& stage,
                             std::vector<double*>& res,
                             fields::FieldsManager& mgr,
                             const bool needs_prev_snapshot) {
        std::apply([&](auto&... m) {
            (m.append_update_slots(u, prev, stage, res, mgr, needs_prev_snapshot), ...);
        }, modules);
    }

    void register_halo(halo::HaloExchanger& halo, fields::FieldsManager& mgr) {
        std::apply([&](auto&... m) {
            (m.register_halo(halo, mgr), ...);
        }, modules);
    }

    void bind_primitives(const fields::ConstPrimitiveView q) {
        std::apply([&](auto&... m) {
            (m.bind_primitives(q), ...);
        }, modules);
    }

    void set_freestream_state(const double rho, const double p) {
        std::apply([&](auto&... m) {
            (m.set_freestream_state(rho, p), ...);
        }, modules);
    }

    template <eos::EquationOfState EOS>
    void initialize(const mesh::MeshPart& mp,
                    const BoundaryConfig& bcfg,
                    const EOS& eos,
                    const gradient::VertexAdjacency& adj,
                    halo::HaloExchanger& halo,
                    const MPI_Comm comm) {
        std::apply([&](auto&... m) {
            (m.template initialize<EOS>(mp, bcfg, eos, adj, halo, comm), ...);
        }, modules);
    }

    void init_state(fields::FieldsManager& mgr) const {
        std::apply([&](const auto&... m) {
            (m.init_state(mgr), ...);
        }, modules);
    }

    template <eos::EquationOfState EOS>
    void apply_bcs(const EOS& eos, const mesh::MeshPart& mp) const {
        std::apply([&](const auto&... m) {
            (m.template apply_bcs<EOS>(eos, mp), ...);
        }, modules);
    }

    void compute_gradients(const gradient::GradientMethod& gm,
                           const mesh::MeshPart& mp) const {
        std::apply([&](const auto&... m) {
            (m.compute_gradients(gm, mp), ...);
        }, modules);
    }

    template <eos::EquationOfState EOS>
    void pre_sweep(const EOS& eos, const mesh::MeshPart& mp) {
        std::apply([&](auto&... m) {
            (m.template pre_sweep<EOS>(eos, mp), ...);
        }, modules);
    }

    template <eos::EquationOfState EOS>
    void face_sweep(const EOS& eos,
                    const mesh::MeshPart& mp,
                    double* lam,
                    const double* mdot,
                    const Geometry& geom) {
        std::apply([&](auto&... m) {
            (m.template face_sweep<EOS, Geometry>(eos, mp, lam, mdot, geom), ...);
        }, modules);
    }

    template <eos::EquationOfState EOS>
    void cell_sources(const EOS& eos,
                      const mesh::MeshPart& mp,
                      const fields::ConstPrimitiveGradView mean_grad) {
        std::apply([&](auto&... m) {
            (m.template cell_sources<EOS>(eos, mp, mean_grad), ...);
        }, modules);
    }

    void post_stage(const std::span<double* const> state) const {
        std::size_t offset = BaseT::kNumVars;
        std::apply([&](const auto&... m) {
            ([&] {
                using ModT = std::decay_t<decltype(m)>;
                if constexpr (ModT::kNumVars > 0) {
                    m.post_stage(state.subspan(offset, ModT::kNumVars));
                    offset += ModT::kNumVars;
                }
            }(), ...);
        }, modules);
    }

    void append_output(std::vector<io::vtk::SolutionField>& out) const {
        std::apply([&](const auto&... m) {
            (m.append_output(out), ...);
        }, modules);
    }

    /** @brief Eddy viscosity of the first providing module (nullptr if none). */
    [[nodiscard]] const double* mut_data() const noexcept {
        const double* result = nullptr;
        std::apply([&](const auto&... m) {
            ((result = result ? result : m.mut_data()), ...);
        }, modules);
        return result;
    }
};

static_assert(PhysicsGeneral<PhysicsStack<InviscidFlow>>);
static_assert(PhysicsGeneral<PhysicsStack<ViscousFlow>>);


} // namespace cfd::solver::physics