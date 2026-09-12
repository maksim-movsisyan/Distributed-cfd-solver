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
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/fields/fields_manager.hpp"
#include "cfd/fields/halo.hpp"
#include "cfd/numerics/gradient/gradient_manager.hpp"
#include "cfd/solver/bc/config.hpp"
#include "cfd/solver/eos/concepts.hpp"
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
    static constexpr mesh::AuxGeomType kAuxGeometry =
                    (BaseT::kAuxGeometry | ... | Modules::kAuxGeometry);
    static constexpr mesh::AuxConnType kAuxConnectivity =
                    (BaseT::kAuxConnectivity | ... | Modules::kAuxConnectivity);

    static constexpr std::size_t kNumExtraVars = (Modules::kNumVars + ... + 0);
    static constexpr bool kHasEddyViscosity = (Modules::kHasEddyViscosity || ...);
    static constexpr bool kNeedsFaceMdot = (Modules::kNeedsFaceMdot || ...);
    static constexpr bool kNeedsWallDist = (Modules::kNeedsWallDist || ...);

    
    // --- PhysicsGeneral interface: delegating to the mean-flow base -------------
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
                         const std::size_t n_total) {
        std::apply([&](auto&... m) {
            (m.register_fields(mgr, n_total), ...);
        }, modules);
    }

    void append_update_slots(std::vector<double*>& u,
                             std::vector<double*>& stage,
                             std::vector<double*>& res,
                             fields::FieldsManager& mgr) {
        std::apply([&](auto&... m) {
            (m.append_update_slots(u, stage, res, mgr), ...);
        }, modules);
    }

    void register_halo(fields::halo::HaloExchanger& halo, fields::FieldsManager& mgr) {
        std::apply([&](auto&... m) {
            (m.register_halo(halo, mgr), ...);
        }, modules);
    }

    void bind_primitives(std::span<const double*> q) {
        std::apply([&](auto&... m) {
            (m.bind_primitives(q), ...);
        }, modules);
    }

    void set_freestream_state(const double rho, const double p) {
        std::apply([&](auto&... m) {
            (m.set_freestream_state(rho, p), ...);
        }, modules);
    }

    template <eos::EquationOfStatePolicy EOS>
    void initialize(const mesh::MeshPart& mesh,
                    const mesh::MeshAuxConnectivity& aux_conn,
                    const mesh::MeshAuxGeometry& aux_geom,
                    const bc::BoundaryConfig& bcfg,
                    const EOS& eos,
                    fields::halo::HaloExchanger& halo,
                    const MPI_Comm comm) {
        std::apply([&](auto&... m) {
            (m.template initialize<EOS>(mesh, aux_conn, aux_geom, bcfg, eos, halo, comm), ...);
        }, modules);
    }

    void init_state(fields::FieldsManager& mgr) const {
        std::apply([&](const auto&... m) {
            (m.init_state(mgr), ...);
        }, modules);
    }

    template <eos::EquationOfStatePolicy EOS>
    void apply_bcs(const EOS& eos, const mesh::MeshPart& mesh) const {
        std::apply([&](const auto&... m) {
            (m.template apply_bcs<EOS>(eos, mesh), ...);
        }, modules);
    }

    void compute_gradients(const numerics::gradient::GradientManager& gm,
                           const mesh::MeshPart& mesh) const {
        std::apply([&](const auto&... m) {
            (m.compute_gradients(gm, mesh), ...);
        }, modules);
    }

    template <eos::EquationOfStatePolicy EOS>
    void pre_sweep(const EOS& eos, const mesh::MeshPart& mesh) {
        std::apply([&](auto&... m) {
            (m.template pre_sweep<EOS>(eos, mesh), ...);
        }, modules);
    }

    template <eos::EquationOfStatePolicy EOS>
    void face_sweep(const EOS& eos,
                    const mesh::MeshPart& mesh,
                    const mesh::MeshAuxConnectivity& aux_conn,
                    const mesh::MeshAuxGeometry& aux_geom,
                    double* lam,
                    const double* mdot) {
        std::apply([&](auto&... m) {
            (m.template face_sweep<EOS>(eos, mesh, aux_conn, aux_geom, lam, mdot), ...);
        }, modules);
    }

    template <eos::EquationOfStatePolicy EOS>
    void cell_sources(const EOS& eos,
                      const mesh::MeshPart& mesh,
                      std::span<const double*> gx,
                      std::span<const double*> gy,
                      std::span<const double*> gz) {
        std::apply([&](auto&... m) {
            (m.template cell_sources<EOS>(eos, mesh, gx, gy, gz), ...);
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

    /**
     * @brief Molecular (laminar) Prandtl number Pr.
     * Taken from the viscous mean-flow base, or default constant for Euler.
     */
    [[nodiscard]] constexpr double prandtl() const noexcept {
        if constexpr (BaseT::kHasViscous) {
            return base.prandtl;
        } else {
            return constants::kAirPrandtl;  
        }
    }

    /**
     * @brief Turbulent Prandtl number Pr_t.
     * Extracted from the active turbulence model (the module with kHasEddyViscosity),
     * or defaults to standard Reynolds analogy (0.85-0.90) if no turbulence module is present.
     */
    [[nodiscard]] double prandtl_turb() const noexcept {
        if constexpr (!kHasEddyViscosity) {
            return constants::kTurbPrandtl; // e.g. 0.85 or 0.90
        } else {
            double pr_t = constants::kTurbPrandtl;
            bool found = false;

            std::apply([&](const auto&... m) {
                ([&] {
                    using ModT = std::decay_t<decltype(m)>;
                    if constexpr (ModT::kHasEddyViscosity) {
                        if (!found) {
                            if constexpr (requires { m.prandtl_turb; }) {
                                pr_t = m.prandtl_turb;
                                found = true;
                            } else if constexpr (requires { m.prandtl_turb(); }) {
                                pr_t = m.prandtl_turb();
                                found = true;
                            }
                        }
                    }
                }(), ...);
            }, modules);

            return pr_t;
        }
    }
};

static_assert(PhysicsGeneral<PhysicsStack<InviscidFlow>>);
static_assert(PhysicsGeneral<PhysicsStack<ViscousFlow>>);


} // namespace cfd::solver::physics