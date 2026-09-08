#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <span>

#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/solver/gradient/gradient_operator.hpp"

namespace cfd::solver::gradient {

enum class GradientType {
    GreenGaussFace,
    GreenGaussCell,
    LeastSquaresCellFace,
    LeastSquaresCellNode
};

/**
 * @class GradientManager
 * @brief Gradient aggregator and lifecycle manager on CPU.
 */
class GradientManager {
public:
    using GOPtr = std::unique_ptr<GradientOperator>;
    using GOBuilder = std::function<GOPtr()>;

    GradientManager() { register_all(); }

    void create_gradient(const GradientType& name);

    void setup_gradient(const mesh::MeshPart& mesh, mesh::MeshAuxConnectivity& aux_conn);

    void apply_gradient(const double* s, double* g, const std::size_t stride, 
                        const mesh::MeshPart& mesh, const mesh::MeshAuxConnectivity& aux_conn) const;

    void apply_gradient_set(std::span<const double*> s, std::span<double*> g, const std::size_t stride, 
                            const mesh::MeshPart& mesh, const mesh::MeshAuxConnectivity& aux_conn) const;

    [[nodiscard]] const char* active_gradient_name() const noexcept {
        return grad_ ? grad_->name() : "none";
    }

private:
    GOPtr grad_ = nullptr;
    std::unordered_map<GradientType, GOBuilder> registry_;

    void register_all();
};

} // namespace cfd::solver::gradient