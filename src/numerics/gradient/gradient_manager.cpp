#include "cfd/numerics/gradient/gradient_manager.hpp"

#include <cassert>

namespace cfd::numerics::gradient {

void GradientManager::register_all() {
    registry_[GradientType::GreenGaussCell] = []() -> GOPtr {
        return std::make_unique<GreenGaussCellGradient>();
    };
    registry_[GradientType::GreenGaussFace] = []() -> GOPtr {
        return std::make_unique<GreenGaussFaceGradient>();
    };
    registry_[GradientType::LeastSquaresCellFace] = []() -> GOPtr {
        return std::make_unique<LeastSquaresCellFaceGradient>();
    };
    registry_[GradientType::LeastSquaresCellNode] = []() -> GOPtr {
        return std::make_unique<LeastSquaresCellNodeGradient>();
    };
}

void GradientManager::create_gradient(const GradientType& name) {
    auto it = registry_.find(name);
    if (it == registry_.end()) {
        it = registry_.find(GradientType::GreenGaussFace);
    }
    grad_ = it->second();
}

void GradientManager::setup_gradient(const mesh::MeshPart& mesh, mesh::MeshAuxConnectivity& aux_conn) {
    assert(grad_ && "GradientManager::setup_gradient: gradient operator was not created! Call create_gradient() first.");
    grad_->setup(mesh, aux_conn);
}

void GradientManager::apply_gradient(const double* s, double* g, const std::size_t stride, 
                        const mesh::MeshPart& mesh, const mesh::MeshAuxConnectivity& aux_conn) const {
    assert(grad_ && "GradientManager::apply_gradient: gradient operator is nullptr!");
    grad_->apply(s, g, stride, mesh, aux_conn);
}

void GradientManager::apply_gradient_set(std::span<const double*> s, std::span<double*> g, const std::size_t stride, 
                            const mesh::MeshPart& mesh, const mesh::MeshAuxConnectivity& aux_conn) const {
    assert(grad_ && "GradientManager::apply_gradient_set: gradient operator is nullptr!");
    grad_->apply_set(s, g, stride, mesh, aux_conn);
}

} // namespace cfd::numerics::gradient