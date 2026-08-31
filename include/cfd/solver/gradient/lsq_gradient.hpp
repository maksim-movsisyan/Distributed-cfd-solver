// Weighted least-squares (WLSQ) gradient with precomputed inverse matrices.
#pragma once

#include <cstddef>
#include <vector>

#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/fields/fields_view.hpp"
#include "cfd/solver/gradient/gradient.hpp"

namespace cfd::solver::gradient {

/**
 * @class LsqGradient
 * @brief High-performance Weighted Least-Squares gradient evaluator.
 * 
 * Inverts the normal equation matrix A_c = sum_j w_j (d_j x d_j) once at startup.
 * Runtime evaluation per stage performs only linear neighbor gathering and a 3x3 * 3x5 multiply.
 */
class LsqGradient final : public GradientMethod {
public:
    LsqGradient(const mesh::MeshPart& mp, const VertexAdjacency& adj);
    ~LsqGradient() override = default;

    void compute(fields::ConstPrimitiveView q,
                 fields::PrimitiveGradView<double> grad) const override;

    void compute_scalar(const double* f, double* grad3, std::size_t stride) const override;

    [[nodiscard]] const char* name() const noexcept override { return "WLSQ"; }

private:
    const VertexAdjacency& adj_;
    std::size_t n_own_{0};

    // Row-major 3x3 inverse matrices: 9 doubles per owned cell [9 * n_own]
    std::vector<double> mat9_;

    // Pre-weighted displacement vectors: w_j * dx_j, w_j * dy_j, w_j * dz_j
    std::vector<double> wdx_;
    std::vector<double> wdy_;
    std::vector<double> wdz_;
};

} // namespace cfd::solver::gradient