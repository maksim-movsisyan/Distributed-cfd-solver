#pragma once

#include <cstddef>
#include <vector>
#include <span>

#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_connectivity.hpp"

namespace cfd::numerics::gradient {

/**
 * @class GradientOperator
 * @brief Abstract base class for spatial gradient evaluation on CPU.
 */
class GradientOperator {
public:
    virtual ~GradientOperator() = default;

    /** @brief Precomputes metric coefficients, CSR stencils, and inverted matrices */
    virtual void setup(const mesh::MeshPart& mesh, mesh::MeshAuxConnectivity& aux_conn) = 0;

    /**
     * @brief Evaluates spatial gradient for a single scalar field on owned cells [0, n_own).
     * @param[in]  s      Scalar field of size mesh.n_cells (halo-complete).
     * @param[out] g_xyz  SoA gradient: [d/dx | d/dy | d/dz ] of size mesh.n_cells.
     * @param[in]  m      Migrated local mesh partition.
     */
    virtual void apply(const double* CFD_RESTRICT s,
                       double* CFD_RESTRICT gx,
                       double* CFD_RESTRICT gy,
                       double* CFD_RESTRICT gz,
                       const mesh::MeshPart& mesh,
                       const mesh::MeshAuxConnectivity& aux_conn) const = 0;
    
    virtual void apply_set(std::span<const double* const> s,
                           std::span<double* const> gx,
                           std::span<double* const> gy,
                           std::span<double* const> gz,
                           const mesh::MeshPart& mesh,
                           const mesh::MeshAuxConnectivity& aux_conn) const = 0;

    [[nodiscard]] virtual const char* name() const noexcept = 0;
};

/**
 * @class GreenGaussCellGradient
 * @brief Cell-Based Green-Gauss gradient evaluator.
 * Precomputes cell->faces CSR topology and iterates cell-by-cell (Zero thread contention).
 */
class GreenGaussCellGradient : public GradientOperator {
public:
    void setup(const mesh::MeshPart& mesh, mesh::MeshAuxConnectivity& aux_conn) override;
    void apply(const double* CFD_RESTRICT s,
               double* CFD_RESTRICT gx,
               double* CFD_RESTRICT gy,
               double* CFD_RESTRICT gz,
               const mesh::MeshPart& mesh,
               const mesh::MeshAuxConnectivity& aux_conn) const override;
    void apply_set(std::span<const double* const> s,
                   std::span<double* const> gx,
                   std::span<double* const> gy,
                   std::span<double* const> gz,
                   const mesh::MeshPart& mesh,
                   const mesh::MeshAuxConnectivity& aux_conn) const override;
    [[nodiscard]] const char* name() const noexcept override { return "GREEN_GAUSS_CELL"; }
};

/**
 * @class GreenGaussFaceGradient
 * @brief Face-Based Green-Gauss gradient evaluator.
 * Iterates face-by-face and scatters fluxes to owner and neighbor cells.
 */
class GreenGaussFaceGradient : public GradientOperator {
public:
    void setup(const mesh::MeshPart& /*mesh*/, mesh::MeshAuxConnectivity& /*aux_conn*/) override;
    void apply(const double* CFD_RESTRICT s,
               double* CFD_RESTRICT gx,
               double* CFD_RESTRICT gy,
               double* CFD_RESTRICT gz,
               const mesh::MeshPart& mesh,
               const mesh::MeshAuxConnectivity& aux_conn) const override;
    void apply_set(std::span<const double* const> s,
                   std::span<double* const> gx,
                   std::span<double* const> gy,
                   std::span<double* const> gz,
                   const mesh::MeshPart& mesh,
                   const mesh::MeshAuxConnectivity& aux_conn) const override;
    [[nodiscard]] const char* name() const noexcept override { return "GREEN_GAUSS_FACE"; }
};

/**
 * @class LeastSquaresCellFaceGradient
 * @brief Cell-Based Weighted Least-Squares (WLSQ) gradient with precomputed inverse matrix.
 * Precomputes 3 weights (Cx, Cy, Cz) per cell-face connection into a single contiguous buffer.
 */
class LeastSquaresCellFaceGradient : public GradientOperator {
public:
    void setup(const mesh::MeshPart& mesh, mesh::MeshAuxConnectivity& aux_conn) override;
    void apply(const double* CFD_RESTRICT s,
               double* CFD_RESTRICT gx,
               double* CFD_RESTRICT gy,
               double* CFD_RESTRICT gz,
               const mesh::MeshPart& mesh,
               const mesh::MeshAuxConnectivity& aux_conn) const override;
    void apply_set(std::span<const double* const> s,
                   std::span<double* const> gx,
                   std::span<double* const> gy,
                   std::span<double* const> gz,
                   const mesh::MeshPart& mesh,
                   const mesh::MeshAuxConnectivity& aux_conn) const override;
    [[nodiscard]] const char* name() const noexcept override { return "LEAST_SQUARES_FACE"; }

private:
    /** @brief Precomputes weights C_{c, j} = A_c^{-1} * (w_j * dr_j) on CPU */
    void lsq_precompute_coeffs(const mesh::MeshPart& mesh, std::vector<double>& temp_coeffs);
    
    std::vector<double> m_coeffs_; // Size: 3 * total_connections
    std::size_t m_coeffs_off_ = 0; // Stride = total_connections
};

/**
 * @class LeastSquaresCellNodeGradient
 * @brief Vertex-Neighborhood Weighted Least-Squares (WLSQ) gradient.
 * Uses extended stencil of cells sharing at least one node.
 * Highly robust on boundary tetrahedra and degenerate meshes.
 */
class LeastSquaresCellNodeGradient : public GradientOperator {
public:
    void setup(const mesh::MeshPart& mesh, mesh::MeshAuxConnectivity& aux_conn) override;
    void apply(const double* CFD_RESTRICT s,
               double* CFD_RESTRICT gx,
               double* CFD_RESTRICT gy,
               double* CFD_RESTRICT gz,
               const mesh::MeshPart& mesh,
               const mesh::MeshAuxConnectivity& aux_conn) const override;
    void apply_set(std::span<const double* const> s,
                   std::span<double* const> gx,
                   std::span<double* const> gy,
                   std::span<double* const> gz,
                   const mesh::MeshPart& mesh,
                   const mesh::MeshAuxConnectivity& aux_conn) const override;
    [[nodiscard]] const char* name() const noexcept override { return "LEAST_SQUARES_NODE"; }

private:
    std::vector<double> m_coeffs_; // Size: 3 * total_node_connections
    std::size_t m_coeffs_off_ = 0; // Stride = total_node_connections
};

} // namespace cfd::numerics::gradient