// Generic gradient method interface and the shared vertex adjacency.
//
// GradientMethod is a runtime-virtual strategy applied to ALL primitive
// fields in one call (never per cell) — one virtual dispatch per residual
// evaluation. Concrete methods: weighted least-squares with precomputed
// matrices (lsq_gradient.hpp); Green-Gauss can be added later without
// touching the solver.
#pragma once

#include <cstddef>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/fields/fields_view.hpp"

namespace cfd::solver::gradient {

/**
 * @struct VertexAdjacency
 * @brief Vertex-sharing cell adjacency of every OWNED cell in CSR layout.
 * 
 * Stencil for weighted least-squares gradient and MUSCL limiter extrema.
 * Neighbours of cell c occupy indices [offsets[c], offsets[c+1]).
 */
struct VertexAdjacency {
    std::vector<LocalIndex> offsets; ///< [n_own + 1] CSR row pointers
    std::vector<LocalIndex> cells;   ///< Neighbour cell indices (may include halo ghosts)
    std::vector<double> dx;          ///< centroid(nb) - centroid(c) [m]
    std::vector<double> dy;          ///< centroid(nb) - centroid(c) [m]
    std::vector<double> dz;          ///< centroid(nb) - centroid(c) [m]
    std::vector<double> w;           ///< Inverse-distance weights 1 / |dr| [1/m]

    [[nodiscard]] std::size_t size_owned() const noexcept {
        return offsets.empty() ? 0 : offsets.size() - 1;
    }

    [[nodiscard]] std::size_t num_neighbors(const std::size_t c) const noexcept {
        return static_cast<std::size_t>(offsets[c + 1] - offsets[c]);
    }

    [[nodiscard]] std::size_t total_connections() const noexcept {
        return cells.size();
    }
};

/**
 * @brief Builds the vertex-sharing adjacency for owned cells of the partition.
 * Executed once during solver initialization.
 */
[[nodiscard]] VertexAdjacency build_vertex_adjacency(const mesh::MeshPart& mp);

/**
 * @class GradientMethod
 * @brief Abstract strategy interface for spatial gradient reconstruction.
 */
class GradientMethod {
public:
    virtual ~GradientMethod() = default;

    /**
     * @brief Computes spatial gradients of all primitive variables for owned cells [0, n_own).
     * 
     * @param[in]  q    Primitive fields [prs, vx, vy, vz, tmp] (halo-complete).
     * @param[out] grad Output gradient view.
     */
    virtual void compute(fields::ConstPrimitiveView q,
                         fields::PrimitiveGradView<double> grad) const = 0;

    [[nodiscard]] virtual const char* name() const noexcept = 0;
};

} // namespace cfd::solver::gradient