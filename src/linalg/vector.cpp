#include "cfd/linalg/vector.hpp"

#include <algorithm>
#include <cmath>

#include "cfd/linalg/types.hpp"

namespace cfd::linalg {

Vector::Vector(const VectorLayout& layout, int block_size)
    : layout_(layout), block_size_(block_size) {
    check(block_size > 0, layout.comm(), "Vector: block size must be positive");
    values_.assign(static_cast<std::size_t>(layout_.localSize() + layout_.ghostSize()) *
                       static_cast<std::size_t>(block_size), 0.0);
}

void Vector::setZero() { std::fill(values_.begin(), values_.end(), 0.0); }

void Vector::copyFrom(const Vector& other) {
    check(other.block_size_ == block_size_ && other.scalarSize() == scalarSize(),
          layout_.comm(), "Vector::copyFrom: incompatible vectors");
    values_ = other.values_;
}

void Vector::scale(double alpha) {
    for (double& v : values_) v *= alpha;
}

void Vector::axpy(double alpha, const Vector& x) {
    check(x.block_size_ == block_size_ && x.scalarSize() == scalarSize(), layout_.comm(),
          "Vector::axpy: incompatible vectors");
    const std::size_t n = values_.size();
    const double* CFD_RESTRICT xv = x.values_.data();
    double* CFD_RESTRICT yv = values_.data();
    for (std::size_t i = 0; i < n; ++i) yv[i] += alpha * xv[i];
}

double Vector::dot(const Vector& other) const {
    check(other.block_size_ == block_size_ && other.layout_.compatibleWith(layout_),
          layout_.comm(), "Vector::dot: incompatible vectors");
    const std::size_t n = ownedScalarCount();
    const double* CFD_RESTRICT xv = values_.data();
    const double* CFD_RESTRICT yv = other.values_.data();
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += xv[i] * yv[i];
    MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_SUM, layout_.comm());
    return s;
}

double Vector::norm2() const { return std::sqrt(dot(*this)); }

void Vector::batchedDots(std::span<const std::pair<const Vector*, const Vector*>> products,
                         std::span<double> out) {
    check(products.size() == out.size(), MPI_COMM_WORLD,
          "Vector::batchedDots: output size mismatch");
    check(!products.empty(), MPI_COMM_WORLD, "Vector::batchedDots: empty input");
    MPI_Comm comm = products.front().first->layout_.comm();
    for (std::size_t k = 0; k < products.size(); ++k) {
        const Vector& a = *products[k].first;
        const Vector& b = *products[k].second;
        check(a.block_size_ == b.block_size_ && a.scalarSize() == b.scalarSize(), comm,
              "Vector::batchedDots: incompatible vectors");
        const std::size_t n = a.ownedScalarCount();
        const double* CFD_RESTRICT xv = a.values_.data();
        const double* CFD_RESTRICT yv = b.values_.data();
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += xv[i] * yv[i];
        out[k] = s;
    }
    MPI_Allreduce(MPI_IN_PLACE, out.data(), static_cast<int>(out.size()), MPI_DOUBLE, MPI_SUM,
                  comm);
}

}  // namespace cfd::linalg
