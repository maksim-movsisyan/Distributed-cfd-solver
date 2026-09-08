#include "cfd/linalg/vector.hpp"

#include <algorithm>
#include <cmath>

#include "cfd/linalg/types.hpp"

namespace cfd::linalg {

void Vector::setZero() { std::fill(values_.begin(), values_.end(), 0.0); }

void Vector::copyFrom(const Vector& other) {
    check(other.block_size_ == block_size_ && other.scalarSize() == scalarSize(),
          layout_.comm(), "Vector::copyFrom: incompatible vectors");
    values_ = other.values_;
}

void Vector::scale(double alpha) {
    const std::size_t n = scalarSize();
    double* CFD_RESTRICT v = values_.data();
    for (std::size_t i = 0; i < n; ++i) v[i] *= alpha;
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

double Vector::norm2() const { 
    const std::size_t n = ownedScalarCount();
    const double* CFD_RESTRICT xv = values_.data();
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += xv[i] * xv[i];
    MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_SUM, layout_.comm());
    return std::sqrt(s);
}

void Vector::batchedDots(std::span<const std::pair<const Vector*, const Vector*>> products,
                         std::span<double> out) {
    check(!products.empty(), MPI_COMM_SELF, "Vector::batchedDots: empty input");
    MPI_Comm comm = products.front().first->layout_.comm();
    check(products.size() == out.size(), comm, "Vector::batchedDots: output size mismatch");
    
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
