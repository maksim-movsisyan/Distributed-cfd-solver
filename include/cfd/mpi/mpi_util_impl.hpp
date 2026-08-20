#pragma once
// Implementation of the templated all-to-all (included only from mpi_util.hpp).
// Exchange uses byte blocks (MPI_BYTE); counts and offsets are in bytes.
#include <cstring>
#include <type_traits>

template <class T>
std::vector<std::vector<T>> alltoall_vec(const std::vector<std::vector<T>>& in) {
    static_assert(std::is_trivially_copyable_v<T>);
    int p = 0, r = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    if (p == 1) return in;

    std::vector<int> scounts(p), rcounts(p), sdispl(p + 1), rdispl(p + 1);
    for (int i = 0; i < p; ++i) {
        sdispl[i + 1] = sdispl[i] + static_cast<int>(in[i].size() * sizeof(T));
        scounts[i] = sdispl[i + 1] - sdispl[i];
    }
    MPI_Alltoall(scounts.data(), 1, MPI_INT, rcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < p; ++i) rdispl[i + 1] = rdispl[i] + rcounts[i];

    std::vector<char> sbuf(sdispl[p]), rbuf(rdispl[p]);
    for (int i = 0; i < p; ++i)
        if (!in[i].empty()) std::memcpy(sbuf.data() + sdispl[i], in[i].data(), scounts[i]);

    MPI_Alltoallv(sbuf.data(), scounts.data(), sdispl.data(), MPI_BYTE, rbuf.data(), rcounts.data(),
                  rdispl.data(), MPI_BYTE, MPI_COMM_WORLD);

    std::vector<std::vector<T>> out(p);
    for (int i = 0; i < p; ++i)
        out[i].assign(reinterpret_cast<const T*>(rbuf.data() + rdispl[i]),
                      reinterpret_cast<const T*>(rbuf.data() + rdispl[i + 1]));
    return out;
}
