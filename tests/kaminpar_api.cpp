// Minimal test of the distributed KaMinPar API: a ring graph over P ranks.
#include <dkaminpar.h>
#include <mpi.h>

#include <cstdio>
#include <span>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int n_per = argc > 1 ? std::atoi(argv[1]) : 8;  // nodes per rank
    const int nthreads = argc > 2 ? std::atoi(argv[2]) : 1;
    const long long n_g = static_cast<long long>(n_per) * nprocs;

    // Ring: node i -> i-1, i+1 (global ids)
    std::vector<uint64_t> node_dist(nprocs + 1);
    for (int p = 0; p <= nprocs; ++p) node_dist[p] = static_cast<uint64_t>(p) * n_per;
    std::vector<uint64_t> xadj(n_per + 1), adjncy;
    for (int i = 0; i < n_per; ++i) {
        const long long g = rank * n_per + i;
        xadj[i] = adjncy.size();
        adjncy.push_back((g - 1 + n_g) % n_g);
        adjncy.push_back((g + 1) % n_g);
    }
    xadj[n_per] = adjncy.size();

    std::printf("[r%d] n=%d m=%zu\n", rank, n_per, adjncy.size());
    std::fflush(stdout);

    auto ctx = kaminpar::dist::create_default_context();
    ctx.parallel.num_threads = static_cast<std::size_t>(nthreads);
    kaminpar::dKaMinPar part(MPI_COMM_WORLD, nthreads, ctx);
    part.set_output_level(kaminpar::OutputLevel::QUIET);
    part.copy_graph(node_dist, xadj, adjncy);
    std::printf("[r%d] graph set\n", rank);
    std::fflush(stdout);
    part.set_k(2);
    part.set_uniform_max_block_weights(0.03);
    std::vector<kaminpar::dist::BlockID> partition(n_per);
    const auto cut = part.compute_partition(partition);
    std::printf("[r%d] cut=%lld blocks:", rank, static_cast<long long>(cut));
    for (auto b : partition) std::printf(" %u", b);
    std::printf("\n");
    std::fflush(stdout);

    MPI_Finalize();
    return 0;
}
