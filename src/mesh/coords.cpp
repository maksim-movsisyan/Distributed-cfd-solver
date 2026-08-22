#include <mpi.h>

#include <algorithm>
#include <unordered_map>

#include "cfd/mesh/cgns_reader.hpp"
#include "cfd/mpi/mpi_util.hpp"

std::vector<double> fetch_coords(RawMesh& m, const std::vector<int32_t>& node_gids) {
    const int nprocs = m.nprocs;
    // Unique sorted ids -> requests to the owners
    std::vector<int32_t> uniq(node_gids);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    std::vector<std::vector<int32_t>> req(nprocs);
    for (int32_t g : uniq) req[block_owner(g, m.node_displ)].push_back(g);
    auto incoming = alltoall_vec(req);

    std::vector<std::vector<double>> resp(nprocs);
    for (int p = 0; p < nprocs; ++p) {
        resp[p].reserve(incoming[p].size() * 3);
        for (int32_t g : incoming[p]) {
            const long long i = g - m.my_node_begin();
            resp[p].push_back(m.my_node_coords[3 * i]);
            resp[p].push_back(m.my_node_coords[3 * i + 1]);
            resp[p].push_back(m.my_node_coords[3 * i + 2]);
        }
    }
    auto mine = alltoall_vec(resp);

    std::unordered_map<int32_t, std::array<double, 3>> coords;
    coords.reserve(uniq.size() * 2);
    {
        // mine[p] are the replies to my requests req[p]; the keys are the requests.
        for (int p = 0; p < nprocs; ++p)
            for (size_t i = 0; i < req[p].size(); ++i) {
                const int32_t g = req[p][i];
                coords[g] = {mine[p][3 * i], mine[p][3 * i + 1], mine[p][3 * i + 2]};
            }
    }
    std::vector<double> out(node_gids.size() * 3);
    for (size_t i = 0; i < node_gids.size(); ++i) {
        const auto& c = coords.at(node_gids[i]);
        out[3 * i] = c[0];
        out[3 * i + 1] = c[1];
        out[3 * i + 2] = c[2];
    }
    return out;
}
