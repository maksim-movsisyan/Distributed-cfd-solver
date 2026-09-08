#pragma once

#include <array>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <initializer_list>
#include <string>

#include <mpi.h>
#include <toml++/toml.hpp>

namespace cfd::io::toml_utils {

// Rank 0 reads the entire file into a std::string and broadcasts it to all ranks.
// Zero filesystem contention: only 1 rank accesses the disk.
[[nodiscard]] std::string broadcast_file_content(const std::string& path, const MPI_Comm comm);

// Parses a TOML string in-memory or aborts with the parser's diagnostic.
[[nodiscard]] toml::table parse_in_memory_or_die(const std::string& content,
                                                 const std::string& source_path,
                                                 const MPI_Comm comm);

[[nodiscard]] const toml::table* req_table(const toml::table& t, const char* key,
                                           const std::string& ctx, const MPI_Comm comm);

[[nodiscard]] double req_number(const toml::table& t, const char* key,
                                const std::string& ctx, const MPI_Comm comm);

[[nodiscard]] double opt_number(const toml::table& t, const char* key, const double def);

[[nodiscard]] std::int64_t req_integer(const toml::table& t, const char* key,
                                       const std::string& ctx, const MPI_Comm comm);

[[nodiscard]] std::int64_t opt_integer(const toml::table& t, const char* key, const std::int64_t def);

[[nodiscard]] std::string req_string(const toml::table& t, const char* key,
                                     const std::string& ctx, const MPI_Comm comm);
[[nodiscard]] std::string opt_string(const toml::table& t, const char* key,
                                     const std::string& default_val);
[[nodiscard]] std::array<double, 3> req_vec3(const toml::table& t,
                                             const char* key,
                                             const std::string& ctx,
                                             const MPI_Comm comm);

void check_allowed_keys(const toml::table& t,
                        const std::initializer_list<const char*> allowed,
                        const std::string& ctx, const MPI_Comm comm);

void check_allowed_keys(const toml::table& t,
                        const std::vector<std::string>& allowed,
                        const std::string& ctx, const MPI_Comm comm);

void check_positive(const double v, const char* what, const std::string& ctx,
                    const MPI_Comm comm);

} // namespace cfd::io::toml_utils