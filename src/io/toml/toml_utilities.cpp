#include "cfd/io/toml/toml_utilities.hpp"
#include "cfd/mpi/log.hpp"

#include <fstream>
#include <sstream>

namespace cfd::io::toml_utils {

namespace {

    [[noreturn]] void fail(const MPI_Comm comm, const std::string& what) {
    mpi::fatal(comm, "config: " + what);
    std::abort();
}

}

[[nodiscard]] std::string broadcast_file_content(const std::string& path, const MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::string content;
    int content_size = 0;
    bool read_success = false;

    if (rank == 0) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::stringstream ss;
            ss << file.rdbuf();
            content = ss.str();
            content_size = static_cast<int>(content.size());
            read_success = true;
        }
    }

    // Broadcast file open status
    int status_int = read_success ? 1 : 0;
    MPI_Bcast(&status_int, 1, MPI_INT, 0, comm);

    if (status_int == 0) {
        fail(comm, "cannot open file '" + path + "' on Rank 0");
    }

    // Broadcast string length, allocate buffer on workers, broadcast string bytes
    MPI_Bcast(&content_size, 1, MPI_INT, 0, comm);
    if (rank != 0) {
        content.resize(static_cast<std::size_t>(content_size));
    }
    MPI_Bcast(content.data(), content_size, MPI_CHAR, 0, comm);

    return content;
}

[[nodiscard]] toml::table parse_in_memory_or_die(const std::string& content,
                                                 const std::string& source_path,
                                                 const MPI_Comm comm) {
    try {
        return toml::parse(content, source_path);
    } catch (const toml::parse_error& err) {
        fail(comm, "cannot parse '" + source_path + "': " + std::string(err.description()));
    }
}

[[nodiscard]] const toml::table* req_table(const toml::table& t, const char* key,
                                           const std::string& ctx, const MPI_Comm comm) {
    const toml::table* sub = t.get_as<toml::table>(key);
    if (sub == nullptr) {
        fail(comm, ctx + ": missing required section [" + key + "]");
    }
    return sub;
}

[[nodiscard]] double req_number(const toml::table& t, const char* key,
                                const std::string& ctx, const MPI_Comm comm) {
    const toml::node* node = t.get(key);
    if (node == nullptr) {
        fail(comm, ctx + ": missing required key '" + key + "'");
    }
    const auto val = node->value<double>();
    if (!val.has_value()) {
        fail(comm, ctx + ": key '" + key + "' must be a number");
    }
    return *val;
}

[[nodiscard]] double opt_number(const toml::table& t, const char* key, const double def) {
    const toml::node* node = t.get(key);
    if (node == nullptr) {
        return def;
    }
    return node->value_or(def);
}

[[nodiscard]] std::int64_t req_integer(const toml::table& t, const char* key,
                                       const std::string& ctx, const MPI_Comm comm) {
    const toml::node* node = t.get(key);
    if (node == nullptr) {
        fail(comm, ctx + ": missing required key '" + key + "'");
    }
    const auto val = node->value<std::int64_t>();
    if (!val.has_value()) {
        fail(comm, ctx + ": key '" + key + "' must be an integer");
    }
    return *val;
}

[[nodiscard]] std::int64_t opt_integer(const toml::table& t, const char* key, const std::int64_t def) {
    const toml::node* node = t.get(key);
    if (node == nullptr) {
        return def;
    }
    return node->value_or(def);
}

[[nodiscard]] std::string req_string(const toml::table& t, const char* key,
                                     const std::string& ctx, const MPI_Comm comm) {
    const toml::node* node = t.get(key);
    if (node == nullptr) {
        fail(comm, ctx + ": missing required key '" + key + "'");
    }
    const auto val = node->value<std::string>();
    if (!val.has_value()) {
        fail(comm, ctx + ": key '" + key + "' must be a string");
    }
    return *val;
}

[[nodiscard]] std::string opt_string(const toml::table& t, const char* key,
                       const std::string& default_val) {
    const toml::node* node = t.get(key);
    if (node == nullptr) {
        return default_val;
    }
    const auto val = node->value<std::string>();
    if (!val.has_value()) {
        return default_val;
    }
    return *val;
}

[[nodiscard]] std::array<double, 3> req_vec3(const toml::table& t,
                                             const char* key,
                                             const std::string& ctx,
                                             const MPI_Comm comm) {
    const toml::node* node = t.get(key);
    if (node == nullptr) {
        fail(comm, ctx + ": missing required key '" + key + "'");
    }
    const toml::array* arr = node->as_array();
    if (arr == nullptr || arr->size() != 3) {
        fail(comm, ctx + ": key '" + key + "' must be an array of 3 numbers");
    }
    std::array<double, 3> out{};
    for (std::size_t i = 0; i < 3; ++i) {
        const auto val = (*arr)[i].value<double>();
        if (!val.has_value()) {
            fail(comm, ctx + ": key '" + key + "' must be an array of 3 numbers");
        }
        out[i] = *val;
    }
    return out;
}

void check_allowed_keys(const toml::table& t,
                        const std::initializer_list<const char*> allowed,
                        const std::string& ctx, const MPI_Comm comm) {
    for (const auto& [k, v] : t) {
        (void)v;
        const std::string_view name = k.str();
        bool known = false;
        for (const char* a : allowed) {
            if (name == std::string_view(a)) {
                known = true;
                break;
            }
        }
        if (!known) {
            fail(comm, ctx + ": unknown key '" + std::string(name) + "'");
        }
    }
}

void check_allowed_keys(const toml::table& t,
                        const std::vector<std::string>& allowed,
                        const std::string& ctx, const MPI_Comm comm) {
    for (const auto& [k, v] : t) {
        (void)v;
        const std::string_view name = k.str();
        bool known = false;
        for (const std::string& a : allowed) {
            if (name == a) {
                known = true;
                break;
            }
        }
        if (!known) {
            fail(comm, ctx + ": unknown key '" + std::string(name) + "'");
        }
    }
}

void check_positive(const double v, const char* what, const std::string& ctx,
                    const MPI_Comm comm) {
    if (!(v > 0.0)) {
        fail(comm, ctx + ": '" + what + "' must be positive (got " + std::to_string(v) + ")");
    }
}

} // namespace cfd::io::toml_utils