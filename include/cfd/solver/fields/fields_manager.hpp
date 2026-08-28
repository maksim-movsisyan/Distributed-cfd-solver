#pragma once

#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string>
#include <cstddef>

namespace cfd::solver::fields {
/**
 * @class FieldLocation
 * @brief Specifies where the field data is centered.
 */
enum class FieldLocation {
    OwnCell,   ///< Cell-centered values (size = n_own_cells)
    Cell,      ///< Cell-centered values (size = n_cells: owned + ghost + boundary ghost)
    Face,      ///< Face-centered values (size = n_faces)
    Node,      ///< Node-centered values (size = n_nodes)
    Custom     ///< Arbitrary user-defined size
};


/**
 * @class AbstractField
 * @brief Base class for all physical fields.
 */
class AbstractField {
public:
    std::string name;           ///< Field name
    FieldLocation location;     ///< Field location

    /**
     * @brief AbstractField constructor.
     * @param[in] n Field name.
     * @param[in] loc Field location.
     */
    AbstractField(std::string n, FieldLocation loc) 
        : name(std::move(n)), location(loc) {}

    virtual ~AbstractField() = default;

    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    [[nodiscard]] virtual std::size_t byte_size() const noexcept = 0;
};


/**
 * @class Field
 * @brief Concrete implementation of a scalar/vector field.
 * @tparam T Underlying data type (e.g., real, integer).
 */
template <typename T>
class Field final: public AbstractField {
public:
    std::vector<T> data;        ///< Field data

    /**
     * @brief Field constructor.
     * @param[in] n Field name.
     * @param[in] size Field size.
     * @param[in] loc Field location.
     */
    Field(std::string n, const std::size_t count, const FieldLocation loc)
        : AbstractField(std::move(n), loc), data(count, T{}) {}

    [[nodiscard]] std::size_t size() const noexcept override { return data.size(); }
    [[nodiscard]] std::size_t byte_size() const noexcept override { return data.size() * sizeof(T); }

    [[nodiscard]] T* raw_ptr() noexcept { return data.data(); }
    [[nodiscard]] const T* raw_ptr() const noexcept { return data.data(); }
};


/**
 * @class FieldsManager
 * @brief Manager of all physical fields of all types (integer, real ...)
 * 
 * Provides methods to create, store, and retrieve fields by name and type.
 */
class FieldsManager {
private:
    std::unordered_map<std::string, std::unique_ptr<AbstractField>> registry_;       ///< Fields registry [name, data]

public:
    FieldsManager() = default;
    
    /**
     * @brief Add field to registry.
     * @param[in] n Field name.
     * @param[in] size Field size.
     * @param[in] loc Field location.
     * 
     * @example manager.add_field<cfd::real>("pressure", mesh.n_cells, FieldLocation::Cell);
     */
    template <typename T>
    void add_field(const std::string& name, const std::size_t size, const FieldLocation loc) {
        if (registry_.contains(name)) {
            throw std::invalid_argument("FieldsManager::add_field: field '" + name + "' already exists!");
        }
        registry_[name] = std::make_unique<Field<T>>(name, size, loc);
    }
    
    /**
     * @brief Returns field pointer by name.
     * @param[in] n Field name.
     * 
     * @return Pointer to field data.
     */
    template <typename T>
    [[nodiscard]] T* get_field_ptr(const std::string& name) noexcept {
        const auto it = registry_.find(name);
        if (it == registry_.end()) return nullptr;

        auto* concrete_field = dynamic_cast<Field<T>*>(it->second.get());
        return concrete_field ? concrete_field->raw_ptr() : nullptr;
    }

    /**
     * @brief Returns field const pointer by name.
     * @param[in] n Field name.
     * 
     * @return Pointer to field data (read only).
     */
    template <typename T>
    [[nodiscard]] const T* get_field_ptr(const std::string& name) const noexcept {
        const auto it = registry_.find(name);
        if (it == registry_.end()) return nullptr;

        const auto* concrete_field = dynamic_cast<const Field<T>*>(it->second.get());
        return concrete_field ? concrete_field->raw_ptr() : nullptr;
    }

    /**
     * @brief Strict typed pointer lookup (throws if missing or type mismatch).
     */
    template <typename T>
    [[nodiscard]] T* get_required_field_ptr(const std::string& name) {
        T* ptr = get_field_ptr<T>(name);
        if (!ptr) {
            throw std::runtime_error("FieldsManager: required field '" + name + "' not found or type mismatch!");
        }
        return ptr;
    }

    template <typename T>
    [[nodiscard]] const T* get_required_field_ptr(const std::string& name) const {
        const T* ptr = get_field_ptr<T>(name);
        if (!ptr) {
            throw std::runtime_error("FieldsManager: required field '" + name + "' not found or type mismatch!");
        }
        return ptr;
    }


    /** @brief Fill field with value */
    template <typename T>
    void fill_field(const std::string& name, const T value) {
        auto* field = dynamic_cast<Field<T>*>(get_field(name));
        if (!field) {
            throw std::runtime_error("FieldsManager::fill_field: invalid field or cast for '" + name + "'");
        }
        std::fill(field->data.begin(), field->data.end(), value);
    }

    /**
     * @brief Returns abstract field object by name.
     * @param[in] name Field name.
     * @return Pointer to abstract field (object) or nullptr if not found.
     */
    [[nodiscard]] AbstractField* get_field(const std::string& name) noexcept {
        const auto it = registry_.find(name);
        return (it == registry_.end()) ? nullptr : it->second.get();
    }

    /** @brief Const version of get_field. */
    [[nodiscard]] const AbstractField* get_field(const std::string& name) const noexcept {
        const auto it = registry_.find(name);
        return (it == registry_.end()) ? nullptr : it->second.get();
    }

    /** @brief Checks if a field with the given name is registered. */
    [[nodiscard]] bool has_field(const std::string& name) const noexcept {
        return registry_.contains(name);
    }

    /** @brief Return field size by its name. */
    [[nodiscard]] std::size_t get_field_size(const std::string& name) const noexcept {
        const auto it = registry_.find(name);
        return (it == registry_.end()) ? 0 : it->second->size();
    }
};

} // namespace cfd