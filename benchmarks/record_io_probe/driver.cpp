#include <array>
#include <cstddef>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "probe.h"

import anndata;

namespace {

using namespace anndata;

struct Probe {
    record_probe_enable_fn enable = nullptr;
    record_probe_reset_fn reset = nullptr;
    record_probe_json_fn json = nullptr;
};

Probe probe() {
    Probe p{
        reinterpret_cast<record_probe_enable_fn>(dlsym(RTLD_DEFAULT, RECORD_PROBE_ENABLE)),
        reinterpret_cast<record_probe_reset_fn>(dlsym(RTLD_DEFAULT, RECORD_PROBE_RESET)),
        reinterpret_cast<record_probe_json_fn>(dlsym(RTLD_DEFAULT, RECORD_PROBE_JSON)),
    };
    if (!p.enable || !p.reset || !p.json) throw std::runtime_error("LD_PRELOAD probe hooks are unavailable");
    return p;
}

template<class F>
void measured(Probe p, F&& operation) {
    p.reset();
    p.enable(1);
    try { operation(); }
    catch (...) { p.enable(0); throw; }
    p.enable(0);
    std::array<char, 32768> json{};
    if (!p.json(json.data(), json.size())) throw std::runtime_error("probe JSON buffer was too small");
    std::cout << "hdf5_counts=" << json.data() << '\n';
}

void check_values(const RecordArrayView& record, std::size_t fields) {
    if (record.shape() != std::vector<std::size_t>{8192} || record.n_fields() != fields)
        throw std::runtime_error("record shape or field count differs");
    for (std::size_t j = 0; j < fields; ++j) {
        const auto name = "field_" + std::to_string(j);
        auto values = record.array(name).dense<double>();
        for (std::size_t i = 0; i < 8192; ++i)
            if (values.data()[i] != double(j) * 100000.0 + double(i % 256))
                throw std::runtime_error("record value differs at field " + name + ", row " + std::to_string(i));
    }
}

int run(std::string_view operation, const char* source, std::size_t fields, const char* output) {
    BackedOptions options;
    auto store = open_h5ad(source, options);
    auto record = store.uns().at("probe");
    if (record.kind() != ElementKind::record_array || record.field_names().size() != fields)
        throw std::runtime_error("fixture was not read as the expected record array");

    if (operation == "field") {
        Element value;
        measured(probe(), [&] { value = record.read_field("field_0"); });
        auto values = value.view().array().dense<double>();
        for (std::size_t i = 0; i < 8192; ++i)
            if (values.data()[i] != double(i % 256)) throw std::runtime_error("field result differs");
    } else if (operation == "full") {
        Element value;
        measured(probe(), [&] { value = record.read(); });
        check_values(value.view().record_array(), fields);
    } else if (operation == "export") {
        if (!output) throw std::runtime_error("export operation needs an output path");
        measured(probe(), [&] { write_h5ad(output, store); });
        auto reopened = open_h5ad(output);
        auto exported = reopened.uns().at("probe").read();
        check_values(exported.view().record_array(), fields);
    } else throw std::runtime_error("operation must be field, full, or export");
    store.close();
    std::cout << "values=exact fields=" << fields << " operation=" << operation << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) throw std::runtime_error("usage: record_io_probe DRIVER-OP SOURCE FIELDS [OUTPUT]");
        const auto fields = std::stoul(argv[3]);
        if (fields != 2 && fields != 8) throw std::runtime_error("fields must be 2 or 8");
        return run(argv[1], argv[2], fields, argc == 5 ? argv[4] : nullptr);
    } catch (const std::exception& e) {
        std::cerr << "record_io_probe: " << e.what() << '\n';
        return 2;
    }
}
