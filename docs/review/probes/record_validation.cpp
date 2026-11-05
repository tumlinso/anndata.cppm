#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

import anndatapp;

namespace {
using namespace anndatapp;

void require(bool ok, std::string_view message) {
    if (!ok) throw std::runtime_error(std::string(message));
}

template<class F>
bool expect_success(F&& operation, std::string_view label) {
    try {
        operation();
        std::cout << "PASS " << label << '\n';
        return true;
    } catch (const Error& error) {
        std::cout << "FAIL " << label << ": " << error.what() << '\n';
        return false;
    }
}

template<class F>
bool expect_invalid_utf8(F&& operation, std::string_view label) {
    try {
        operation();
    } catch (const Error& error) {
        const std::string_view message(error.what());
        if (error.code() == ErrorCode::invalid_argument &&
            message.find("invalid UTF-8 string") != std::string_view::npos) {
            std::cout << "PASS " << label << ": " << error.what() << '\n';
            return true;
        }
        std::cout << "FAIL " << label << ": expected invalid UTF-8 diagnostic, got "
                  << error.what() << '\n';
        return false;
    }
    std::cout << "FAIL " << label << ": malformed record text was accepted\n";
    return false;
}

void make_valid(const std::filesystem::path& path) {
    auto ids = Array::empty<std::int32_t>({2});
    auto values = ids.edit().dense<std::int32_t>();
    values.data()[0] = 17;
    values.data()[1] = 29;

    auto records = RecordArray::with_shape({2});
    records.set("id", std::move(ids));
    records.set("label", StringArray::from_strings({2}, {"alpha", "βeta"}));

    auto data = AnnData::empty(0, 0);
    data.set_uns("record_validation", Element(std::move(records)));
    write_h5ad(path.string(), data);
}

void check_valid(const std::filesystem::path& source_path,
                 const std::filesystem::path& full_mismatch_path,
                 const std::filesystem::path& none_same_path,
                 const std::filesystem::path& full_same_path) {
    BackedOptions opened;
    opened.validation = Validation::none;
    auto source = open_h5ad(source_path.string(), opened);

    // validate_all() and full export must request full validation independently
    // of the open setting.
    bool passed = expect_success([&] { source.validate_all(); },
                                 "valid rec-array validate_all after Validation::none open");

    WriteOptions written;
    written.validation = Validation::full;
    passed = expect_success([&] { write_h5ad(full_mismatch_path.string(), source, written); },
                            "full export after Validation::none open") && passed;

    WriteOptions none;
    none.validation = Validation::none;
    passed = expect_success([&] { write_h5ad(none_same_path.string(), source, none); },
                            "Validation::none export after Validation::none open") && passed;

    BackedOptions full_open;
    full_open.validation = Validation::full;
    auto source_full = open_h5ad(source_path.string(), full_open);
    passed = expect_success([&] { write_h5ad(full_same_path.string(), source_full, written); },
                            "Validation::full export after Validation::full open") && passed;

    for (const auto& path : {none_same_path, full_same_path}) {
        auto exported = read_h5ad(path.string());
        auto records = exported.uns().at("record_validation").record_array();
        auto ids = records.array("id").dense<std::int32_t>();
        require(records.field_names() == std::vector<std::string>{"id", "label"} &&
                ids.data()[0] == 17 && ids.data()[1] == 29 &&
                records.string("label", 0) == "alpha" &&
                records.string("label", 1) == "βeta",
                "same-mode record export changed its fields or values");
    }
    if (!passed) throw std::runtime_error("valid rec-array validate_all did not pass");
}

void check_invalid(const std::filesystem::path& source_path,
                   const std::filesystem::path& full_mismatch_path,
                   const std::filesystem::path& full_same_path,
                   const std::filesystem::path& none_same_path) {
    BackedOptions opened;
    opened.validation = Validation::none;
    auto source = open_h5ad(source_path.string(), opened);

    bool passed = expect_invalid_utf8([&] { source.validate_all(); },
                                      "validate_all after Validation::none open");

    WriteOptions full;
    full.validation = Validation::full;
    passed = expect_invalid_utf8([&] { write_h5ad(full_mismatch_path.string(), source, full); },
                                 "full export after Validation::none open") && passed;

    // Full HDF5-backed export with the same validation mode is the control: it
    // must get through record dispatch and diagnose the actual malformed text.
    BackedOptions fully_opened;
    fully_opened.validation = Validation::full;
    auto source_full = open_h5ad(source_path.string(), fully_opened);
    passed = expect_invalid_utf8([&] { write_h5ad(full_same_path.string(), source_full, full); },
                                 "full export after Validation::full open") && passed;

    // Same-mode Validation::none export controls for HDF5 record copying while
    // deliberately retaining the malformed bytes.
    WriteOptions none;
    none.validation = Validation::none;
    passed = expect_success([&] { write_h5ad(none_same_path.string(), source, none); },
                            "Validation::none export after Validation::none open") && passed;
    if (!passed) throw std::runtime_error("a full validation operation did not reach the malformed text");
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 6 && std::string_view(argv[1]) == "valid") {
            make_valid(argv[2]);
            check_valid(argv[2], argv[3], argv[4], argv[5]);
            return 0;
        }
        if (argc == 6 && std::string_view(argv[1]) == "invalid") {
            check_invalid(argv[2], argv[3], argv[4], argv[5]);
            return 0;
        }
        throw std::runtime_error("expected: valid SOURCE FULL_MISMATCH NONE_SAME FULL_SAME | invalid SOURCE FULL_MISMATCH FULL_SAME NONE_SAME");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
