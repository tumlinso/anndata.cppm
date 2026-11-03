#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

import anndata;

namespace {
using namespace anndata;

void require(bool ok, std::string_view message) {
    if (!ok) throw std::runtime_error(std::string(message));
}

const char* code_name(ErrorCode code) {
    switch (code) {
    case ErrorCode::invalid_argument: return "invalid_argument";
    case ErrorCode::out_of_bounds: return "out_of_bounds";
    case ErrorCode::dtype_mismatch: return "dtype_mismatch";
    case ErrorCode::unsupported_layout: return "unsupported_layout";
    case ErrorCode::shape_mismatch: return "shape_mismatch";
    case ErrorCode::invalid_sparse_structure: return "invalid_sparse_structure";
    case ErrorCode::non_contiguous: return "non_contiguous";
    case ErrorCode::io_error: return "io_error";
    case ErrorCode::unsupported_encoding: return "unsupported_encoding";
    case ErrorCode::unsupported_version: return "unsupported_version";
    case ErrorCode::unsupported_dtype: return "unsupported_dtype";
    case ErrorCode::unavailable_filter: return "unavailable_filter";
    }
    return "unknown";
}

RecordArray native_records() {
    auto values = Array::empty<float>({2});
    auto v = values.edit().dense<float>();
    v.data()[0] = 2.5f; v.data()[1] = -3.0f;
    auto r = RecordArray::with_shape({2});
    r.set("value", std::move(values));
    r.set("gene", StringArray::from_strings({2}, {"μ", "gene-β"}));
    return r;
}

void inspect(const AnnData& data) {
    require(data.n_obs() == 8 && data.n_vars() == 5 && data.has_raw_x(), "AnnData dimensions or raw differ");
    require(data.x().layout() == Layout::csr && data.layer("counts").layout() == Layout::csr,
            "sparse X or layer changed");
    auto mixed = data.uns().at("mixed").record_array();
    require(mixed.shape() == std::vector<std::size_t>({2, 3}), "mixed record shape differs");
    require(mixed.field_names() == std::vector<std::string>({"signed", "unsigned", "score", "flag", "label"}),
            "mixed record field order differs");
    require(mixed.field("signed").byte_order == ByteOrder::big &&
            mixed.field("score").byte_order == ByteOrder::big, "record field byte order differs");
    auto s = mixed.member<std::int16_t>("signed");
    auto u = mixed.member<std::uint64_t>("unsigned");
    require(*s.flat_ptr(0) == -3 && *s.flat_ptr(5) == 2 && *u.flat_ptr(0) == (std::uint64_t(1) << 40) &&
            *u.flat_ptr(5) == (std::uint64_t(1) << 40) + 5, "mixed numerical values differ");
    auto flag = mixed.member<bool>("flag");
    require(!*flag.flat_ptr(0) && *flag.flat_ptr(1) && !*flag.flat_ptr(5), "record boolean values differ");
    require(mixed.string("label", 0) == "α" && mixed.string("label", 3).empty() &&
            mixed.string("label", 5) == "other", "mixed UTF-8 values differ");
    auto empty = data.uns().at("empty").record_array();
    require(empty.shape() == std::vector<std::size_t>({0, 2}) && empty.n_fields() == 2 && !empty.size(),
            "empty record shape differs");
    auto scalar = data.uns().at("scalar").record_array();
    require(!scalar.rank() && scalar.size() == 1 && scalar.member<std::int32_t>("a")[0] == 3 &&
            scalar.member<double>("b")[0] == 1.5, "scalar record differs");
    auto fixed = data.uns().at("fixed").record_array();
    require(fixed.string("id", 0) == "AAAA" && fixed.string("id", 1) == "bb" &&
            fixed.member<std::int32_t>("n")[1] == 5, "fixed-width text record differs");
}

struct ScoreRow { float a, b; };
struct NameRow { Utf8Ref a, b; };

void eager(const char* input, const char* output) {
    auto data = read_h5ad(input);
    inspect(data);
    auto rank = data.uns().at("rank_genes_groups").copy();
    auto scores = rank.at("scores").record_array().copy();
    require(scores.view().field_names() == std::vector<std::string>({"A", "B"}), "score field names differ");
    RecordLayout score_layout{sizeof(ScoreRow), alignof(ScoreRow), {
        {"A", RecordFieldKind::numeric, DType::float32, offsetof(ScoreRow, a), sizeof(float), ByteOrder::native},
        {"B", RecordFieldKind::numeric, DType::float32, offsetof(ScoreRow, b), sizeof(float), ByteOrder::native}}};
    auto packed_scores = scores.copy_interleaved(score_layout);
    auto a = packed_scores.edit_member<float>("A");
    require(a.stride_bytes(0) == std::ptrdiff_t(sizeof(ScoreRow)), "score stride differs");
    const auto* first = packed_scores.view().record_bytes().data();
    require(reinterpret_cast<const std::byte*>(a.data()) == first + offsetof(ScoreRow, a),
            "member acquisition did not point to record storage");
    a[0] += 10.0f;

    auto names = rank.at("names").record_array().copy();
    RecordLayout text_layout{sizeof(NameRow), alignof(NameRow), {
        {"A", RecordFieldKind::utf8_ref, DType::uint8, offsetof(NameRow, a), sizeof(Utf8Ref), ByteOrder::native},
        {"B", RecordFieldKind::utf8_ref, DType::uint8, offsetof(NameRow, b), sizeof(Utf8Ref), ByteOrder::native}}};
    auto packed_names = names.copy_interleaved(text_layout);
    auto refs = packed_names.unsafe_edit_text_refs("A");
    constexpr std::string_view changed = "gene-β";
    refs[0] = {changed.data(), changed.size()};
    auto copied_names = packed_names.copy();
    auto copied_ref = copied_names.view().member<Utf8Ref>("A");
    require(copied_names.view().string("A", 0) == changed && copied_ref[0].data != changed.data(),
            "record copy did not own and retarget referenced text");

    rank.set("scores", Element(std::move(packed_scores)));
    rank.set("names", Element(std::move(copied_names)));
    auto uns = data.uns().copy();
    uns.set("rank_genes_groups", std::move(rank));
    uns.set("native_records", Element(native_records()));
    data.set_uns(std::move(uns));
    write_h5ad(output, data, WriteOptions{.compression = Compression::gzip});
}

void backed(const char* input, const char* output) {
    BackedOptions options;
    options.mode = AccessMode::read_write;
    auto store = open_h5ad(input, options);
    auto scores = store.uns().at("rank_genes_groups").at("scores");
    require(scores.kind() == ElementKind::record_array &&
            scores.field_names() == std::vector<std::string>({"A", "B"}), "backed score schema differs");
    auto field = scores.read_field("B");
    auto values = field.view().array().dense<float>();
    const float old = values.data()[1];
    auto all = scores.read();
    auto record = all.view().record_array().copy();
    auto b = record.edit_member<float>("B");
    require(b[1] == old, "single-field and full-record reads disagree");
    b[1] -= 5.0f;
    store.set("/uns/rank_genes_groups/scores", Element(std::move(record)));
    bool stale = false;
    try { (void)scores.read(); }
    catch (const Error& e) { stale = e.code() == ErrorCode::invalid_argument; }
    require(stale, "replaced backed record descriptor was not invalidated");

    auto names = store.uns().at("rank_genes_groups").at("names");
    auto selected = names.read_field("A");
    auto first = selected.view().string_array().string(0);
    require(!first.empty(), "backed string field was empty");
    store.set("/uns/native_records", Element(native_records()));
    write_h5ad(output, store, WriteOptions{.compression = Compression::gzip});
    store.close();
    require(values.data()[1] == old && selected.view().string_array().string(0) == first,
            "materialized backed field did not survive close");
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "features") {
            std::cout << "zarr=" << ANNDATA_HAS_ZARR << '\n';
            return 0;
        }
        if (argc < 3) throw std::runtime_error("missing record command or file");
        std::string_view cmd = argv[1];
        if (cmd == "eager" && argc == 4) eager(argv[2], argv[3]);
        else if (cmd == "backed" && argc == 4) backed(argv[2], argv[3]);
        else if (cmd == "fixed-eager" && argc == 4) {
            auto data = read_h5ad(argv[2]);
            auto rec = data.uns().at("fixed_utf8").record_array();
            require(rec.string("id", 0) == "λ" && rec.string("id", 1) == "😀", "fixed UTF-8 values differ");
            write_h5ad(argv[3], data);
        } else if (cmd == "fixed-backed" && argc == 4) {
            auto data = open_h5ad(argv[2]);
            auto selected = data.uns().at("fixed_utf8").read_field("id");
            auto text = selected.view().string_array();
            require(text.string(0) == "λ" && text.string(1) == "😀", "backed fixed UTF-8 values differ");
            write_h5ad(argv[3], data);
        } else if (cmd == "read" && argc == 4) (void)read_h5ad(argv[2]);
        else if (cmd == "backed-read" && argc == 4) {
            auto data = open_h5ad(argv[2]);
            (void)data.uns().at(argv[3]).read();
        } else if (cmd == "budget" && argc == 3) {
            auto data = open_h5ad(argv[2], BackedOptions{.scratch_bytes = 512});
            auto rec = data.uns().at("oversized");
            auto small = rec.read_field("value");
            require(small.view().array().dense<std::int32_t>().data()[0] == 37,
                    "projected read touched unrelated oversized text");
            (void)rec.read_field("text");
#if ANNDATA_HAS_ZARR
        } else if ((cmd == "zarr-eager" || cmd == "zarr-backed") && argc == 5) {
            ZarrWriteOptions options;
            options.format = std::string_view(argv[4]) == "2" ? ZarrFormat::v2 : ZarrFormat::v3;
            if (cmd == "zarr-eager") write_zarr(argv[3], read_h5ad(argv[2]), options);
            else write_zarr(argv[3], open_h5ad(argv[2]), options);
#endif
        } else throw std::runtime_error("unknown record command");
        return 0;
    } catch (const Error& e) {
        std::cerr << "anndata error[" << code_name(e.code()) << "] " << e.path() << ": " << e.what() << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; }
    return 1;
}
