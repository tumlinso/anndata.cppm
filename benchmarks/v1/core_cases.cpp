#include "harness.hpp"
#include <algorithm>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

import anndatapp;

namespace v1 {
namespace {
using namespace anndatapp;

void direct() {
    constexpr std::size_t n = 4 * 1024 * 1024, reps = 8, workers = 4;
    std::vector<float> x(n), serial(n), parallel(n);
    for (std::size_t i = 0; i < n; ++i) { x[i] = float(i % 31); serial[i] = parallel[i] = 1; }
    auto s = measure([&] {
        for (std::size_t r = 0; r < reps; ++r)
            for (std::size_t i = 0; i < n; ++i) serial[i] += 0.5f * x[i];
    });
    emit("pointer_saxpy_serial", reps, s, n * sizeof(float));
    std::barrier start(workers + 1), done(workers + 1);
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < workers; ++t) threads.emplace_back([&, t] {
        start.arrive_and_wait();
        for (std::size_t r = 0; r < reps; ++r)
            for (std::size_t i = t * n / workers; i < (t + 1) * n / workers; ++i)
                parallel[i] += 0.5f * x[i];
        done.arrive_and_wait();
    });
    auto p = measure([&] { start.arrive_and_wait(); done.arrive_and_wait(); });
    for (auto& thread : threads) thread.join();
    emit("pointer_saxpy_4_workers", reps, p, n * sizeof(float));
    for (std::size_t i = 0; i < n; ++i)
        require(serial[i] == 1 + 4 * x[i] && parallel[i] == serial[i], "SAXPY output");
    sink = static_cast<std::uint64_t>(serial[n - 1]);
}

void acquisition() {
    constexpr std::size_t reps = 10000;
    std::vector<float> raw(512 * 512, 2);
    auto a = Array::borrow_dense(raw.data(), {512, 512}, {512, 1});
    auto view = a.view();
    std::uint64_t total = 0;
    auto checked = measure([&] { for (std::size_t i = 0; i < reps; ++i) total += view.dense<float>().extent(0); });
    emit("checked_dense", reps, checked);
    auto unsafe = measure([&] { for (std::size_t i = 0; i < reps; ++i) total += view.unsafe_dense<float>().extent(0); });
    emit("unsafe_dense", reps, unsafe);
    const std::size_t shape[] = {512, 512};
    const std::ptrdiff_t strides[] = {512, 1};
    auto borrowed = measure([&] {
        for (std::size_t i = 0; i < reps; ++i) {
#if V1_NEW_CORE
            auto v = borrowed_dense<float>(raw.data(), 2, shape, strides, Validation::full);
#else
            auto owner = Array::borrow_dense(raw.data(), {shape[0], shape[1]}, {strides[0], strides[1]}, Validation::full);
            auto v = owner.view().dense<float>();
#endif
            total += v.extent(0);
        }
    });
#if V1_NEW_CORE
    emit("borrowed_dense", reps, borrowed, 0, -1, -1, "free_borrowed_dense");
#else
    emit("borrowed_dense", reps, borrowed, 0, -1, -1, "owner_view_dense_fallback");
#endif
    std::vector<std::size_t> pos(256);
    for (std::size_t i = 0; i < pos.size(); ++i) pos[i] = (i * 17) % 512;
    auto gather = Axis::gather(pos.data(), pos.size());
    auto affine = Axis::stride(0, 256, 2);
    auto sa = measure([&] { for (std::size_t i = 0; i < reps; ++i) total += view.select({affine, Axis::all()}).extent(0); });
    emit("affine_select", reps, sa);
    auto sg = measure([&] { for (std::size_t i = 0; i < reps; ++i) total += view.select({gather, Axis::all()}).extent(0); });
    emit("gather_select", reps, sg);
    auto ad = AnnData::empty(16, 0);
    for (std::size_t i = 0; i < 128; ++i) {
        auto values = Array::zeros<double>({16});
        values.edit().dense<double>().data()[0] = double(i);
        ad.set_obs_column("c" + std::to_string(i), Column::numeric(std::move(values)));
    }
    auto focused = measure([&] { for (std::size_t i = 0; i < reps; ++i) total += ad.obs_column("c63").size(); });
    emit("obs_column_focused_128", reps, focused);
    auto whole = measure([&] { for (std::size_t i = 0; i < reps; ++i) total += ad.obs().column("c63").size(); });
    emit("obs_column_whole_128", reps, whole);
    require(total == reps * (512 * 3 + 256 * 2 + 16 * 2), "acquisition extents");
    require(ad.obs_column("c63").values().dense<double>().data()[0] == 63, "focused column value");
    sink = total;
}

void dense_copy() {
    constexpr std::size_t n = 512, m = 256, reps = 12;
    auto a = Array::empty<float>({n, n});
    auto* p = a.edit().dense<float>().data();
    for (std::size_t i = 0; i < n * n; ++i) p[i] = float(i);
    std::vector<std::size_t> positions(m);
    for (std::size_t i = 0; i < m; ++i) positions[i] = (i * 17) % n;
    auto g = Axis::gather(positions.data(), m);
    auto v = a.view();
    auto slab = Array::borrow_dense(p, {m, m}, {std::ptrdiff_t(m), 1});
    auto contiguous = slab.view();
    auto strided = v.select({Axis::stride(0, m, 2), Axis::stride(0, m, 2)});
    auto gathered = v.select({g, g});
    auto run = [&](const char* label, const ArrayView& source, int mode, std::size_t cols) {
        auto sample = measure([&] { for (std::size_t i = 0; i < reps; ++i) { auto copy = source.copy(); sink = copy.extent(0); } });
        emit(label, reps, sample, m * cols * sizeof(float));
        auto out = source.copy();
        auto d = out.view().dense<float>();
        require(d.extent(0) == m && d.extent(1) == cols && d.contiguous(), "dense copy shape");
        for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < cols; ++j) {
            auto row = mode == 0 ? i : mode == 1 ? 2 * i : positions[i];
            auto col = mode == 0 ? j : mode == 1 ? 2 * j : positions[j];
            require(d.data()[i * cols + j] == p[row * (mode == 0 ? m : n) + col], "dense copy value");
        }
        sink = static_cast<std::uint64_t>(d.data()[m * cols - 1]);
        auto target = Array::empty<float>({m, cols});
        auto edit = target.edit();
        auto into = measure([&] { for (std::size_t i = 0; i < reps; ++i) source.copy_into(edit); });
        const auto into_label = std::string(label) + "_into";
        emit(into_label.c_str(), reps, into, m * cols * sizeof(float));
        const auto result = target.view().dense<float>();
        for (std::size_t i = 0; i < m * cols; ++i)
            require(result.data()[i] == d.data()[i], "caller-provided dense copy value");
    };
    run("dense_copy_contiguous", contiguous, 0, m);
    run("dense_copy_strided", strided, 1, m);
    run("dense_copy_gather", gathered, 2, m);
}

void sparse() {
    constexpr std::size_t n = 512, k = 8, m = 256, reps = 8;
    std::vector<float> values(n * k);
    std::vector<std::int32_t> indices(n * k);
    std::vector<std::int64_t> offsets(n + 1);
    // Each major group has unsorted minor indices and a deliberate duplicate.
    for (std::size_t i = 0; i < n; ++i) {
        offsets[i] = std::int64_t(i * k);
        for (std::size_t j = 0; j < k; ++j) {
            indices[i * k + j] = std::int32_t((i * 11 + (j == 7 ? 1 : j) * 37) % n);
            values[i * k + j] = float(1 + (i * k + j) % 23);
        }
    }
    offsets[n] = n * k;
    std::vector<std::size_t> positions(m);
    for (std::size_t i = 0; i < m; ++i) positions[i] = (i * 17) % n;
    positions[1] = positions[0]; // duplicate selection, in non-monotone order
    auto g = Axis::gather(positions.data(), m);
    for (auto layout : {Layout::csr, Layout::csc}) {
        auto source = Array::borrow_sparse(layout, n, n, n * k, values.data(), indices.data(), offsets.data());
        for (int native = 0; native < 2; ++native) {
            bool select_rows = (layout == Layout::csr) == bool(native);
            auto selected = source.view().select({select_rows ? g : Axis::all(), select_rows ? Axis::all() : g});
            auto sample = measure([&] { for (std::size_t i = 0; i < reps; ++i) { auto copy = selected.copy(); sink = copy.extent(0); } });
            const char* label = layout == Layout::csr ? (native ? "csr_native_gather_copy" : "csr_opposite_gather_copy")
                                                       : (native ? "csc_native_gather_copy" : "csc_opposite_gather_copy");
            emit(label, reps, sample, values.size() * sizeof(float) + indices.size() * sizeof(std::int32_t) + offsets.size() * sizeof(std::int64_t));
            auto out = selected.copy();
            auto sv = layout == Layout::csr ? out.view().csr<float, std::int32_t, std::int64_t>()
                                             : out.view().csc<float, std::int32_t, std::int64_t>();
            const auto rows = select_rows ? m : n, cols = select_rows ? n : m;
            require(sv.rows() == rows && sv.columns() == cols, "sparse copy shape");
            std::vector<float> expected(rows * cols, 0), actual(rows * cols, 0);
            for (std::size_t r = 0; r < rows; ++r) for (std::size_t c = 0; c < cols; ++c) {
                auto sr = select_rows ? positions[r] : r, sc = select_rows ? c : positions[c];
                auto major = layout == Layout::csr ? sr : sc, minor = layout == Layout::csr ? sc : sr;
                for (std::size_t j = 0; j < k; ++j) if (std::size_t(indices[major * k + j]) == minor)
                    expected[r * cols + c] += values[major * k + j];
            }
            auto major_count = layout == Layout::csr ? rows : cols;
            // Check raw multiplicity and order, not only coordinate sums.
            std::size_t q_expected = 0;
            for (std::size_t major = 0; major < major_count; ++major) {
                require(std::size_t(sv.offsets()[major]) == q_expected, "sparse offset order");
                auto source_major = native ? positions[major] : major;
                for (std::size_t j = 0; j < k; ++j) {
                    auto q = source_major * k + j;
                    if (native) {
                        require(q_expected < sv.nnz() && sv.indices()[q_expected] == indices[q] && sv.values()[q_expected] == values[q], "sparse native entry order");
                        ++q_expected;
                    } else for (std::size_t target = 0; target < m; ++target) if (positions[target] == std::size_t(indices[q])) {
                        require(q_expected < sv.nnz() && sv.indices()[q_expected] == std::int32_t(target) && sv.values()[q_expected] == values[q], "sparse opposite entry order");
                        ++q_expected;
                    }
                }
            }
            require(q_expected == sv.nnz() && std::size_t(sv.offsets()[major_count]) == q_expected, "sparse entry count");
            for (std::size_t major = 0; major < major_count; ++major)
                for (auto q = sv.offsets()[major]; q < sv.offsets()[major + 1]; ++q) {
                    auto minor = std::size_t(sv.indices()[q]);
                    auto r = layout == Layout::csr ? major : minor, c = layout == Layout::csr ? minor : major;
                    require(r < rows && c < cols, "sparse copy index bounds");
                    actual[r * cols + c] += sv.values()[q];
                }
            require(actual == expected, "sparse selected copy values and duplicate semantics");
            sink = sv.nnz();
        }
    }
}

void strings() {
    constexpr std::size_t n = 16384, reps = 8;
    for (int utf8 = 0; utf8 < 2; ++utf8) {
        std::vector<std::string> strings(n);
        for (std::size_t i = 0; i < n; ++i) strings[i] = (utf8 ? "β-café-" : "cell-") + std::to_string(i % 997);
        auto a = StringArray::from_strings({n}, strings);
        auto view = a.view();
        auto sample = measure([&] { for (std::size_t i = 0; i < reps; ++i) { auto copy = view.copy(); sink = copy.size(); } });
        emit(utf8 ? "packed_utf8_copy" : "packed_ascii_copy", reps, sample, view.byte_size());
        auto out = view.copy();
        auto result = out.view();
        require(result.size() == n, "string copy size");
        for (std::size_t i = 0; i < n; ++i) require(result.string(i) == strings[i], "string copy value");
        sink = result.byte_size();
    }
    std::vector<std::string> input(n);
    std::vector<std::uint8_t> valid(n);
    for (std::size_t i = 0; i < n; ++i) { input[i] = "gene-" + std::to_string(i % 997); valid[i] = i % 7 != 0; }
    auto column = Column::nullable_strings(input, valid);
    auto view = column.view();
    auto sample = measure([&] { for (std::size_t i = 0; i < reps; ++i) { auto copy = view.copy(); sink = copy.size(); } });
    emit("nullable_strings_copy", reps, sample);
    auto out = view.copy();
    auto result = out.view();
    require(result.size() == n && result.nullable(), "nullable string copy shape");
    for (std::size_t i = 0; i < n; ++i) require(result.valid(i) == bool(valid[i]) && result.string(i) == input[i], "nullable string copy value");
    sink = result.size();
}

void metadata() {
    constexpr std::size_t n = 16384, repeats = 8, acq_reps = 10000;
    auto codes = Array::empty<std::int32_t>({n});
    auto* p = codes.edit().dense<std::int32_t>().data();
    for (std::size_t i = 0; i < n; ++i) p[i] = std::int32_t(i % 32);
    std::vector<std::string> categories(32);
    for (std::size_t i = 0; i < 32; ++i) categories[i] = "group-" + std::to_string(i);
    auto column = Column::categorical(std::move(codes), Column::strings(categories));
    std::vector<std::size_t> positions(n / 2);
    for (std::size_t i = 0; i < positions.size(); ++i) positions[i] = i * 2;
    auto axis = Axis::gather(positions.data(), positions.size());
    auto sample = measure([&] { for (std::size_t i = 0; i < repeats; ++i) { auto copy = column.view().select(axis, true).copy(); sink = copy.size(); } });
    emit("categorical_prune_copy", repeats, sample);
    auto out = column.view().select(axis, true).copy();
    auto result = out.view();
    require(result.size() == n / 2 && result.categories().size() == 16, "categorical pruned size");
    for (std::size_t i = 0; i < n / 2; ++i)
        require(result.categories().string(std::size_t(result.code(i))) == categories[positions[i] % 32], "categorical pruned values");
    auto records = RecordArray::with_shape({16});
    for (std::size_t i = 0; i < 128; ++i) {
        auto a = Array::zeros<double>({16});
        a.edit().dense<double>().data()[0] = double(i);
        records.set("f" + std::to_string(i), std::move(a));
    }
    const auto& owner = records;
    std::uint64_t total = 0;
    auto focused = measure([&] {
        for (std::size_t i = 0; i < acq_reps; ++i) {
#if V1_NEW_CORE
            auto member = owner.member<double>("f63");
#else
            auto member = owner.view().member<double>("f63");
#endif
            total += std::uint64_t(member[0]);
        }
    });
#if V1_NEW_CORE
    emit("record_member_focused_128", acq_reps, focused, 0, -1, -1, "owner_member");
#else
    emit("record_member_focused_128", acq_reps, focused, 0, -1, -1, "view_member_fallback");
#endif
    auto whole = measure([&] { for (std::size_t i = 0; i < acq_reps; ++i) total += std::uint64_t(owner.view().member<double>("f63")[0]); });
    emit("record_member_whole_128", acq_reps, whole);
    require(total == acq_reps * 63 * 2, "record member acquisition");
    sink = total;
}
}
void core_case(const std::string& name) {
    if (name == "direct") direct();
    else if (name == "acquisition") acquisition();
    else if (name == "dense_copy") dense_copy();
    else if (name == "sparse") sparse();
    else if (name == "strings") strings();
    else if (name == "metadata") metadata();
    else require(false, "unknown core case");
}
}
