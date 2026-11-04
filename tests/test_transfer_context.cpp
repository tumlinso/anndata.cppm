#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

import anndata;

using namespace anndata;
namespace {
void check(bool v, std::string_view message) { if (!v) throw std::runtime_error(std::string(message)); }
template<class F> void error(ErrorCode code, F&& f) {
    try { f(); } catch (const Error& e) { check(e.code() == code, "wrong error code"); return; }
    throw std::runtime_error("missing expected error");
}
AnnData fixture() {
    auto d = AnnData::empty(4, 5);
    auto x = Array::empty<float>({4,5});
    auto xv = x.edit().dense<float>();
    for (std::size_t i = 0; i < 20; ++i) xv.data()[i] = float(i);
    d.set_x(std::move(x));
    auto csr = Array::csr<float,std::int32_t,std::int64_t>(4,5,6);
    auto a = csr.edit().csr<float,std::int32_t,std::int64_t>();
    const float v[]{0,2,3,4,5,6}; const std::int32_t j[]{2,2,0,4,1,3}; const std::int64_t p[]{0,3,3,5,6};
    for (int i = 0; i < 6; ++i) { a.values()[i] = v[i]; a.unsafe_indices()[i] = j[i]; }
    for (int i = 0; i < 5; ++i) a.unsafe_offsets()[i] = p[i];
    d.set_layer("csr", std::move(csr));
    auto csc = Array::csc<double,std::int64_t,std::int32_t>(4,5,5);
    auto b = csc.edit().csc<double,std::int64_t,std::int32_t>();
    const std::int32_t q[]{0,1,1,4,4,5};
    const std::int64_t k[]{3,0,0,2,1};
    for (int i = 0; i < 5; ++i) { b.values()[i] = double(i); b.unsafe_indices()[i] = k[i]; }
    for (int i = 0; i < 6; ++i) b.unsafe_offsets()[i] = q[i];
    d.set_layer("csc", std::move(csc));
    auto scalar = Array::empty<std::int32_t>({}); scalar.edit().dense<std::int32_t>().data()[0] = 37;
    d.set_uns("scalar", Element(std::move(scalar)));
    d.set_uns("empty", Element(Array::empty<double>({0,3})));
    return d;
}
void dense(BackedAnnData& d) {
    auto t = d.x().prepare_transfer(1024*1024);
    check(t.scratch_capacity() == 1024*1024, "scratch budget differs");
    std::array<float,6> x{};
    t.read_into({1,1},{2,3},DenseBuffer::from(x.data(),{2,3}));
    check(x == std::array<float,6>{6,7,8,11,12,13}, "rectangular read differs");
    const auto allocation = t.stats().scratch_allocations;
    for (int i = 0; i < 3; ++i) t.read_into({1,1},{2,3},DenseBuffer::from(x.data(),{2,3}));
    check(t.stats().scratch_allocations == allocation, "repeated transfer grows scratch");
    t.read_into({1,1},{2,3},DenseBuffer::from(x.data()+5,{2,3},{-12,-4}));
    check(x == std::array<float,6>{13,12,11,8,7,6}, "negative output strides differ");
    auto g = d.x().select({indices{3,0},indices{4,1,4}}).prepare_transfer(1024*1024);
    g.read_into({0,0},{2,3},DenseBuffer::from(x.data(),{2,3}));
    check(x == std::array<float,6>{19,16,19,4,1,4}, "gather ordering/repetitions differ");
    error(ErrorCode::out_of_bounds,[&]{ t.read_into({3,4},{2,1},DenseBuffer::from(x.data(),{2,1})); });
    error(ErrorCode::shape_mismatch,[&]{ t.read_into({0,0},{2,3},DenseBuffer::from(x.data(),{3,2})); });
    error(ErrorCode::invalid_argument,[&]{ auto unused = d.x().prepare_transfer(100); });
    std::int32_t scalar = 0;
    auto s = d.uns().at("scalar").array().prepare_transfer(1024*1024);
    s.read_into({}, {}, DenseBuffer::from(&scalar,{})); check(scalar == 37, "scalar read differs");
    auto e = d.uns().at("empty").array().prepare_transfer(1024*1024);
    e.read_into({0,0},{0,3},DenseBuffer::from(static_cast<double*>(nullptr),{0,3}));
    check(e.stats().read_calls == 0, "empty transfer performs I/O");
    if (d.writable()) {
        std::array<float,2> v{-7,-8};
        t.write_from({1,2},{1,2},DenseBuffer::from(v.data(),{1,2}));
        std::array<float,2> q{};
        t.read_into({1,2},{1,2},DenseBuffer::from(q.data(),{1,2}));
        check(q == v, "direct write differs");
        auto ro = d.view().x().prepare_transfer(1024*1024);
        error(ErrorCode::invalid_argument,[&]{ ro.write_from({1,2},{1,2},DenseBuffer::from(v.data(),{1,2})); });
    }
}
void sparse(BackedAnnData& d) {
    auto t = d.layer("csr").prepare_transfer(1024*1024);
    auto r = t.requirements({1,0},{2,5});
    check(r.layout == Layout::csr && r.nnz == 2 && r.index_dtype == DType::int32 && r.offset_dtype == DType::int64, "CSR requirements differ");
    std::array<float,6> v{}; std::array<std::int32_t,6> j{}; std::array<std::int64_t,5> p{};
    SparseBuffer b{v.data(),j.data(),p.data(),DType::float32,DType::int32,DType::int64,6,6,5};
    const auto before = t.stats().read_calls;
    auto small = b; small.values_capacity = 1;
    error(ErrorCode::shape_mismatch,[&]{ t.read_into({1,0},{2,5},small); });
    check(t.stats().read_calls == before, "insufficient capacity reads numerical payload");
    t.read_into({1,0},{2,5},b);
    check(v[0] == 4 && v[1] == 5 && j[0] == 4 && j[1] == 1 && p[0] == 0 && p[1] == 0 && p[2] == 2, "CSR block differs");
    t.read_into({0,0},{4,5},b);
    check(v[0] == 0 && j[0] == 2 && j[1] == 2 && p[4] == 6, "CSR duplicate/explicit zero differs");
    t.read_into({2,0},{0,5},b); check(p[0] == 0, "empty CSR block differs");
    error(ErrorCode::unsupported_layout,[&]{ t.requirements({0,0},{1,2}); });
    error(ErrorCode::unsupported_layout,[&]{ auto unused = d.layer("csr").select({indices{2,0},all}).prepare_transfer(); });
    auto c = d.layer("csc").prepare_transfer(1024*1024);
    auto q = c.requirements({0,1},{4,2}); check(q.nnz == 3 && q.layout == Layout::csc, "CSC requirements differ");
    std::array<double,3> cv{}; std::array<std::int64_t,3> cj{}; std::array<std::int32_t,3> cp{};
    c.read_into({0,1},{4,2},{cv.data(),cj.data(),cp.data(),DType::float64,DType::int64,DType::int32,3,3,3});
    check(cv == std::array<double,3>{1,2,3} && cj == std::array<std::int64_t,3>{0,0,2} && cp == std::array<std::int32_t,3>{0,0,3}, "CSC block differs");
}
template<class Open, class Write> void run(const std::filesystem::path& path, Open open, Write write) {
    auto data = fixture(); write(path.string(),data);
    BackedOptions options; options.mode = AccessMode::read_write;
    auto d = open(path.string(),options);
    dense(d); sparse(d);
    auto stale = d.x().prepare_transfer();
    d.set("/X",Element(Array::zeros<float>({4,5})));
    error(ErrorCode::invalid_argument,[&]{ stale.geometry(); });
    auto closed = d.x().prepare_transfer(); d.close();
    error(ErrorCode::invalid_argument,[&]{ closed.geometry(); });
    TransferContext retained;
    { auto owner = open(path.string(),BackedOptions{}); retained = owner.x().prepare_transfer(); }
    std::array<float,20> values{};
    retained.read_into({0,0},{4,5},DenseBuffer::from(values.data(),{4,5}));
    check(values[0] == 0 && values[19] == 0, "context did not retain source session");
}
}
int main(int argc,char** argv) {
    static_assert(!std::is_copy_constructible_v<TransferContext> && std::is_nothrow_move_constructible_v<TransferContext>);
    try {
        const auto root = std::filesystem::path(argc > 1 ? argv[1] : "transfer-tests");
        std::filesystem::create_directories(root);
#if ANNDATA_HAS_HDF5
        run(root/"transfer.h5ad",[](auto p,auto o){return open_h5ad(p,o);},[](auto p,const auto& d){write_h5ad(p,d);});
#endif
#if ANNDATA_HAS_ZARR
        for (auto format : {ZarrFormat::v2,ZarrFormat::v3}) {
            run(root/(format == ZarrFormat::v2 ? "transfer2.zarr" : "transfer3.zarr"),[](auto p,auto o){return open_zarr(p,o);},[&](auto p,const auto& d){ZarrWriteOptions o; o.format=format; write_zarr(p,d,o);});
        }
#endif
        std::cout << "transfer contexts passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
