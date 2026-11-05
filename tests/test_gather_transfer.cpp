#include "zarr_engine.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

import anndatapp;

using namespace anndatapp;
using anndatapp_zarr::Json;
namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
Json load_json(const std::filesystem::path& p) { std::ifstream in(p); Json j; in >> j; return j; }
void save_json(const std::filesystem::path& p, const Json& j) { std::ofstream out(p); out << j.dump(2); }

struct Fixture {
    std::filesystem::path root;
    std::vector<float> x, cube;
    std::vector<double> weights;
    std::vector<std::int64_t> counts;
    std::array<bool,99> mask{};
};
template<class T> Array array(std::vector<std::size_t> shape, const std::vector<T>& values) {
    auto a = Array::empty<T>(shape); auto* p = a.edit().template dense<T>().data();
    std::copy(values.begin(), values.end(), p); return a;
}
template<class T> std::vector<T> sequence(std::size_t n, T base, T step) {
    std::vector<T> v(n); for (std::size_t i = 0; i < n; ++i) v[i] = base + T(i) * step; return v;
}
void shard_metadata(const std::filesystem::path& root, const std::string& path,
                    const std::vector<std::size_t>& shape,
                    const std::vector<std::size_t>& outer,
                    const std::vector<std::size_t>& inner, Json fill = nullptr) {
    const auto file = root / path / "zarr.json"; auto m = load_json(file);
    m["shape"] = shape; m["chunk_grid"]["configuration"]["chunk_shape"] = outer;
    m["codecs"] = Json::array({Json{{"name","sharding_indexed"},{"configuration",{
        {"chunk_shape",inner},
        {"codecs",Json::array({Json{{"name","bytes"},{"configuration",{{"endian","little"}}}},
                                Json{{"name","zstd"},{"configuration",{{"level",1}}}}})},
        {"index_codecs",Json::array({Json{{"name","bytes"},{"configuration",{{"endian","little"}}}},
                                      Json{{"name","crc32c"}}})},
        {"index_location","end"}}}}});
    if (!fill.is_null()) m["fill_value"] = fill;
    save_json(file,m);
    const auto root_file = root / "zarr.json"; auto r = load_json(root_file);
    if (r.contains("consolidated_metadata")) r["consolidated_metadata"]["metadata"][path] = m;
    save_json(root_file,r);
    std::filesystem::remove_all(root / path / "c");
}
Fixture fixture(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    Fixture f; f.root = path;
    f.x = sequence<float>(99, 0.5f, 3.0f);
    f.weights = sequence<double>(99, -2.25, 0.125);
    f.counts = sequence<std::int64_t>(99, 100000, 17);
    for (std::size_t i = 0; i < f.mask.size(); ++i) f.mask[i] = (i % 3) == 1;
    f.cube = sequence<float>(5*4*7, -10.0f, 0.75f);

    auto d = AnnData::empty(9,11);
    d.set_x(array<float>({9,11},f.x));
    d.set_uns("weights",Element(array<double>({9,11},f.weights)));
    d.set_uns("counts",Element(array<std::int64_t>({9,11},f.counts)));
    auto mask = Array::empty<bool>({9,11}); std::copy(f.mask.begin(),f.mask.end(),mask.edit().dense<bool>().data());
    d.set_uns("mask",Element(std::move(mask)));
    d.set_uns("cube",Element(array<float>({5,4,7},f.cube)));
    d.set_uns("huge",Element(array<std::int64_t>({1,1},std::vector<std::int64_t>{0})));
    d.set_uns("fill",Element(array<double>({5,7},std::vector<double>(35,0.0))));
    auto scalar = Array::empty<std::int32_t>({}); scalar.edit().dense<std::int32_t>().data()[0] = 37;
    d.set_uns("scalar",Element(std::move(scalar)));
    d.set_uns("empty",Element(Array::empty<double>({0,3})));
    ZarrWriteOptions options; options.format = ZarrFormat::v3; write_zarr(path.string(),d,options);

    for (const auto* name : {"X","uns/weights","uns/counts","uns/mask"})
        shard_metadata(path,name,{9,11},{4,6},{2,3});
    shard_metadata(path,"uns/cube",{5,4,7},{4,4,6},{2,2,3});
    shard_metadata(path,"uns/huge",{1000003,1000007},{4,6},{2,3},-7);
    shard_metadata(path,"uns/fill",{5,7},{4,6},{2,3},-19.5);

    anndatapp_zarr::Store store(path,true,128*1024);
    store.write_numeric("X",{0,0},{9,11},f.x.data());
    store.write_numeric("uns/weights",{0,0},{9,11},f.weights.data());
    store.write_numeric("uns/counts",{0,0},{9,11},f.counts.data());
    store.write_numeric("uns/mask",{0,0},{9,11},f.mask.data());
    store.write_numeric("uns/cube",{0,0,0},{5,4,7},f.cube.data());
    const std::int64_t remote = 1234567890123;
    store.write_numeric("uns/huge",{1000002,1000006},{1,1},&remote);
    return f;
}

template<class T> void typed_gather(const BackedArray& a, const T* source, std::size_t width) {
    const std::array<std::size_t,3> rows{7,1,7}; const std::array<std::size_t,4> cols{10,0,5,10};
    auto t = a.select({indices{7,1,7},indices{10,0,5,10}}).prepare_transfer(32*1024);
    std::array<T,12> out{}; t.read_into({0,0},{3,4},DenseBuffer::from(out.data(),{3,4}));
    for (std::size_t i = 0; i < rows.size(); ++i) for (std::size_t j = 0; j < cols.size(); ++j)
        check(out[i*cols.size()+j] == source[rows[i]*width+cols[j]],"typed gathered Zarr values differ");
}
void gathered_x(const Fixture& f, const BackedAnnData& d) {
    const std::array<std::size_t,4> rows{8,0,5,8}; const std::array<std::size_t,5> cols{10,1,6,0,10};
    auto a = d.x().select({indices{8,0,5,8},indices{10,1,6,0,10}});
    auto t = a.prepare_transfer(32*1024);
    check(t.scratch_capacity() == 32*1024,"small transfer budget differs");
    std::array<float,6> block{}; t.read_into({1,1},{2,3},DenseBuffer::from(block.data(),{2,3}));
    for (std::size_t i = 0; i < 2; ++i) for (std::size_t j = 0; j < 3; ++j)
        check(block[i*3+j] == f.x[rows[i+1]*11+cols[j+1]],"partial logical gather differs");
    auto low = a.prepare_transfer(2048); std::array<float,20> fallback{};
    low.read_into({0,0},{4,5},DenseBuffer::from(fallback.data(),{4,5}));
    for (std::size_t i = 0; i < 4; ++i) for (std::size_t j = 0; j < 5; ++j)
        check(fallback[i*5+j] == f.x[rows[i]*11+cols[j]],"low-budget gather differs");

    std::vector<float> column(20,-999.0f);
    a.read_into(DenseBuffer::from(column.data(),{4,5},{std::ptrdiff_t(sizeof(float)),std::ptrdiff_t(4*sizeof(float))}));
    for (std::size_t i = 0; i < 4; ++i) for (std::size_t j = 0; j < 5; ++j)
        check(column[i+j*4] == f.x[rows[i]*11+cols[j]],"column-major output differs");

    std::vector<float> negative(20,-999.0f);
    a.read_into(DenseBuffer::from(negative.data()+19,{4,5},{-std::ptrdiff_t(5*sizeof(float)),-std::ptrdiff_t(sizeof(float))}));
    for (std::size_t i = 0; i < 4; ++i) for (std::size_t j = 0; j < 5; ++j)
        check(negative[(3-i)*5+4-j] == f.x[rows[i]*11+cols[j]],"negative output strides differ");

    std::vector<float> holes(69,-999.0f);
    a.read_into(DenseBuffer::from(holes.data(),{4,5},{std::ptrdiff_t(20*sizeof(float)),std::ptrdiff_t(2*sizeof(float))}));
    for (std::size_t i = 0; i < holes.size(); ++i) {
        bool written = false;
        for (std::size_t r = 0; r < 4; ++r) for (std::size_t c = 0; c < 5; ++c)
            written = written || i == r*20+c*2;
        check(holes[i] == (written ? f.x[rows[(i/20)]*11+cols[(i%20)/2]] : -999.0f),"strided output hole changed");
    }

    float alias = -1;
    a.read_into(DenseBuffer::from(&alias,{4,5},{0,0}));
    check(alias == f.x[rows.back()*11+cols.back()],"zero-stride destination lost last C-order write");
    std::array<float,8> overlap{}; overlap.fill(-999.0f);
    auto expected = overlap;
    for (std::size_t i = 0; i < 4; ++i) for (std::size_t j = 0; j < 5; ++j)
        expected[i+j] = f.x[rows[i]*11+cols[j]];
    a.read_into(DenseBuffer::from(overlap.data(),{4,5},{std::ptrdiff_t(sizeof(float)),std::ptrdiff_t(sizeof(float))}));
    check(overlap == expected,"overlapping output strides changed logical write order");

    auto materialized = a.read(); auto mv = materialized.view().dense<float>();
    check(mv.extent(0) == 4 && mv.extent(1) == 5,"materialized gather shape differs");
    for (std::size_t i = 0; i < 4; ++i) for (std::size_t j = 0; j < 5; ++j)
        check(mv.data()[i*5+j] == f.x[rows[i]*11+cols[j]],"gather materialization differs");

    auto reverse = d.x().select({range{8,1,-2},range{10,0,-3}});
    std::array<float,16> reversed{}; reverse.prepare_transfer(32*1024).read_into({0,0},{4,4},DenseBuffer::from(reversed.data(),{4,4}));
    for (std::size_t i = 0; i < 4; ++i) for (std::size_t j = 0; j < 4; ++j)
        check(reversed[i*4+j] == f.x[(8-2*i)*11+(10-3*j)],"reverse affine gather differs");
}
void higher_ranks_and_dtypes(const Fixture& f, const BackedAnnData& d) {
    typed_gather(d.uns().at("weights").array(),f.weights.data(),11);
    typed_gather(d.uns().at("counts").array(),f.counts.data(),11);
    typed_gather(d.uns().at("mask").array(),f.mask.data(),11);
    auto cube = d.uns().at("cube").array().select({indices{4,0,4},indices{3,1},indices{6,0,6}});
    std::array<float,18> out{}; cube.prepare_transfer(32*1024).read_into({0,0,0},{3,2,3},DenseBuffer::from(out.data(),{3,2,3}));
    const std::array<std::size_t,3> r{4,0,4}; const std::array<std::size_t,2> z{3,1}; const std::array<std::size_t,3> c{6,0,6};
    for (std::size_t i=0;i<3;++i) for (std::size_t j=0;j<2;++j) for (std::size_t k=0;k<3;++k)
        check(out[(i*2+j)*3+k] == f.cube[(r[i]*4+z[j])*7+c[k]],"rank-three gathered values differ");
}
void sparse_huge_and_fill(const BackedAnnData& d) {
    auto huge = d.uns().at("huge").array().select({indices{1000002,0,1000000},indices{1000006,1,1000003}});
    std::array<std::int64_t,9> out{}; huge.prepare_transfer(32*1024).read_into({0,0},{3,3},DenseBuffer::from(out.data(),{3,3}));
    const std::array<std::int64_t,9> expected{1234567890123,-7,-7,-7,-7,-7,-7,-7,-7};
    check(out == expected,"huge sparse gaps or missing inner chunks differ from fill");
    auto fill = d.uns().at("fill").array().select({indices{4,0,4},indices{6,1}});
    std::array<double,6> empty{}; fill.prepare_transfer(32*1024).read_into({0,0},{3,2},DenseBuffer::from(empty.data(),{3,2}));
    check(std::all_of(empty.begin(),empty.end(),[](double x){return x == -19.5;}),"absent shards do not return fill value");
    std::int32_t scalar = 0; auto s = d.uns().at("scalar").array().prepare_transfer(32*1024);
    s.read_into({}, {}, DenseBuffer::from(&scalar,{})); check(scalar == 37,"scalar read differs");
    auto e = d.uns().at("empty").array().prepare_transfer(32*1024);
    e.read_into({0,0},{0,3},DenseBuffer::from(static_cast<double*>(nullptr),{0,3}));
    check(e.stats().read_calls == 0,"empty transfer performs I/O");
}
}
int main(int argc, char** argv) {
    try {
        auto root = std::filesystem::path(argc > 1 ? argv[1] : "gather-transfer-tests");
        root /= "run-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        auto f = fixture(root); auto d = open_zarr(root.string());
        gathered_x(f,d); higher_ranks_and_dtypes(f,d); sparse_huge_and_fill(d);
        std::cout << "gathered sharded transfers passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
