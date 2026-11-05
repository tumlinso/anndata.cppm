#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

import anndatapp;

namespace fs = std::filesystem;
using namespace anndatapp;

namespace {
int failures = 0;
void check(bool ok,const char* message) {
    if (!ok) { std::cerr << message << '\n'; ++failures; }
}
template<class F> void rejects(F&& f) {
    try { f(); check(false,"Zarr accepted a record"); }
    catch (const Error& e) { check(e.code()==ErrorCode::unsupported_encoding,"wrong record error code"); }
}
std::string read(const fs::path& p) {
    std::ifstream f(p,std::ios::binary);
    return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
void intact(const fs::path& p) {
    check(read(p/"keep")=="original bytes","record rejection changed destination");
    check(std::distance(fs::directory_iterator(p),fs::directory_iterator())==1,"record rejection created store files");
}
auto snapshot(const fs::path& root) {
    std::vector<std::pair<std::string,std::string>> result;
    for (const auto& e : fs::recursive_directory_iterator(root))
        if (e.is_regular_file()) result.emplace_back(fs::relative(e.path(),root).string(),read(e.path()));
    std::sort(result.begin(),result.end()); return result;
}
template<class F> void invalid_root(F&& f) {
    try { f(); check(false,"Zarr accepted an invalid root path"); }
    catch (const Error& e) { check(e.code()==ErrorCode::invalid_argument,"wrong invalid-root error code"); }
}
void root_paths(const fs::path& root) {
    auto data = AnnData::empty(1,1);
    const auto path = root/"root-paths";
    write_zarr(path.string(),data);
    auto source = open_zarr(path.string());
    const auto original = snapshot(path);
    std::string nul = path.string(); nul.append("\0suffix",7);
    for (const auto& invalid : {std::string{},nul}) {
        invalid_root([&] { (void)open_zarr(invalid); });
        invalid_root([&] { (void)read_zarr(invalid); });
        for (auto mode : {ZarrWriteMode::truncate,ZarrWriteMode::exclusive}) {
            ZarrWriteOptions options; options.mode = mode;
            invalid_root([&] { write_zarr(invalid,data,options); });
            invalid_root([&] { write_zarr(invalid,data.view(),options); });
            invalid_root([&] { write_zarr(invalid,source,options); });
            invalid_root([&] { write_zarr(invalid,source.view(),options); });
        }
    }
    check(snapshot(path)==original,"invalid root path changed the source/prefix store");
    check(source.n_obs()==1 && source.n_vars()==1,"invalid root path invalidated source");
}
}

int main(int argc,char** argv) {
    if (argc!=2) return 2;
    try {
        const fs::path root(argv[1]); fs::create_directories(root);
        root_paths(root);
        auto data=AnnData::empty(2,1);
        auto record=RecordArray::with_shape({2});
        record.set("score",Array::zeros<float>({2}));
        auto nested=Element::mapping(); nested.set("record",Element(std::move(record)));
        data.set_uns("nested",std::move(nested));
        for (auto format:{ZarrFormat::v2,ZarrFormat::v3}) {
            const auto p=root/(format==ZarrFormat::v2?"v2":"v3");
            fs::create_directories(p); { std::ofstream f(p/"keep"); f << "original bytes"; }
            ZarrWriteOptions options; options.format=format;
            rejects([&] { write_zarr(p.string(),data,options); }); intact(p);
            auto absent=p; absent+="-new";
            rejects([&] { write_zarr(absent.string(),data.view(),options); });
            check(!fs::exists(absent),"record rejection created a new destination");
#if ANNDATAPP_HAS_HDF5
            const auto source=root/(format==ZarrFormat::v2?"v2-source.h5ad":"v3-source.h5ad");
            write_h5ad(source.string(),data);
            auto backed=open_h5ad(source.string());
            rejects([&] { write_zarr(p.string(),backed.view(),options); }); intact(p);
            rejects([&] { write_zarr(absent.string(),backed,options); });
            check(!fs::exists(absent),"backed record rejection created a destination");
            check(backed.uns().at("nested").at("record").field_names().size()==1,"rejection changed source");
#endif
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    return failures?1:0;
}
