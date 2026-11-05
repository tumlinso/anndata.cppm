#include "zarr_engine.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct alignas(std::max_align_t) Allocation { std::size_t size; bool counted; };
bool measuring=false;
std::size_t live=0,peak=0,ceiling=std::numeric_limits<std::size_t>::max();
void* allocate(std::size_t n) {
    if(measuring && n>ceiling-live) throw std::bad_alloc();
    auto* h=static_cast<Allocation*>(std::malloc(sizeof(Allocation)+std::max<std::size_t>(n,1)));
    if(!h)throw std::bad_alloc();h->size=n;h->counted=measuring;
    if(measuring){live+=n;peak=std::max(peak,live);}return h+1;
}
void release(void* p) noexcept {if(!p)return;auto* h=static_cast<Allocation*>(p)-1;if(h->counted)live-=h->size;std::free(h);}
void check(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
using namespace anndatapp_zarr;

ArrayMetadata fixture(Store& store,const std::filesystem::path& root) {
    store.create_group("",Json::object(),3);
    CreateOptions o;o.format=3;o.compression="zstd";o.compression_level=1;
    store.create_array("X",Type::float32,{9,11},Json::object(),'<',o);
    const auto path=root/"X/zarr.json";Json m;{std::ifstream in(path);in>>m;}
    m["chunk_grid"]["configuration"]["chunk_shape"]={4,6};
    m["codecs"]=Json::array({{{"name","sharding_indexed"},{"configuration",{
        {"chunk_shape",{2,3}},{"codecs",Json::array({{{"name","bytes"},{"configuration",{{"endian","little"}}}},{{"name","zstd"},{"configuration",{{"level",1}}}}})},
        {"index_codecs",Json::array({{{"name","bytes"},{"configuration",{{"endian","little"}}}},{{"name","crc32c"}}})},{"index_location","end"}}}}});
    {std::ofstream out(path);out<<m.dump();}store.flush();return store.prepare_numeric("X");
}
void admission(const std::filesystem::path& root,std::size_t budget,bool fail) {
    std::filesystem::create_directories(root);Store store(root,true,budget);auto m=fixture(store,root);
    std::vector<std::size_t> rows(10000),columns(10000);
    for(std::size_t i=0;i<rows.size();++i){rows[i]=i%9;columns[i]=i%11;}
    float value=7;NumericWrite w;w.axes={{rows.size(),0,1,rows.data()},{columns.size(),0,1,columns.data()}};
    w.data=&value;w.strides={0,0};w.scratch_bytes=budget;
    ceiling=budget;peak=live;measuring=true;bool rejected=false;
    try {store.write_selection("X",m,w);}catch(const Error& e){rejected=e.failure==Failure::argument;}
    measuring=false;
    check(rejected==fail,"unexpected workspace admission result");check(peak<=budget,"controlled live allocation exceeds scratch cap");
    if(fail)check(!std::filesystem::exists(root/"X/c"),"capacity failure created shard payloads");
    else {
        const auto allocations=store.scratch_allocations();const auto first_peak=peak;
        measuring=true;peak=live;store.write_selection("X",m,w);measuring=false;
        check(peak<=budget,"reused write exceeds scratch cap");check(store.scratch_allocations()==allocations,"same-shape write grows scratch capacity");
        std::vector<float> out(99);store.read_numeric("X",{0,0},{9,11},out.data());
        check(std::all_of(out.begin(),out.end(),[](float x){return x==7;}),"gathered broadcast values differ");
        std::cout<<"gather points=20000 logical cells=100000000 budget="<<budget<<" live_peak="<<first_peak<<" reused_peak="<<peak<<'\n';
    }
}
void duplicate_order(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);Store store(root,true,128*1024);auto m=fixture(store,root);
    const auto metadata=root/"zarr.json";
    Json header;{std::ifstream in(metadata);in>>header;}
    header["consolidated_metadata"]["metadata"]["X"]["retained_extension"]=17;
    {std::ofstream out(metadata);out<<header.dump();}
    auto metadata_bytes=[&]{std::ifstream in(metadata);return std::string(std::istreambuf_iterator<char>(in),{});};
    const auto before=metadata_bytes();
    const std::size_t rows[]{2,0,2},columns[]{3,1,3};float values[]{1,2,3,4,5,6,7,8,9};
    NumericWrite w;w.axes={{3,0,1,rows},{3,0,1,columns}};w.data=values+8;w.strides={-12,-4};
    store.write_selection("X",m,w);
    std::vector<float> out(99);store.read_numeric("X",{0,0},{9,11},out.data());
    check(out[1]==5 && out[3]==4 && out[2*11+1]==2 && out[2*11+3]==1,"duplicate destinations lose logical C-order");
    for(std::size_t i=0;i<99;++i)if(i!=1 && i!=3 && i!=23 && i!=25)check(out[i]==0,"unselected cell changed");
    w.axes={{4,7,-2,nullptr},{3,8,-3,nullptr}};w.data=values;w.strides={0,4};
    store.write_selection("X",m,w);store.read_numeric("X",{0,0},{9,11},out.data());
    for(std::size_t i=0;i<4;++i)for(std::size_t j=0;j<3;++j)check(out[(7-2*i)*11+8-3*j]==values[j],"reversed affine destination differs");
    store.flush();check(metadata_bytes()==before,"selected payload flush rewrites consolidated metadata");
    store.write_numeric("X",{0,0},{1,1},values);store.flush();
    check(metadata_bytes()==before,"ordinary payload flush rewrites consolidated metadata");
}
}
void* operator new(std::size_t n){return allocate(n);}void* operator new[](std::size_t n){return allocate(n);}
void operator delete(void* p)noexcept{release(p);}void operator delete[](void* p)noexcept{release(p);}
void operator delete(void* p,std::size_t)noexcept{release(p);}void operator delete[](void* p,std::size_t)noexcept{release(p);}
int main(int argc,char** argv) {
    try {
        auto root=std::filesystem::path(argc>1?argv[1]:"build/shard-selection");
        root/="run-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        std::filesystem::create_directories(root);
        duplicate_order(root/"duplicates");
        check(!live,"live allocation tracking before peak check");admission(root/"admitted",512*1024,false);
        check(!live,"tracked scratch leaked after write owner destruction");admission(root/"rejected",128*1024,true);
        check(!live,"tracked scratch leaked after admission failure");std::cout<<"selected shard writes passed\n";
    } catch(const std::exception& e){measuring=false;std::cerr<<e.what()<<'\n';return 1;}
}
