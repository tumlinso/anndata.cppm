#include <hdf5.h>
#include <dlfcn.h>
#include "h5ad_record_scratch.hpp"
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

import anndatapp;

namespace {
bool observe_record_reads = false;
std::size_t successful_record_reads = 0, failed_record_reads = 0;
std::size_t largest_record_prefix = 0, peak_record_staging = 0;
// Test-only observation of actual HDF5 requests and the registered allocator.
// Public numerical loops contain no instrumentation.
void observe_record_read(hid_t dataset,hid_t memory,hid_t transfer,herr_t status) {
    char path[64]{};
    if (observe_record_reads && H5Iget_name(dataset,path,sizeof(path)) >= 0 && std::string_view(path) == "/uns/budget") {
        const auto n = std::size_t(H5Sget_select_npoints(memory));
        if (status >= 0) { ++successful_record_reads; largest_record_prefix = std::max(largest_record_prefix,n); }
        else ++failed_record_reads;
        H5MM_allocate_t alloc = nullptr; H5MM_free_t release = nullptr;
        void* context = nullptr; void* free_context = nullptr;
        if (transfer != H5P_DEFAULT && H5Pget_vlen_mem_manager(transfer,&alloc,&context,&release,&free_context) >= 0 &&
            alloc == anndatapp_h5ad_detail::RecordVlenAllocator::allocate) {
            const auto& a = *static_cast<anndatapp_h5ad_detail::RecordVlenAllocator*>(context);
            const auto metadata = sizeof(a)+sizeof(std::unique_ptr<char*[]>)+n*sizeof(char*);
            peak_record_staging = std::max(peak_record_staging,metadata+a.peak);
            if (a.peak > a.limit || metadata+a.limit > anndatapp_h5ad_detail::record_cap) std::abort();
        }
    }
}
}
#ifndef __APPLE__
extern "C" herr_t H5Dread(hid_t dataset,hid_t type,hid_t memory,hid_t file,hid_t transfer,void* data) {
    using Read = herr_t (*)(hid_t,hid_t,hid_t,hid_t,hid_t,void*);
    static auto read = reinterpret_cast<Read>(dlsym(RTLD_NEXT,"H5Dread"));
    if (!read) std::abort();
    const auto status = read(dataset,type,memory,file,transfer,data);
    observe_record_read(dataset,memory,transfer,status); return status;
}
#endif

namespace {
using namespace anndatapp;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void fails(F f, ErrorCode code) {
    try { f(); } catch (const Error& e) { require(e.code() == code, "wrong error code"); return; }
    throw std::runtime_error("expected anndatapp error");
}
template<class T> Array values(std::vector<std::size_t> shape, std::initializer_list<T> x) {
    auto a = Array::empty<T>(shape); auto v = a.edit().template dense<T>();
    std::size_t i = 0; for (auto y : x) v.data()[i++] = y;
    std::size_t n = 1; for (auto d : shape) n *= d;
    require(i == n, "fixture length mismatch"); return a;
}
RecordArray mixed() {
    auto r = RecordArray::with_shape({2,2});
    auto ids = values<std::int16_t>({2,2}, {-2,3,4,5}); ids.set_serialization_byte_order(ByteOrder::big);
    r.set("id", std::move(ids)); r.set("p/value", values<double>({2,2}, {0.1,0.2,0.3,0.4}));
    r.set("name", StringArray::from_strings({2,2}, {"α", "", "longer-gene", "δ"}));
    return r;
}
void check(RecordArrayView r) {
    require(r.shape() == std::vector<std::size_t>({2,2}), "record shape mismatch");
    require(r.field_names() == std::vector<std::string>({"id","p/value","name"}), "record fields reordered");
    require(r.field("id").byte_order == ByteOrder::big, "record byte order lost");
    auto id = r.member<std::int16_t>("id"); auto p = r.member<double>("p/value");
    require(*id.flat_ptr(0) == -2 && *id.flat_ptr(3) == 5, "record integer values differ");
    require(*p.flat_ptr(0) == 0.1 && *p.flat_ptr(3) == 0.4, "record floating values differ");
    require(r.string("name",0) == "α" && r.string("name",2) == "longer-gene", "record text differs");
}
AnnData fixture() {
    auto d = AnnData::empty(2,2);
    auto x = Array::csr<float,std::int32_t,std::int64_t>(2,2,3);
    auto s = x.edit().csr<float,std::int32_t,std::int64_t>();
    s.values()[0]=1; s.values()[1]=0; s.values()[2]=3;
    s.unsafe_indices()[0]=1; s.unsafe_indices()[1]=0; s.unsafe_indices()[2]=0;
    s.unsafe_offsets()[0]=0; s.unsafe_offsets()[1]=1; s.unsafe_offsets()[2]=3;
    d.set_x(std::move(x)); d.set_uns("mixed", Element(mixed()));
    auto empty = RecordArray::with_shape({0,2});
    empty.set("score", Array::empty<float>({0,2}));
    empty.set("name", StringArray::from_strings(std::vector<std::size_t>{0,2}, std::vector<std::string>{}));
    d.set_uns("empty", Element(std::move(empty)));
    auto scalar = RecordArray::with_shape({}); scalar.set("value", values<std::uint64_t>({}, {77}));
    d.set_uns("scalar", Element(std::move(scalar)));
    d.set_raw(d.x().copy(), d.var().copy()); return d;
}
struct Row { std::int16_t id; double score; Utf8Ref name; char code[8]; };
RecordLayout layout() {
    return {sizeof(Row),alignof(Row),{
        {"id",RecordFieldKind::numeric,DType::int16,offsetof(Row,id),0,ByteOrder::big},
        {"score",RecordFieldKind::numeric,DType::float64,offsetof(Row,score)},
        {"name",RecordFieldKind::utf8_ref,DType::uint8,offsetof(Row,name)},
        {"code",RecordFieldKind::fixed_utf8,DType::uint8,offsetof(Row,code),8}}};
}
void interleaved(const std::filesystem::path& root) {
    Row x[]{{1,1.5,{"α",2},"A"},{2,2.5,{"β",2},"B"},{3,3.5,{"γ",2},"C"},{4,4.5,{"δ",2},"D"}};
    auto r = RecordArray::borrow_interleaved(x+1,{2,2},layout(),{2*std::ptrdiff_t(sizeof(Row)),-std::ptrdiff_t(sizeof(Row))});
    auto d = AnnData::empty(0,0); d.set_uns("rows",Element(std::move(r)));
    auto path = root / "interleaved.h5ad"; write_h5ad(path.string(),d);
    auto loaded = read_h5ad(path.string()); auto v = loaded.uns().at("rows").record_array();
    auto id = v.member<std::int16_t>("id");
    require(*id.flat_ptr(0)==2 && *id.flat_ptr(1)==1 && *id.flat_ptr(2)==4 && *id.flat_ptr(3)==3,"signed record strides lost");
    require(v.string("name",0)=="β" && v.string("code",3)=="C","interleaved text differs");
    auto packed = v.copy_interleaved(layout());
    auto b = open_h5ad(path.string(),{.mode=AccessMode::read_write});
    b.set("/uns/packed",Element(std::move(packed)));
    auto back = b.uns().at("packed").read();
    require(back.view().record_array().string("name",0)=="β","backed interleaved replacement failed");
    b.close();
}
void scratch_limits(const std::filesystem::path& root) {
    using namespace anndatapp_h5ad_detail;
    require(record_pointer_limit(8,32)==0,"pointer metadata admitted before budget check");
    RecordVlenAllocator small{128};
    auto* first=RecordVlenAllocator::allocate(32,&small);
    require(first && small.used==sizeof(RecordVlenAllocator::Header)+32,"allocator bytes not counted");
    require(!RecordVlenAllocator::allocate(128,&small) && small.exceeded && small.peak<=small.limit,"allocator exceeded byte cap");
    RecordVlenAllocator::release(first,&small);
    require(!small.used && !small.first,"allocator release left live staging");

    const auto roundtrip=[&](std::size_t n,std::size_t width,const char* filename) {
        std::vector<char> bytes(n*width,'x'); std::vector<std::uint64_t> offsets(n+1);
        for(std::size_t i=0;i<n;++i) { bytes[i*width]=char('A'+i%26); offsets[i]=i*width; }
        offsets[n]=bytes.size();
        auto r=RecordArray::with_shape({n});
        r.set("text",StringArray::from_packed({n},std::move(bytes),std::move(offsets)));
        auto data=AnnData::empty(0,0); data.set_uns("budget",Element(std::move(r)));
        const auto path=root/filename; write_h5ad(path.string(),data);
        successful_record_reads=failed_record_reads=largest_record_prefix=peak_record_staging=0;
        observe_record_reads=true;
        auto loaded=read_h5ad(path.string()); observe_record_reads=false;
        auto v=loaded.uns().at("budget").record_array();
        require(v.size()==n && v.string("text",n-1).size()==width && v.string("text",n-1)[0]==char('A'+(n-1)%26),"bounded string round trip differs");
    };
    // 4096-count admission would permit a multi-megabyte batch. The actual
    // capped allocator forces a shorter prefix before packing final output.
    roundtrip(64,65536,"bounded-prefix.h5ad");
    require(failed_record_reads>0 && successful_record_reads>1 && largest_record_prefix<64,"record prefix did not shrink after byte admission failure");
    require(peak_record_staging>0 && peak_record_staging<=record_target,"normal staging exceeded 1 MiB target");
    std::cout<<"record_staging batch reads="<<successful_record_reads<<" retries="<<failed_record_reads<<" max_prefix="<<largest_record_prefix<<" peak="<<peak_record_staging<<'\n';
    // Both eager directions admit one intrinsically larger record explicitly.
    roundtrip(2,2*record_target,"large-record.h5ad");
    require(failed_record_reads==2 && successful_record_reads==2 && largest_record_prefix==1 && peak_record_staging>record_target && peak_record_staging<=record_cap,"large records repeated a learned staging failure");
    std::cout<<"record_staging single reads="<<successful_record_reads<<" retries="<<failed_record_reads<<" peak="<<peak_record_staging<<'\n';
}
void padding_fixture(const std::filesystem::path& path) {
    auto eager = read_h5ad(path.string());
    auto backed = open_h5ad(path.string(),{.scratch_bytes=4096});
    const char* modes[]{"str_nullterm","str_nullpad","str_spacepad"};
    for (std::size_t mode = 0; mode < 3; ++mode) {
        const std::string expected[]{mode ? std::string("A\0B",3) : "A",
            mode==2 ? std::string("tail\0",5) : "tail",mode ? std::string("mid\0x",5) : "mid"};
        const auto name = std::string(modes[mode]), prefix = "padding_"+name;
        auto column = backed.obs().column(name).read();
        auto array = backed.uns().at(prefix+"_array").read();
        auto field = backed.uns().at("padding_record").read_field(name);
        auto record = backed.uns().at("padding_record").read();
        for (std::size_t i = 0; i < 3; ++i) {
            require(eager.obs().column(name).string(i)==expected[i] && column.view().string(i)==expected[i],"ordinary fixed padding differs");
            require(eager.uns().at(prefix+"_array").string_array().string(i)==expected[i] && array.view().string_array().string(i)==expected[i],"packed fixed padding differs");
            require(eager.uns().at("padding_record").record_array().string(name,i)==expected[i] &&
                field.view().string_array().string(i)==expected[i] && record.view().record_array().string(name,i)==expected[i],"record fixed padding differs");
        }
        auto scalar = backed.uns().at(prefix+"_scalar").read();
        auto empty = backed.uns().at(prefix+"_empty").read();
        require(eager.uns().at(prefix+"_scalar").string()==expected[0] && scalar.view().string()==expected[0],"scalar fixed padding differs");
        require(eager.uns().at(prefix+"_empty").string_array().size()==0 && empty.view().string_array().size()==0,"empty fixed padding differs");
    }
}
void run(const std::filesystem::path& root) {
    auto d=fixture(); auto input=root/"records.h5ad", output=root/"export.h5ad";
    WriteOptions options; options.compression=Compression::gzip; write_h5ad(input.string(),d,options);
    auto loaded=read_h5ad(input.string()); check(loaded.uns().at("mixed").record_array());
    require(loaded.uns().at("empty").record_array().shape()==std::vector<std::size_t>({0,2}),"empty shape lost");
    auto scalar=loaded.uns().at("scalar").record_array();
    require(scalar.rank()==0 && *scalar.member<std::uint64_t>("value").flat_ptr(0)==77,"scalar record lost");
    auto b=open_h5ad(input.string(),{.mode=AccessMode::read_write,.scratch_bytes=4096});
    auto record=b.uns().at("mixed"); require(record.kind()==ElementKind::record_array,"wrong backed record kind");
    require(record.field_names()==std::vector<std::string>({"id","p/value","name"}),"backed fields reordered");
    auto field=record.read_field("p/value");
    require(field.view().array().dense<double>().data()[2]==0.3,"independent numeric field differs");
    auto text=record.read_field("name"); require(text.view().string_array().string(0)=="α","independent text field differs");
    auto element=record.read(); check(element.view().record_array());
    fails([&]{(void)record.read_field("absent");},ErrorCode::out_of_bounds);
    auto selected=b.slice(indices{1},all); write_h5ad(output.string(),selected,options);
    auto subset=read_h5ad(output.string()); check(subset.uns().at("mixed").record_array());
    require(subset.n_obs()==1 && subset.x().layout()==Layout::csr && subset.has_raw(),"other slots changed");
    b.set("/uns/mixed",Element(mixed()));
    fails([&]{(void)record.read();},ErrorCode::invalid_argument);
    b.close(); check(element.view().record_array());
    require(field.view().array().dense<double>().data()[0]==0.1,"materialized field died on close");
    interleaved(root); scratch_limits(root);
}
}
int main(int argc,char** argv) {
    try {
#ifdef __APPLE__
        using SetObserver = void (*)(void (*)(hid_t,hid_t,hid_t,herr_t));
        const auto set = reinterpret_cast<SetObserver>(dlsym(RTLD_DEFAULT,"anndatapp_test_observe_h5dread"));
        if (!set) throw std::runtime_error("HDF5 read observer is required");
        set(&observe_record_read);
#endif
        const auto root=std::filesystem::path(argc>1?argv[1]:"records-h5ad-tests");
        std::filesystem::create_directories(root); run(root);
        if (argc > 2) padding_fixture(argv[2]);
        std::cout<<"record H5AD tests passed\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
