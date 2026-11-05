module;
#include "zarr_engine.hpp"
#include <bit>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <variant>
#include <vector>

module anndatapp.backed;
import :storage;

namespace anndatapp {
namespace {
namespace ze = ::anndatapp_zarr;
[[noreturn]] void translate(const ze::Error& e) {
    auto code=ErrorCode::io_error;
    switch(e.failure) {
        case ze::Failure::io: break;
        case ze::Failure::argument: code=ErrorCode::invalid_argument; break;
        case ze::Failure::encoding: code=ErrorCode::unsupported_encoding; break;
        case ze::Failure::version: code=ErrorCode::unsupported_version; break;
        case ze::Failure::dtype: code=ErrorCode::unsupported_dtype; break;
        case ze::Failure::codec: code=ErrorCode::unavailable_filter; break;
    }
    throw Error(code,e.path,e.what());
}
template<class F> decltype(auto) zarr_call(F&& f) {
    try { return std::forward<F>(f)(); }
    catch(const ze::Error& e) { translate(e); }
    catch(const std::filesystem::filesystem_error& e) { throw Error(ErrorCode::io_error,e.path1().string(),e.what()); }
    catch(const ze::Json::exception& e) { throw Error(ErrorCode::unsupported_encoding,"zarr",e.what()); }
}
ze::Json to_json(const backed_detail::Attributes& attrs) {
    ze::Json result=ze::Json::object();
    for(const auto& [name,value]:attrs) std::visit([&](const auto& v) { result[name]=v; },value);
    return result;
}
bool semantic_attribute(std::string_view name) {
    return name=="encoding-type" || name=="encoding-version" || name=="_index" ||
        name=="column-order" || name=="shape" || name=="ordered" || name=="na-value";
}
backed_detail::Attributes from_json(const ze::Json& attrs) {
    backed_detail::Attributes result;
    if(!attrs.is_object()) throw ze::Error(ze::Failure::encoding,"attributes","Zarr attributes must be an object");
    for(const auto& [name,value]:attrs.items()) {
        if(!semantic_attribute(name)) continue;
        if(value.is_string()) result[name]=value.get<std::string>();
        else if(value.is_boolean()) result[name]=value.get<bool>();
        else if(value.is_number_integer()) result[name]=value.get<std::int64_t>();
        else if(value.is_number_float()) result[name]=value.get<double>();
        else if(value.is_array()) {
            bool strings=true,integers=true;
            for(const auto& item:value) { strings=strings&&item.is_string(); integers=integers&&item.is_number_integer(); }
            if(integers && (!value.empty() || name=="shape")) result[name]=value.get<std::vector<std::int64_t>>();
            else if(strings) result[name]=value.get<std::vector<std::string>>();
        }
        // Opaque extension attributes remain in the engine metadata. The
        // generic AnnData codec only needs its small typed attribute vocabulary.
    }
    return result;
}
char byte_order(ByteOrder order) { return order==ByteOrder::little?'<':order==ByteOrder::big?'>':'='; }
struct Mapping {
    void* data=MAP_FAILED;
    std::size_t size=0;
    ~Mapping() { if(data!=MAP_FAILED) ::munmap(data,size); }
};
class ZarrNumeric final:public backed_detail::NumericTransfer {
    ze::Store& store_;
    std::string path_;
    ze::ArrayMetadata metadata_;
    ze::NumericCodec codec_;
    bool writable_;
    backed_detail::NumericTransferStats stats_;
    std::size_t bytes(const std::vector<std::size_t>& count) const {
        auto n=ze::checked_product(count);
        if(metadata_.item_size && n>std::numeric_limits<std::size_t>::max()/metadata_.item_size)
            throw Error(ErrorCode::invalid_argument,path_,"transfer byte size overflows size_t");
        return n*metadata_.item_size;
    }
public:
    ZarrNumeric(ze::Store& store,std::string path,bool writing):store_(store),path_(std::move(path)),
        metadata_(zarr_call([&] { return store_.prepare_numeric(path_); })),
        codec_(zarr_call([&] { return store_.numeric_codec(metadata_); })),writable_(writing) {}
    void read(const std::vector<std::size_t>& start,const std::vector<std::size_t>& count,void* out) override {
        zarr_call([&] { store_.read_numeric(path_,metadata_,codec_,start,count,out); });
        const auto n=bytes(count); if(n) { ++stats_.read_calls; stats_.read_bytes+=n; }
    }
    void write(const std::vector<std::size_t>& start,const std::vector<std::size_t>& count,const void* in) override {
        if(!writable_) throw Error(ErrorCode::invalid_argument,path_,"transfer was not prepared for writing");
        zarr_call([&] { store_.write_numeric(path_,metadata_,codec_,start,count,in); });
        const auto n=bytes(count); if(n) { ++stats_.write_calls; stats_.write_bytes+=n; }
    }
    bool write_selection(const backed_detail::NumericWrite& input) override {
        if(!writable_) throw Error(ErrorCode::invalid_argument,path_,"transfer was not prepared for writing");
        ze::NumericWrite selected;selected.axes.reserve(input.axes.size());
        std::vector<std::size_t> count;count.reserve(input.axes.size());
        for(const auto& a:input.axes) {selected.axes.push_back({a.count,a.start,a.step,a.positions});count.push_back(a.count);}
        selected.data=input.data;selected.strides=input.strides;selected.context=input.context;selected.pack=input.pack;
        selected.scratch_bytes=input.scratch_bytes;selected.packing_metadata_bytes=input.packing_metadata_bytes;
        const bool done=zarr_call([&]{return store_.write_selection(path_,metadata_,selected);});
        const auto n=bytes(count);if(done && n){++stats_.write_calls;stats_.write_bytes+=n;}
        return done;
    }
    backed_detail::NumericTransferStats stats() const override { return stats_; }
    std::size_t scratch_capacity() const override { return 0; }
};
class ZarrBackend final:public backed_detail::Backend {
    std::unique_ptr<ze::Store> store_;
    std::string location_;
    std::string identity_;
    std::size_t scratch_bytes_;
    bool writable_;
    ze::CreateOptions create_;
    ze::Store& store() { if(!store_) throw Error(ErrorCode::invalid_argument,location_,"Zarr session is closed"); return *store_; }
public:
    ZarrBackend(std::string path,bool writable,std::size_t scratch,ze::CreateOptions create={}):location_(std::move(path)),scratch_bytes_(scratch),writable_(writable),create_(std::move(create)) {
        struct stat directory{};
        if(::stat(location_.c_str(),&directory)!=0 || !S_ISDIR(directory.st_mode))
            throw Error(ErrorCode::io_error,location_,"cannot identify the Zarr store directory");
        identity_="zarr:"+std::to_string(static_cast<std::uintmax_t>(directory.st_dev))+":"+
            std::to_string(static_cast<std::uintmax_t>(directory.st_ino));
        store_=std::make_unique<ze::Store>(location_,writable,scratch);
        const auto root=store_->inspect(""); if(root.exists) {
            if(std::filesystem::exists(std::filesystem::path(location_)/".zgroup")) create_.format=2;
            else create_.format=3;
        }
    }
    std::string location() const override { return location_; }
    std::string identity() const override { return identity_; }
    std::size_t retained_scratch_bytes() const override { return store_ ? store_->retained_scratch_bytes() : 0; }
    std::size_t scratch_allocations() const override { return store_ ? store_->scratch_allocations() : 0; }
    std::unique_ptr<backed_detail::Backend> fork(std::size_t scratch) override {
        return zarr_call([&] { return std::make_unique<ZarrBackend>(location_,writable_,scratch,create_); });
    }
    std::unique_ptr<backed_detail::NumericTransfer> prepare_numeric(std::string_view path) override {
        return zarr_call([&] { return std::make_unique<ZarrNumeric>(store(),std::string(path),writable_); });
    }
    std::size_t numeric_workspace_bytes(const backed_detail::NodeInfo& info, bool writing) override {
        // The codec keeps at most three quarter-budget payload buffers live:
        // original encoded/decoded data, its pipeline input, and its output.
        // Rewriting a shard additionally retains its bounded index pipeline.
        if (writing && info.sharded_payload) return (scratch_bytes_ / 8) * 7;
        return (scratch_bytes_ / 4) * 3;
    }
    backed_detail::NodeInfo inspect(std::string_view path) override { return zarr_call([&] {
        const auto node=store().inspect(std::string(path)); backed_detail::NodeInfo info;
        info.exists=node.exists; info.group=node.group;
        info.parallel_reads=true; info.parallel_writes=true;
        if(!node.exists) return info;
        info.attrs=from_json(node.attributes);
        for(const auto& [name,value]:node.attributes.items()) { (void)value; if(!semantic_attribute(name)) info.has_opaque_attrs=true; }
        if(!node.group) {
            info.strings=node.array.type==ze::Type::string;
            if(!info.strings) info.dtype=static_cast<DType>(node.array.type);
            info.shape=node.array.shape;
            info.physical.shape=node.array.shape;
            info.physical.organization=StorageOrganization::chunked;
            info.physical.chunks=node.array.chunks;
            info.physical.inner_chunks=node.array.chunks;
            info.physical.write_unit=node.array.chunks;
            info.byte_order=node.array.endian=='<'?ByteOrder::little:node.array.endian=='>'?ByteOrder::big:ByteOrder::native;
            if(!info.strings) {
                info.codec_tile_shape=node.array.chunks;
                const auto& codecs=node.array.codecs;
                if(codecs.is_array() && !codecs.empty() && codecs.front().is_object() &&
                   codecs.front().value("name",std::string())=="sharding_indexed") {
                    info.sharded_payload=true;
                    info.physical.organization=StorageOrganization::sharded;
                    info.codec_tile_shape=codecs.front().at("configuration").at("chunk_shape").get<std::vector<std::size_t>>();
                    info.physical.inner_chunks=info.codec_tile_shape;
                    if(info.codec_tile_shape.size()!=node.array.chunks.size())
                        throw Error(ErrorCode::unsupported_encoding,std::string(path),"shard and inner chunk ranks differ");
                    for(std::size_t axis=0;axis<info.codec_tile_shape.size();++axis)
                        if(!info.codec_tile_shape[axis] || info.codec_tile_shape[axis]>node.array.chunks[axis] ||
                           node.array.chunks[axis]%info.codec_tile_shape[axis])
                            throw Error(ErrorCode::unsupported_encoding,std::string(path),"invalid inner chunk shape");
                }
            }
        }
        return info;
    }); }
    std::vector<std::string> children(std::string_view path) override { return zarr_call([&] { return store().children(std::string(path)); }); }
    void read_numeric(std::string_view path,const std::vector<std::size_t>& start,const std::vector<std::size_t>& count,void* output) override { zarr_call([&] { store().read_numeric(std::string(path),start,count,output); }); }
    void write_numeric(std::string_view path,const std::vector<std::size_t>& start,const std::vector<std::size_t>& count,const void* input) override { zarr_call([&] { store().write_numeric(std::string(path),start,count,input); }); }
    std::vector<std::string> read_strings(std::string_view path,const std::vector<std::size_t>& start,const std::vector<std::size_t>& count) override { return zarr_call([&] { return store().read_strings(std::string(path),start,count); }); }
    std::vector<std::string> read_strings_bounded(std::string_view path,const std::vector<std::size_t>& start,
        const std::vector<std::size_t>& count,std::size_t byte_budget) override {
        return zarr_call([&] { return store().read_strings_bounded(std::string(path),start,count,byte_budget); });
    }
    void write_strings(std::string_view path,const std::vector<std::size_t>& start,const std::vector<std::size_t>& count,const std::vector<std::string>& values) override { zarr_call([&] { store().write_strings(std::string(path),start,count,values); }); }
    void create_group(std::string_view path,const backed_detail::Attributes& attrs) override { zarr_call([&] { store().create_group(std::string(path),to_json(attrs),create_.format); }); }
    void create_numeric(std::string_view path,DType dtype,const std::vector<std::size_t>& shape,const backed_detail::Attributes& attrs,ByteOrder order) override { zarr_call([&] {
        auto actual=dtype;
        auto encoding=attrs.find("encoding-type");
        if(encoding!=attrs.end() && std::get_if<std::string>(&encoding->second) && std::get<std::string>(encoding->second)=="null") actual=DType::boolean;
        store().create_array(std::string(path),static_cast<ze::Type>(actual),shape,to_json(attrs),byte_order(order),create_);
    }); }
    void create_strings(std::string_view path,const std::vector<std::size_t>& shape,const backed_detail::Attributes& attrs) override { zarr_call([&] { store().create_array(std::string(path),ze::Type::string,shape,to_json(attrs),'|',create_); }); }
    void attributes(std::string_view path,const backed_detail::Attributes& attrs) override { zarr_call([&] { store().attributes(std::string(path),to_json(attrs)); }); }
    void copy_attributes(std::string_view source,std::string_view target) override { zarr_call([&] {
        const auto old=store().inspect(std::string(source)); const auto replacement=store().inspect(std::string(target));
        if(!old.exists || !replacement.exists) throw ze::Error(ze::Failure::argument,std::string(source),"attribute copy requires existing Zarr elements");
        auto additions=ze::Json::object();
        for(const auto& [key,value]:old.attributes.items()) if(!replacement.attributes.contains(key)) additions[key]=value;
        store().attributes(std::string(target),additions);
    }); }
    void erase(std::string_view path) override { zarr_call([&] { store().erase(std::string(path)); }); }
    void rename(std::string_view source,std::string_view target) override { zarr_call([&] { store().rename(std::string(source),std::string(target)); }); }
    void flush() override { zarr_call([&] { store().flush(); }); }
    void close() override { store_.reset(); }
    std::optional<MappedArray> map_numeric(std::string_view path) override { return zarr_call([&]() -> std::optional<MappedArray> {
        const auto node=store().inspect(std::string(path));
        // Canonical bool representation requires conversion and validation.
        if(!node.exists || node.group || node.array.type==ze::Type::boolean) return std::nullopt;
        auto file=store().mapping_file(std::string(path)); if(file.empty()) return std::nullopt;
        auto bytes=ze::checked_product(node.array.shape)*node.array.item_size; if(!bytes) return std::nullopt;
        const int fd=::open(file.c_str(),O_RDONLY|O_CLOEXEC);
        if(fd<0) throw Error(ErrorCode::io_error,file.string(),std::strerror(errno));
        struct stat info{};
        if(::fstat(fd,&info)!=0 || !S_ISREG(info.st_mode) || info.st_size<0 || static_cast<std::uint64_t>(info.st_size)!=bytes) { ::close(fd); return std::nullopt; }
        auto owner=std::make_shared<Mapping>(); owner->size=bytes;
        owner->data=::mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0); const auto saved=errno; ::close(fd);
        if(owner->data==MAP_FAILED) throw Error(ErrorCode::io_error,file.string(),std::strerror(saved));
        return MappedArray(owner,static_cast<const std::byte*>(owner->data),bytes,static_cast<DType>(node.array.type),node.array.shape);
    }); }
};
void prepare_destination(std::string_view path,bool exclusive) {
    const auto location=std::filesystem::path(path);
    if(location.empty()) throw Error(ErrorCode::invalid_argument,std::string(path),"empty Zarr destination");
    if(std::filesystem::exists(location)) {
        if(exclusive) throw Error(ErrorCode::io_error,std::string(path),"Zarr destination already exists");
        if(std::filesystem::is_symlink(location) || !std::filesystem::is_directory(location) ||
           (!std::filesystem::exists(location/".zgroup") && !std::filesystem::exists(location/"zarr.json")))
            throw Error(ErrorCode::invalid_argument,std::string(path),"refusing to replace a directory without a Zarr root marker");
        std::filesystem::remove_all(location);
    }
    std::filesystem::create_directories(location);
}
bool path_prefix(const std::filesystem::path& prefix,const std::filesystem::path& path) {
    auto current=path.begin();
    for(const auto& part:prefix) {
        if(current==path.end() || part!=*current) return false;
        ++current;
    }
    return true;
}
void reject_source_overlap(std::string_view destination,const std::string& source) {
    if(destination.empty()) throw Error(ErrorCode::invalid_argument,std::string(destination),"empty Zarr destination");
    const auto source_path=std::filesystem::weakly_canonical(std::filesystem::absolute(source));
    const auto destination_path=std::filesystem::weakly_canonical(std::filesystem::absolute(std::filesystem::path(destination)));
    const bool equivalent=std::filesystem::exists(destination_path) && std::filesystem::equivalent(source_path,destination_path);
    if(equivalent || path_prefix(source_path,destination_path) || path_prefix(destination_path,source_path))
        throw Error(ErrorCode::invalid_argument,std::string(destination),"Zarr export source and destination paths must not overlap");
}
ze::CreateOptions creation(int format,std::string_view compression,int level) {
    if(format!=2 && format!=3) throw Error(ErrorCode::invalid_argument,"zarr","Zarr output format must be 2 or 3");
    if(compression!="none" && compression!="gzip" && compression!="zstd" && compression!="blosc") throw Error(ErrorCode::invalid_argument,"zarr","unsupported Zarr output compression");
    if((compression=="gzip" || compression=="blosc") && (level<0 || level>9)) throw Error(ErrorCode::invalid_argument,"zarr","gzip and Blosc compression levels must be between 0 and 9");
    ze::CreateOptions options; options.format=format; options.compression=compression; options.compression_level=level; return options;
}
// Records are metadata, not a substitute Zarr group encoding. Check before
// preparing the destination; this walk reads no numerical field payload.
template<class V> void require_zarr_metadata(const V& value,const std::string& path) {
    const auto kind=value.kind();
    if(kind==ElementKind::record_array)
        throw Error(ErrorCode::unsupported_encoding,path,"Zarr rec-array codecs are not implemented");
    if(kind!=ElementKind::mapping) return;
    for(const auto& key:value.keys()) require_zarr_metadata(value.at(key),backed_detail::child(path,key));
}
} // namespace
namespace detail {
BackedAnnData open_zarr_backed(std::string_view path,BackedOptions options) {
    return zarr_call([&] {
        auto backend=std::make_unique<ZarrBackend>(std::string(path),options.mode==AccessMode::read_write,options.scratch_bytes);
        return backed_detail::make_store(std::move(backend),options);
    });
}
void write_zarr_eager(std::string_view path,const AnnDataView& data,int format,bool exclusive,std::size_t scratch_bytes,std::string_view compression,int level,Validation validation) {
    zarr_call([&] {
        auto options=creation(format,compression,level);
        if(scratch_bytes<1024) throw Error(ErrorCode::invalid_argument,std::string(path),"scratch_bytes must be at least 1024");
        require_zarr_metadata(data.uns(),"/uns");
        prepare_destination(path,exclusive); ZarrBackend backend(std::string(path),true,scratch_bytes,options);
        backend.create_group("/",{}); backed_detail::initialize_store(backend,data,scratch_bytes,validation); backend.flush();
    });
}
void write_zarr_backed(std::string_view path,const BackedAnnDataView& data,int format,bool exclusive,std::size_t scratch_bytes,std::string_view compression,int level,Validation validation) {
    zarr_call([&] {
        auto options=creation(format,compression,level);
        if(scratch_bytes<1024) throw Error(ErrorCode::invalid_argument,std::string(path),"scratch_bytes must be at least 1024");
        const auto& source=BackedAccess::get(data);
        if(!source) throw Error(ErrorCode::invalid_argument,std::string(path),"empty backed view");
        source->session->check_open();
        reject_source_overlap(path,source->session->backend->location());
        require_zarr_metadata(data.uns(),"/uns");
        prepare_destination(path,exclusive); ZarrBackend backend(std::string(path),true,scratch_bytes,options);
        backed_detail::export_store(backend,data,scratch_bytes,validation); backend.flush();
    });
}
} // namespace detail
} // namespace anndatapp
