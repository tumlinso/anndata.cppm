#include "zarr_engine.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <set>

namespace anndata_zarr {
namespace {
Json load_json(const std::filesystem::path& file) {
    try {
        std::ifstream stream(file);
        if (!stream) throw Error(Failure::io, file.string(), "cannot open Zarr metadata");
        return Json::parse(stream);
    } catch (const Json::exception& e) { throw Error(Failure::encoding, file.string(), std::string("invalid Zarr JSON: ") + e.what()); }
}
void save_json(const std::filesystem::path& file, const Json& value) {
    std::filesystem::create_directories(file.parent_path());
    auto temporary = file; temporary += ".anndata-tmp";
    std::ofstream stream(temporary, std::ios::trunc);
    if (!stream || !(stream << value.dump(2)))
        throw Error(Failure::io, file.string(), "cannot write Zarr metadata");
    stream.close();
    if (stream.fail()) throw Error(Failure::io, file.string(), "cannot close Zarr metadata");
    std::filesystem::rename(temporary, file);
}
Bytes load_bytes(const std::filesystem::path& file, std::size_t budget) {
    const auto size = std::filesystem::file_size(file);
    if (size > budget) throw Error(Failure::argument, file.string(), "encoded chunk exceeds scratch_bytes; use smaller chunks or increase the explicit budget");
    Bytes result(static_cast<std::size_t>(size));
    std::ifstream stream(file, std::ios::binary);
    if (!stream || (size && !stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(size))))
        throw Error(Failure::io, file.string(), "cannot read Zarr chunk");
    return result;
}
void save_bytes(const std::filesystem::path& file, std::span<const std::byte> bytes) {
    std::filesystem::create_directories(file.parent_path());
    auto temporary=file; temporary += ".anndata-tmp";
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream || (bytes.size() && !stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))))
        throw Error(Failure::io, file.string(), "cannot write Zarr chunk");
    stream.close();
    if (stream.fail()) throw Error(Failure::io, file.string(), "cannot close Zarr chunk");
    std::filesystem::rename(temporary,file);
}
std::string relative_node(std::string path) {
    while (!path.empty() && path.front()=='/') path.erase(path.begin());
    while (!path.empty() && path.back()=='/') path.pop_back();
    if (path.find('\0')!=std::string::npos) throw Error(Failure::argument,path,"NUL in Zarr path");
    for (const auto& part:std::filesystem::path(path))
        if (part==".." || part==".") throw Error(Failure::argument,path,"invalid Zarr path component");
    return path;
}
Shape read_shape(const Json& j, const std::string& path, bool positive=false) {
    if (!j.is_array()) throw Error(Failure::encoding,path,"Zarr shape must be an integer array");
    Shape result;
    for (const auto& n:j) {
        if (!n.is_number_integer() || (n.is_number_integer() && n.get<std::int64_t>()<0))
            throw Error(Failure::encoding,path,"negative or non-integer Zarr extent");
        const auto value=n.get<std::uint64_t>();
        if (value>std::numeric_limits<std::size_t>::max() || (positive && value==0))
            throw Error(Failure::encoding,path,"invalid Zarr chunk extent");
        result.push_back(static_cast<std::size_t>(value));
    }
    checked_product(result);
    return result;
}
Type parse_type(std::string dtype, int format, char& endian, const std::string& path) {
    if (format==2) {
        if (dtype=="|O" || dtype=="O" || dtype=="|O8" || dtype=="object") return Type::string;
        if (dtype.empty()) throw Error(Failure::dtype,path,"invalid Zarr v2 dtype");
        if (dtype.front()=='<' || dtype.front()=='>' || dtype.front()=='|' || dtype.front()=='=') { endian=dtype.front(); dtype.erase(dtype.begin()); }
        if (dtype=="b1" || dtype=="bool" || dtype=="?") return Type::boolean;
        const std::vector<std::string> names={"i1","u1","i2","u2","i4","u4","i8","u8","f4","f8"};
        for(std::size_t i=0;i<names.size();++i) if(dtype==names[i]) return static_cast<Type>(i+1);
    } else {
        const std::vector<std::string> names={"bool","int8","uint8","int16","uint16","int32","uint32","int64","uint64","float32","float64","string"};
        for(std::size_t i=0;i<names.size();++i) if(dtype==names[i]) return static_cast<Type>(i);
        if(dtype=="str" || dtype=="utf8" || dtype=="vlen-utf8") return Type::string;
    }
    throw Error(Failure::dtype,path,"unsupported Zarr dtype: "+dtype);
}
std::string dtype_name(Type type,int format,char endian) {
    const std::vector<std::string> v2={"b1","i1","u1","i2","u2","i4","u4","i8","u8","f4","f8","O"};
    const std::vector<std::string> v3={"bool","int8","uint8","int16","uint16","int32","uint32","int64","uint64","float32","float64","string"};
    if(format==3) return v3.at(static_cast<std::size_t>(type));
    return std::string(1,type_size(type)==1 || type==Type::string?'|':endian)+v2.at(static_cast<std::size_t>(type));
}
char serializer_endian(const Json& codecs) {
    for(const auto& codec:codecs) {
        const auto name=codec.value("name",std::string());
        if(name=="bytes") {
            const auto value=codec.value("configuration",Json::object()).value("endian",std::string("little"));
            if(value!="little" && value!="big") throw Error(Failure::encoding,"bytes","invalid bytes codec endianness");
            return value=="big"?'>':'<';
        }
        if(name=="sharding_indexed") return serializer_endian(codec.at("configuration").at("codecs"));
    }
    return '=';
}
ArrayMetadata parse_array(const Json& j,const std::string& path) {
    try {
        ArrayMetadata m; m.original=j; m.format=j.at("zarr_format").get<int>();
        m.shape=read_shape(j.at("shape"),path);
        if(m.format==2) {
            if(!j.at("dtype").is_string()) throw Error(Failure::dtype,path,"structured Zarr dtypes are unsupported");
            m.type=parse_type(j.at("dtype").get<std::string>(),2,m.endian,path);
            m.chunks=read_shape(j.at("chunks"),path,true);
            const auto order=j.value("order",std::string("C"));
            if(order!="C" && order!="F") throw Error(Failure::encoding,path,"invalid Zarr order");
            m.order=order.front();
            m.separator=j.value("dimension_separator",std::string(".")); m.default_chunk_keys=false;
            if(j.contains("filters") && !j["filters"].is_null()) for(const auto& codec:j["filters"]) m.codecs.push_back(codec);
            if(j.contains("compressor") && !j["compressor"].is_null()) m.codecs.push_back(j["compressor"]);
        } else if(m.format==3) {
            if(j.value("node_type",std::string())!="array") throw Error(Failure::encoding,path,"expected Zarr array");
            auto dtype=j.at("data_type");
            if(dtype.is_object()) {
                if(dtype.contains("configuration") && !dtype["configuration"].empty()) throw Error(Failure::dtype,path,"parameterized Zarr data type unsupported");
                dtype=dtype.at("name");
            }
            m.type=parse_type(dtype.get<std::string>(),3,m.endian,path);
            auto grid=j.at("chunk_grid");
            if(grid.at("name")!="regular") throw Error(Failure::encoding,path,"unsupported Zarr chunk grid");
            m.chunks=read_shape(grid.at("configuration").at("chunk_shape"),path,true);
            m.codecs=j.at("codecs");
            auto keys=j.at("chunk_key_encoding"); auto name=keys.at("name").get<std::string>();
            if(name!="default" && name!="v2") throw Error(Failure::encoding,path,"unsupported Zarr chunk key encoding: "+name);
            m.default_chunk_keys=name=="default";
            m.separator=keys.value("configuration",Json::object()).value("separator",m.default_chunk_keys?std::string("/"):std::string("."));
            if(j.contains("storage_transformers") && !j["storage_transformers"].empty()) throw Error(Failure::encoding,path,"unsupported required Zarr storage transformer");
            m.endian=serializer_endian(m.codecs);
        } else throw Error(Failure::version,path,"unsupported Zarr format version");
        m.item_size=type_size(m.type);
        m.fill=j.value("fill_value",Json());
        if(m.chunks.size()!=m.shape.size()) throw Error(Failure::encoding,path,"Zarr chunk and array ranks differ");
        if(m.separator!="/" && m.separator!=".") throw Error(Failure::encoding,path,"unsupported Zarr chunk separator");
        return m;
    } catch(const Json::exception& e) { throw Error(Failure::encoding,path,std::string("malformed Zarr array metadata: ")+e.what()); }
}
std::string chunk_key(const ArrayMetadata& m,const Shape& coord) {
    std::string result=m.default_chunk_keys?"c":"";
    if(coord.empty()) return m.default_chunk_keys?"c":"0";
    for(std::size_t i=0;i<coord.size();++i) {
        if(i || m.default_chunk_keys) result+=m.separator;
        result+=std::to_string(coord[i]);
    }
    return result;
}
bool sharding(const ArrayMetadata& m) { return m.format==3 && !m.codecs.empty() && m.codecs[0].value("name",std::string())=="sharding_indexed"; }
bool raw_native(const ArrayMetadata& m) {
    if(m.type==Type::string || m.order!='C' || !m.codecs.is_array()) return false;
    if(m.item_size>1 && m.endian!='=' && m.endian!='|' &&
       m.endian!=(std::endian::native==std::endian::little?'<':'>')) return false;
    if(m.format==2) return m.codecs.empty();
    if(m.format!=3 || m.codecs.size()!=1 || !m.codecs[0].is_object() || m.codecs[0].value("name",std::string())!="bytes") return false;
    const auto& codec=m.codecs[0];
    if(codec.contains("configuration") && !codec["configuration"].is_object()) return false;
    const auto config=codec.value("configuration",Json::object());
    if(m.item_size>1 && !config.contains("endian")) return false;
    return !config.contains("endian") || (config["endian"].is_string() &&
        (config["endian"]=="little" || config["endian"]=="big"));
}
ShardSpec shard_spec(const ArrayMetadata& m,std::size_t budget) {
    if(m.codecs.size()!=1) throw Error(Failure::codec,"","codecs after sharding_indexed are unsupported for partial access");
    const auto& config=m.codecs[0].at("configuration");
    ShardSpec spec{m.chunks,read_shape(config.at("chunk_shape"),"sharding",true),config.at("codecs"),config.at("index_codecs"),config.value("index_location",std::string("end")),budget};
    if(spec.chunk_shape.size()!=spec.shard_shape.size()) throw Error(Failure::encoding,"sharding","shard rank mismatch");
    for(std::size_t i=0;i<spec.chunk_shape.size();++i) if(spec.shard_shape[i]%spec.chunk_shape[i]) throw Error(Failure::encoding,"sharding","shard shape must be divisible by inner chunk shape");
    return spec;
}
Shape inner_shape(const ArrayMetadata& m,std::size_t budget) { return sharding(m)?shard_spec(m,budget).chunk_shape:m.chunks; }
template<class T> void fill_values(Bytes& bytes,const Json& value) {
    T scalar{};
    if(!value.is_null()) {
        if(value.is_string()) {
            const auto text=value.get<std::string>();
            if constexpr(std::is_floating_point_v<T>) {
                if(text=="NaN") scalar=std::numeric_limits<T>::quiet_NaN();
                else if(text=="Infinity") scalar=std::numeric_limits<T>::infinity();
                else if(text=="-Infinity") scalar=-std::numeric_limits<T>::infinity();
                else throw Error(Failure::encoding,"fill_value","invalid numeric fill value");
            } else throw Error(Failure::encoding,"fill_value","string fill value for integer array");
        } else scalar=value.get<T>();
    }
    for(std::size_t i=0;i<bytes.size();i+=sizeof(T)) std::memcpy(bytes.data()+i,&scalar,sizeof(T));
}
void fill_numeric(Bytes& bytes,const ArrayMetadata& m) {
    switch(m.type) {
        case Type::boolean: std::fill(bytes.begin(),bytes.end(),std::byte(!m.fill.is_null() && (m.fill.is_boolean()?m.fill.get<bool>():m.fill.get<double>()!=0.0))); break;
        case Type::int8: fill_values<std::int8_t>(bytes,m.fill); break;
        case Type::uint8: fill_values<std::uint8_t>(bytes,m.fill); break;
        case Type::int16: fill_values<std::int16_t>(bytes,m.fill); break;
        case Type::uint16: fill_values<std::uint16_t>(bytes,m.fill); break;
        case Type::int32: fill_values<std::int32_t>(bytes,m.fill); break;
        case Type::uint32: fill_values<std::uint32_t>(bytes,m.fill); break;
        case Type::int64: fill_values<std::int64_t>(bytes,m.fill); break;
        case Type::uint64: fill_values<std::uint64_t>(bytes,m.fill); break;
        case Type::float32: fill_values<float>(bytes,m.fill); break;
        case Type::float64: fill_values<double>(bytes,m.fill); break;
        case Type::string: break;
    }
}
Chunk filled(const ArrayMetadata& m,const Shape& shape,std::size_t budget=64*1024*1024) {
    Chunk result; auto count=checked_product(shape);
    if(m.type==Type::string) {
        const auto value=m.fill.is_string()?m.fill.get<std::string>():"";
        if(value.size()>budget/4 || count>(budget/4)/(sizeof(std::string)+value.size())) throw Error(Failure::argument,"fill_value","string fill chunk exceeds scratch memory budget");
        result.strings.resize(count,value); return result;
    }
    if(count>std::numeric_limits<std::size_t>::max()/m.item_size) throw Error(Failure::encoding,"chunk","chunk byte size overflow");
    result.numeric.resize(count*m.item_size); fill_numeric(result.numeric,m);
    return result;
}
std::size_t flat(const Shape& coord,const Shape& shape) {
    std::size_t value=0; for(std::size_t i=0;i<shape.size();++i) value=value*shape[i]+coord[i]; return value;
}
void coordinates(const Shape& first,const Shape& stop,const std::function<void(const Shape&)>& visit) {
    for(std::size_t i=0;i<first.size();++i) if(first[i]>=stop[i]) return;
    Shape current=first;
    if(current.empty()) { visit(current); return; }
    for(;;) {
        visit(current);
        std::size_t axis=current.size();
        while(axis) { --axis; if(++current[axis]<stop[axis]) break; current[axis]=first[axis]; }
        if(axis==0 && current[0]==first[0]) break;
    }
}
} // namespace

Store::Store(std::filesystem::path root,bool writable,std::size_t scratch_bytes):root_(std::move(root)),writable_(writable),scratch_bytes_(scratch_bytes) {
    if(scratch_bytes_<1024) throw Error(Failure::argument,root_.string(),"scratch_bytes must be at least 1024");
    consolidation_v2_=std::filesystem::exists(root_/".zmetadata");
    if(std::filesystem::exists(root_/"zarr.json")) consolidation_v3_=load_json(root_/"zarr.json").contains("consolidated_metadata");
}
std::filesystem::path Store::node_path(const std::string& path) const { return root_/relative_node(path); }
Node Store::inspect(const std::string& path) const {
    try {
    const auto relative=relative_node(path); const auto directory=node_path(path);
    Json metadata;
    if(std::filesystem::exists(directory/"zarr.json")) metadata=load_json(directory/"zarr.json");
    else if(std::filesystem::exists(directory/".zarray")) metadata=load_json(directory/".zarray");
    else if(std::filesystem::exists(directory/".zgroup")) metadata=load_json(directory/".zgroup");
    else if(std::filesystem::exists(root_/".zmetadata")) {
        const auto consolidated=load_json(root_/".zmetadata"); const auto prefix=relative.empty()?"":relative+"/";
        const auto& all=consolidated.at("metadata");
        if(all.contains(prefix+".zarray")) metadata=all.at(prefix+".zarray");
        else if(all.contains(prefix+".zgroup")) metadata=all.at(prefix+".zgroup");
    } else if(std::filesystem::exists(root_/"zarr.json")) {
        const auto root=load_json(root_/"zarr.json");
        if(root.contains("consolidated_metadata") && root["consolidated_metadata"].contains("metadata") && root["consolidated_metadata"]["metadata"].contains(relative)) metadata=root["consolidated_metadata"]["metadata"][relative];
    }
    if(metadata.is_null()) return {};
    Node result; result.exists=true;
    auto format=metadata.value("zarr_format",0);
    if(format!=2 && format!=3) throw Error(Failure::version,path,"unsupported Zarr format version");
    result.group=format==2?!metadata.contains("shape"):metadata.value("node_type",std::string())=="group";
    result.attributes=metadata.value("attributes",Json::object());
    if(format==2) {
        if(std::filesystem::exists(directory/".zattrs")) result.attributes=load_json(directory/".zattrs");
        else if(std::filesystem::exists(root_/".zmetadata")) {
            auto all=load_json(root_/".zmetadata").at("metadata"); auto key=(relative.empty()?"":relative+"/")+".zattrs";
            if(all.contains(key)) result.attributes=all.at(key);
        }
    }
    if(!result.group) result.array=parse_array(metadata,path);
    return result;
    } catch(const Json::exception& e) { throw Error(Failure::encoding,path,std::string("malformed Zarr node metadata: ")+e.what()); }
}
std::vector<std::string> Store::children(const std::string& path) const {
    std::set<std::string> names; const auto directory=node_path(path);
    if(std::filesystem::exists(directory)) for(const auto& entry:std::filesystem::directory_iterator(directory))
        if(entry.is_directory() && inspect(relative_node(path)+"/"+entry.path().filename().string()).exists) names.insert(entry.path().filename().string());
    const auto prefix=relative_node(path).empty()?"":relative_node(path)+"/";
    const auto add=[&](const std::string& key) { if(key.starts_with(prefix)) { const auto rest=key.substr(prefix.size()); const auto slash=rest.find('/'); if(slash!=std::string::npos && slash) names.insert(rest.substr(0,slash)); } };
    if(std::filesystem::exists(root_/".zmetadata")) {
        const auto consolidated=load_json(root_/".zmetadata");
        for(const auto& [key,value]:consolidated.at("metadata").items()) { (void)value; add(key); }
    }
    if(std::filesystem::exists(root_/"zarr.json")) { auto root=load_json(root_/"zarr.json"); if(root.contains("consolidated_metadata") && root["consolidated_metadata"].contains("metadata")) for(const auto& [key,value]:root["consolidated_metadata"]["metadata"].items()) { (void)value; add(key+"/"); } }
    return {names.begin(),names.end()};
}
void Store::mutation() {
    if(!writable_) throw Error(Failure::argument,root_.string(),"Zarr store is read-only");
    if(std::filesystem::exists(root_/".zmetadata")) { consolidation_v2_=true; std::filesystem::remove(root_/".zmetadata"); }
    if(std::filesystem::exists(root_/"zarr.json")) { auto root=load_json(root_/"zarr.json"); if(root.contains("consolidated_metadata")) { consolidation_v3_=true; root.erase("consolidated_metadata"); save_json(root_/"zarr.json",root); } }
}
Chunk Store::chunk(const std::string& path,const ArrayMetadata& m,const Shape& coord) const {
    const auto shape=inner_shape(m,scratch_bytes_); const auto count=checked_product(shape);
    const auto working_width=m.type==Type::string?sizeof(std::string):m.item_size;
    if(count>scratch_bytes_/(4*working_width)) throw Error(Failure::argument,path,"decoded chunk exceeds bounded scratch_bytes; increase budget or rechunk the source");
    Bytes payload; std::filesystem::path file;
    const auto decode=[&](const Json& codecs) {
        if(payload.empty()) return filled(m,shape,scratch_bytes_);
        try { return decode_payload({m,codecs,scratch_bytes_},shape,payload); }
        catch(const Error& e) { throw Error(e.failure,path+"/"+file.filename().string(),e.what()); }
    };
    if(sharding(m)) {
        auto spec=shard_spec(m,scratch_bytes_/2); Shape shard(coord.size()),local(coord.size());
        for(std::size_t i=0;i<coord.size();++i) { auto factor=m.chunks[i]/shape[i]; shard[i]=coord[i]/factor; local[i]=coord[i]%factor; }
        file=node_path(path)/chunk_key(m,shard); if(!std::filesystem::exists(file)) return filled(m,shape,scratch_bytes_);
        payload=read_shard_chunk(file,spec,local);
        return decode(spec.codecs);
    } else { file=node_path(path)/chunk_key(m,coord); if(!std::filesystem::exists(file)) return filled(m,shape,scratch_bytes_); payload=load_bytes(file,scratch_bytes_/2); if(payload.empty()) throw Error(Failure::encoding,file.string(),"empty encoded Zarr chunk"); }
    return decode(m.codecs);
}
void Store::chunk(const std::string& path,const ArrayMetadata& m,const Shape& coord,const Chunk& value) {
    auto shape=inner_shape(m,scratch_bytes_);
    if(sharding(m)) {
        auto spec=shard_spec(m,scratch_bytes_/2); Shape shard(coord.size()),local(coord.size());
        for(std::size_t i=0;i<coord.size();++i) { auto factor=m.chunks[i]/shape[i]; shard[i]=coord[i]/factor; local[i]=coord[i]%factor; }
        const auto payload=encode_payload({m,spec.codecs,scratch_bytes_},shape,value); write_shard_chunk(node_path(path)/chunk_key(m,shard),spec,local,payload);
    } else { const auto payload=encode_payload({m,m.codecs,scratch_bytes_},shape,value); save_bytes(node_path(path)/chunk_key(m,coord),payload); }
}
void Store::transfer(const std::string& path,const Shape& start,const Shape& count,void* numeric,std::vector<std::string>* strings,bool writing) const {
    const auto node=inspect(path); if(!node.exists || node.group) throw Error(Failure::argument,path,"expected a Zarr array");
    transfer(path,node.array,start,count,numeric,strings,writing);
}
void Store::transfer(const std::string& path,const ArrayMetadata& m,const Shape& start,const Shape& count,void* numeric,std::vector<std::string>* strings,bool writing,const NumericCodec* codec) const {
    const auto rank=m.shape.size();
    if(start.size()!=rank || count.size()!=rank) throw Error(Failure::argument,path,"Zarr rectangle rank mismatch");
    for(std::size_t i=0;i<rank;++i) if(start[i]>m.shape[i] || count[i]>m.shape[i]-start[i]) throw Error(Failure::argument,path,"Zarr rectangle exceeds array bounds");
    if((strings!=nullptr)!=(m.type==Type::string)) throw Error(Failure::dtype,path,"numeric/string Zarr transfer mismatch");
    const auto output_count=checked_product(count);
    if(!output_count) return;
    if(strings && strings->size()!=output_count) throw Error(Failure::argument,path,"string transfer size mismatch");
    if(!strings && !numeric) throw Error(Failure::argument,path,"null Zarr transfer buffer");
    const auto shape=inner_shape(m,scratch_bytes_); Shape first(rank),stop(rank);
    const bool raw=codec && !strings && raw_native(m);
    const bool compressed=codec && !raw && !strings && codec->kind!=NumericCodec::Kind::none;
    // Keep only the workspaces used by this pipeline. A fallback codec uses
    // its original bounded allocations without retained capacity alongside it.
    if(!raw && !compressed) { Bytes().swap(numeric_scratch_); Bytes().swap(encoded_scratch_); Bytes().swap(codec_scratch_); }
    if(raw) { Bytes().swap(encoded_scratch_); Bytes().swap(codec_scratch_); }
    auto reserve=[&](Bytes& b,std::size_t bytes) {
        if(bytes>scratch_bytes_/4) throw Error(Failure::argument,path,"chunk allocation exceeds bounded scratch_bytes");
        if(bytes>b.capacity()) {
            Bytes().swap(b);
            Bytes fresh(bytes);
            if(fresh.capacity()>scratch_bytes_/4) throw Error(Failure::argument,path,"chunk allocation exceeds bounded scratch_bytes");
            b.swap(fresh); ++numeric_scratch_allocations_;
        } else b.resize(bytes);
    };
    if(raw || compressed) {
        const auto elements=checked_product(shape);
        if(!m.item_size || elements>scratch_bytes_/(4*m.item_size))
            throw Error(Failure::argument,path,"decoded chunk exceeds bounded scratch_bytes; increase budget or rechunk the source");
        const auto bytes=elements*m.item_size;
        if(bytes>std::size_t(std::numeric_limits<std::streamsize>::max()))
            throw Error(Failure::argument,path,"chunk exceeds stream size");
        reserve(numeric_scratch_,bytes);
        if(compressed && reusable_needs_block(*codec,false)) reserve(codec_scratch_,std::min<std::size_t>(scratch_bytes_/4,65536));
    }
    for(std::size_t i=0;i<rank;++i) { first[i]=start[i]/shape[i]; stop[i]=(start[i]+count[i]-1)/shape[i]+1; }
    coordinates(first,stop,[&](const Shape& coord) {
        Chunk value;
        if(raw || compressed) {
            const auto file=node_path(path)/chunk_key(m,coord);
            if(!std::filesystem::exists(file)) fill_numeric(numeric_scratch_,m);
            else {
                const auto length=std::filesystem::file_size(file);
                if(raw && length!=numeric_scratch_.size())
                    throw Error(Failure::encoding,file.string(),"decoded chunk has wrong byte size");
                if(compressed && (!length || length>scratch_bytes_/4))
                    throw Error(Failure::argument,file.string(),"encoded chunk exceeds bounded scratch_bytes or is empty");
                if(length>std::uintmax_t(std::numeric_limits<std::streamsize>::max()))
                    throw Error(Failure::argument,file.string(),"chunk exceeds stream size");
                if(compressed) reserve(encoded_scratch_,std::size_t(length));
                auto& bytes=raw?numeric_scratch_:encoded_scratch_;
                std::ifstream stream(file,std::ios::binary);
                if(!stream || (!bytes.empty() && !stream.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()))))
                    throw Error(Failure::io,file.string(),"cannot read Zarr chunk");
                if(compressed) decode_numeric_reusable(*codec,numeric_scratch_.size(),scratch_bytes_/4,encoded_scratch_,numeric_scratch_,codec_scratch_);
            }
        } else value=chunk(path,m,coord);
        Shape begin(rank),end(rank);
        for(std::size_t i=0;i<rank;++i) { begin[i]=std::max(start[i],coord[i]*shape[i]); end[i]=std::min(start[i]+count[i],coord[i]*shape[i]+shape[i]); }
        // Copy contiguous rows, avoiding a per-element numeric conversion loop.
        Shape row_end=end;
        if(rank) row_end.back()=begin.back()+1;
        coordinates(begin,row_end,[&](const Shape& absolute) {
            Shape local(rank),out(rank); for(std::size_t i=0;i<rank;++i) { local[i]=absolute[i]-coord[i]*shape[i]; out[i]=absolute[i]-start[i]; }
            auto source_index=flat(local,shape),target_index=flat(out,count);
            const auto run=rank?end.back()-begin.back():1;
            if(strings) {
                for(std::size_t j=0;j<run;++j) if(writing) value.strings[source_index+j]=(*strings)[target_index+j]; else (*strings)[target_index+j]=value.strings[source_index+j];
            } else {
                auto* output=static_cast<std::byte*>(numeric)+target_index*m.item_size; auto* input=((raw||compressed)?numeric_scratch_.data():value.numeric.data())+source_index*m.item_size;
                if(writing) std::memcpy(input,output,run*m.item_size); else std::memcpy(output,input,run*m.item_size);
            }
        });
        if(writing) {
            if(raw) save_bytes(node_path(path)/chunk_key(m,coord),numeric_scratch_);
            else if(compressed) {
                reserve(encoded_scratch_,reusable_encoded_bound(*codec,numeric_scratch_.size(),scratch_bytes_/4));
                encode_numeric_reusable(*codec,scratch_bytes_/4,numeric_scratch_,encoded_scratch_,codec_scratch_);
                save_bytes(node_path(path)/chunk_key(m,coord),encoded_scratch_);
            }
            else const_cast<Store*>(this)->chunk(path,m,coord,value);
        }
    });
}
void Store::read_numeric(const std::string& path,const Shape& start,const Shape& count,void* out) const { transfer(path,start,count,out,nullptr,false); }
ArrayMetadata Store::prepare_numeric(const std::string& path) {
    auto node=inspect(path);
    if(!node.exists || node.group || node.array.type==Type::string)
        throw Error(Failure::dtype,path,"prepared numerical transfer requires a numeric array");
    return std::move(node.array);
}
NumericCodec Store::numeric_codec(const ArrayMetadata& m) const {
    return sharding(m) ? NumericCodec{} : prepare_numeric_codec({m,m.codecs,scratch_bytes_},m.chunks);
}
void Store::read_numeric(const std::string& path,const ArrayMetadata& m,const NumericCodec& codec,const Shape& start,const Shape& count,void* out) const { transfer(path,m,start,count,out,nullptr,false,&codec); }
void Store::write_numeric(const std::string& path,const ArrayMetadata& m,const NumericCodec& codec,const Shape& start,const Shape& count,const void* input) {
    if(!writable_) throw Error(Failure::argument,path,"Zarr store is read-only");
    // Payload replacement changes no metadata, so consolidated metadata
    // remains valid and independent chunk writers need not modify it.
    transfer(path,m,start,count,const_cast<void*>(input),nullptr,true,&codec);
}
std::vector<std::string> Store::read_strings(const std::string& path,const Shape& start,const Shape& count) const { std::vector<std::string> result(checked_product(count)); transfer(path,start,count,nullptr,&result,false); return result; }
std::vector<std::string> Store::read_strings_bounded(const std::string& path,const Shape& start,
                                                   const Shape& count,std::size_t byte_budget) const {
    const auto node=inspect(path);
    if(!node.exists || node.group || node.array.type!=Type::string)
        throw Error(Failure::dtype,path,"bounded string reads require a Zarr string array");
    const auto& m=node.array;
    if(start.size()!=m.shape.size() || count.size()!=m.shape.size())
        throw Error(Failure::argument,path,"bounded string read rank mismatch");
    for(std::size_t axis=0;axis<m.shape.size();++axis) {
        if(start[axis]>m.shape[axis] || count[axis]>m.shape[axis]-start[axis])
            throw Error(Failure::argument,path,"bounded string read exceeds array bounds");
        if(axis+1<m.shape.size() && count[axis]!=1)
            throw Error(Failure::argument,path,"bounded string reads require one consecutive last-axis run");
    }
    if(!checked_product(count)) return {};
    const auto shape=inner_shape(m,scratch_bytes_);
    Shape coordinate(start.size()),local(start.size());
    for(std::size_t axis=0;axis<start.size();++axis) {
        coordinate[axis]=start[axis]/shape[axis];
        local[axis]=start[axis]%shape[axis];
    }
    auto decoded=chunk(path,m,coordinate);
    const auto offset=flat(local,shape);
    const auto available=count.empty()?std::size_t{1}:std::min(count.back(),shape.back()-local.back());
    std::size_t admitted=0,bytes=sizeof(std::vector<std::string>);
    for(;admitted<available;++admitted) {
        const auto text_bytes=decoded.strings[offset+admitted].capacity()+1;
        if(bytes>byte_budget || sizeof(std::string)>byte_budget-bytes || text_bytes>byte_budget-bytes-sizeof(std::string)) break;
        bytes+=sizeof(std::string)+text_bytes;
    }
    if(!admitted) throw Error(Failure::argument,path,"a single string exceeds the bounded read byte budget");
    // Return a maximal admitted prefix of this decoded chunk. Advancing to the
    // next chunk avoids decompressing it again for each requested string. The
    // final vector is allocated only after byte admission, including objects.
    std::vector<std::string> result;
    result.reserve(admitted);
    for(std::size_t i=0;i<admitted;++i) result.push_back(std::move(decoded.strings[offset+i]));
    return result;
}
void Store::write_numeric(const std::string& path,const Shape& start,const Shape& count,const void* input) { mutation(); transfer(path,start,count,const_cast<void*>(input),nullptr,true); }
void Store::write_strings(const std::string& path,const Shape& start,const Shape& count,const std::vector<std::string>& input) { mutation(); transfer(path,start,count,nullptr,const_cast<std::vector<std::string>*>(&input),true); }
void Store::create_group(const std::string& path,const Json& attributes,int format) {
    mutation(); auto directory=node_path(path);
    if(inspect(path).exists) throw Error(Failure::argument,path,"Zarr element already exists");
    if(format==2) { save_json(directory/".zgroup",{{"zarr_format",2}}); save_json(directory/".zattrs",attributes); if(relative_node(path).empty()) consolidation_v2_=true; }
    else if(format==3) { save_json(directory/"zarr.json",{{"zarr_format",3},{"node_type","group"},{"attributes",attributes}}); if(relative_node(path).empty()) consolidation_v3_=true; }
    else throw Error(Failure::argument,path,"Zarr output format must be 2 or 3");
}
void Store::create_array(const std::string& path,Type type,const Shape& shape,const Json& attributes,char endian,const CreateOptions& options) {
    mutation(); if(inspect(path).exists) throw Error(Failure::argument,path,"Zarr element already exists");
    if(options.format!=2 && options.format!=3) throw Error(Failure::argument,path,"Zarr output format must be 2 or 3");
    if(endian=='=') endian=std::endian::native==std::endian::little?'<':'>';
    Shape chunks=shape; for(auto& n:chunks) n=std::max<std::size_t>(n,1);
    const auto width=type==Type::string?16:type_size(type);
    const auto target=std::min(options.target_chunk_bytes,std::max<std::size_t>(1024,scratch_bytes_/8));
    while(checked_product(chunks)>std::max<std::size_t>(1,target/width)) { auto largest=std::max_element(chunks.begin(),chunks.end()); if(largest==chunks.end() || *largest<=1) break; *largest=(*largest+1)/2; }
    Json fill=type==Type::string?Json(""):type==Type::boolean?Json(false):Json(0); Json codecs=Json::array();
    const auto compression=options.compression;
    auto directory=node_path(path);
    if(options.format==2) {
        Json compressor=nullptr;
        if(compression=="zstd") compressor={{"id","zstd"},{"level",options.compression_level},{"checksum",false}};
        else if(compression=="gzip" || compression=="zlib") compressor={{"id",compression},{"level",options.compression_level}};
        else if(compression=="blosc") compressor={{"id","blosc"},{"cname","lz4"},{"clevel",options.compression_level},{"shuffle",1},{"blocksize",0}};
        else if(compression!="none") throw Error(Failure::codec,path,"unsupported output compressor");
        Json filters=type==Type::string?Json::array({Json{{"id","vlen-utf8"}}}):Json(nullptr);
        save_json(directory/".zarray",{{"zarr_format",2},{"shape",shape},{"chunks",chunks},{"dtype",dtype_name(type,2,endian)},{"compressor",compressor},{"filters",filters},{"fill_value",fill},{"order","C"},{"dimension_separator","."}});
        save_json(directory/".zattrs",attributes);
    } else {
        if(type==Type::string) codecs.push_back({{"name","vlen-utf8"},{"configuration",Json::object()}});
        else codecs.push_back({{"name","bytes"},{"configuration",{{"endian",endian=='>'?"big":"little"}}}});
        if(compression=="zstd") codecs.push_back({{"name","zstd"},{"configuration",{{"level",options.compression_level},{"checksum",false}}}});
        else if(compression=="gzip" || compression=="zlib") codecs.push_back({{"name","gzip"},{"configuration",{{"level",options.compression_level}}}});
        else if(compression=="blosc") codecs.push_back({{"name","blosc"},{"configuration",{{"cname","lz4"},{"clevel",options.compression_level},{"shuffle","shuffle"},{"typesize",width},{"blocksize",0}}}});
        else if(compression!="none") throw Error(Failure::codec,path,"unsupported output compressor");
        save_json(directory/"zarr.json",{{"zarr_format",3},{"node_type","array"},{"shape",shape},{"data_type",dtype_name(type,3,endian)},{"chunk_grid",{{"name","regular"},{"configuration",{{"chunk_shape",chunks}}}}},{"chunk_key_encoding",{{"name","default"},{"configuration",{{"separator","/"}}}}},{"fill_value",fill},{"codecs",codecs},{"attributes",attributes}});
    }
}
void Store::attributes(const std::string& path,const Json& attributes) {
    mutation(); const auto node=inspect(path); if(!node.exists) throw Error(Failure::argument,path,"Zarr element does not exist");
    auto merged=node.attributes; for(const auto& [key,value]:attributes.items()) merged[key]=value;
    const auto directory=node_path(path);
    if(std::filesystem::exists(directory/"zarr.json")) { auto j=load_json(directory/"zarr.json"); j["attributes"]=std::move(merged); save_json(directory/"zarr.json",j); }
    else save_json(directory/".zattrs",merged);
}
void Store::resize(const std::string& path,const Shape& shape) {
    mutation(); const auto node=inspect(path); if(!node.exists || node.group || shape.size()!=node.array.shape.size()) throw Error(Failure::argument,path,"invalid Zarr resize");
    // Delete chunks outside the new grid. Boundary chunks retain padding; a
    // later growth must never expose old out-of-shape values.
    auto previous=node.array.shape; auto m=node.array;
    // Shrinking is uncommon structural work; clear removed cells in intersected
    // chunks before publishing the new shape, without materializing the array.
    auto inner=inner_shape(m,scratch_bytes_); Shape first(shape.size(),0),stop(shape.size());
    for(std::size_t i=0;i<shape.size();++i) stop[i]=(previous[i]+inner[i]-1)/inner[i];
    coordinates(first,stop,[&](const Shape& coord) {
        bool touches=false;
        for(std::size_t i=0;i<shape.size();++i) touches=touches || coord[i]*inner[i]+inner[i]>shape[i];
        if(!touches) return;
        auto value=chunk(path,m,coord); auto empty=filled(m,inner,scratch_bytes_);
        Shape zero(shape.size(),0);
        coordinates(zero,inner,[&](const Shape& local) {
            bool removed=false;
            for(std::size_t i=0;i<shape.size();++i) removed=removed || coord[i]*inner[i]+local[i]>=shape[i];
            if(!removed) return;
            auto index=flat(local,inner);
            if(m.type==Type::string) value.strings[index]=empty.strings[index];
            else std::memcpy(value.numeric.data()+index*m.item_size,empty.numeric.data()+index*m.item_size,m.item_size);
        });
        chunk(path,m,coord,value);
    });
    auto j=m.original; j["shape"]=shape; save_json(node_path(path)/(m.format==2?".zarray":"zarr.json"),j);
}
void Store::erase(const std::string& path) { mutation(); if(relative_node(path).empty()) throw Error(Failure::argument,path,"cannot erase the Zarr store root"); std::filesystem::remove_all(node_path(path)); }
void Store::rename(const std::string& source,const std::string& target) {
    mutation();
    if(relative_node(source).empty() || relative_node(target).empty()) throw Error(Failure::argument,source,"cannot rename the Zarr store root");
    if(!inspect(source).exists || inspect(target).exists) throw Error(Failure::argument,target,"Zarr rename requires an existing source and absent destination");
    std::filesystem::create_directories(node_path(target).parent_path());
    std::filesystem::rename(node_path(source),node_path(target));
}
void Store::flush() {
    if(!writable_) return;
    if(consolidation_v2_) {
        Json all=Json::object();
        for(const auto& entry:std::filesystem::recursive_directory_iterator(root_)) if(entry.is_regular_file()) { auto name=entry.path().filename().string(); if(name==".zarray" || name==".zgroup" || name==".zattrs") all[std::filesystem::relative(entry.path(),root_).generic_string()]=load_json(entry.path()); }
        save_json(root_/".zmetadata",{{"zarr_consolidated_format",1},{"metadata",all}});
    }
    if(consolidation_v3_) {
        auto root=load_json(root_/"zarr.json"); Json all=Json::object();
        for(const auto& entry:std::filesystem::recursive_directory_iterator(root_)) if(entry.is_regular_file() && entry.path().filename()=="zarr.json" && entry.path()!=root_/"zarr.json") { auto metadata=load_json(entry.path()); metadata.erase("consolidated_metadata"); all[std::filesystem::relative(entry.path().parent_path(),root_).generic_string()]=metadata; }
        root["consolidated_metadata"]={{"kind","inline"},{"must_understand",false},{"metadata",all}}; save_json(root_/"zarr.json",root);
    }
}
std::filesystem::path Store::mapping_file(const std::string& path) const {
    const auto node=inspect(path); if(!node.exists || node.group || node.array.type==Type::string) return {};
    const auto& m=node.array;
    if(m.order!='C' || sharding(m)) return {};
    for(std::size_t i=0;i<m.shape.size();++i) if(m.chunks[i]!=m.shape[i]) return {};
    if(m.format==2 && !m.codecs.empty()) return {};
    if(m.format==3 && (m.codecs.size()!=1 || m.codecs[0].value("name",std::string())!="bytes")) return {};
    if(m.item_size>1 && ((m.endian=='>' && std::endian::native!=std::endian::big) || (m.endian=='<' && std::endian::native!=std::endian::little))) return {};
    auto file=node_path(path)/chunk_key(m,Shape(m.shape.size(),0));
    if(!std::filesystem::exists(file) || std::filesystem::file_size(file)!=checked_product(m.shape)*m.item_size) return {};
    return file;
}
} // namespace anndata_zarr
