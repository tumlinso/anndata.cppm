module;
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <type_traits>
#include <variant>
#include <vector>

module anndata.backed;
import :storage;

namespace anndata::backed_detail {
namespace {
using Node = std::shared_ptr<detail::BackedNode>;
[[noreturn]] void fail(ErrorCode code,std::string_view path,std::string message) { throw Error(code,std::string(path),std::move(message)); }
Attributes attrs(std::string type,std::string version) {return {{"encoding-type",std::move(type)},{"encoding-version",std::move(version)}};}
std::vector<std::size_t> shape_of(const Node& n) { std::vector<std::size_t> out;for(const auto& a:n->axes)out.push_back(a.size);return out; }
std::size_t count_of(const std::vector<std::size_t>& shape,std::string_view path) {
    if(std::find(shape.begin(),shape.end(),0)!=shape.end())return 0;
    std::size_t out=1;for(auto n:shape){if(n && out>std::numeric_limits<std::size_t>::max()/n)fail(ErrorCode::shape_mismatch,path,"shape exceeds addressable range");out*=n;}return out;
}
Node subnode(const Node& parent,std::string_view key,std::vector<Axis> axes={}) { auto result=node(parent->session,child(parent->path,key),std::move(axes)); result->prune_categories=parent->prune_categories; return result; }
void require_children(const Node& n,const std::vector<std::string>& expected) {
    auto actual=n->session->backend->children(n->path);auto wanted=expected;
    std::sort(actual.begin(),actual.end());std::sort(wanted.begin(),wanted.end());
    if(actual!=wanted)fail(ErrorCode::unsupported_encoding,n->path,"unexpected or missing stored children");
}
void validate_utf8(std::string_view value,std::string_view path){
    auto* bytes=reinterpret_cast<const unsigned char*>(value.data());std::size_t i=0;
    while(i<value.size()){
        auto first=bytes[i++];if(first<0x80)continue;
        unsigned remaining;std::uint32_t code,minimum;
        if(first>=0xc2&&first<=0xdf){remaining=1;code=first&0x1f;minimum=0x80;}
        else if(first>=0xe0&&first<=0xef){remaining=2;code=first&0x0f;minimum=0x800;}
        else if(first>=0xf0&&first<=0xf4){remaining=3;code=first&7;minimum=0x10000;}
        else fail(ErrorCode::invalid_argument,path,"invalid UTF-8 string");
        if(remaining>value.size()-i)fail(ErrorCode::invalid_argument,path,"truncated UTF-8 string");
        while(remaining--){auto next=bytes[i++];if((next&0xc0)!=0x80)fail(ErrorCode::invalid_argument,path,"invalid UTF-8 continuation");code=(code<<6)|(next&0x3f);}
        if(code<minimum||code>0x10ffff||(code>=0xd800&&code<=0xdfff))fail(ErrorCode::invalid_argument,path,"invalid Unicode scalar");
    }
}
void copy_strings(Backend& out,std::string_view path,const Node& n,const NodeInfo& info,std::size_t budget) {
    if(info.group || !info.strings)fail(ErrorCode::unsupported_encoding,n->path,"string encoding requires a string array");
    auto shape=shape_of(n);auto count=count_of(shape,n->path);
    out.create_strings(path,shape,info.attrs);
    // Backends admit a byte-budgeted consecutive prefix and reject a string
    // or codec workspace that cannot fit the configured staging budget.
    const auto cap=std::max<std::size_t>(1,budget/(2*sizeof(std::string)));
    std::vector<std::size_t> logical(shape.size()),start(shape.size()),length(shape.size(),1);
    for(std::size_t flat=0;flat<count;){
        auto rem=flat;
        for(std::size_t axis=shape.size();axis-->0;){logical[axis]=rem%shape[axis];rem/=shape[axis];start[axis]=n->axes[axis].at(logical[axis]);}
        std::size_t run=1;
        if(!shape.empty()){
            auto last=shape.size()-1;auto limit=std::min(cap,shape[last]-logical[last]);
            while(run<limit && n->axes[last].at(logical[last]+run)==start[last]+run)++run;
            length[last]=run;
        }
        auto values=n->session->backend->read_strings_bounded(n->path,start,length,budget);
        if(values.empty()||values.size()>run)fail(ErrorCode::io_error,n->path,"string transfer returned an incorrect length");
        run=values.size();if(!length.empty())length.back()=run;
        if(n->session->options.validation==Validation::full){
            // Validate only the loaded tile; no decoded cache survives transfer.
            for(const auto& value:values)validate_utf8(value,n->path);
        }
        out.write_strings(path,logical,length,values);flat+=run;
    }
}
template<class T> std::int64_t signed_value(const std::byte* p,std::string_view path) {
    T value;std::memcpy(&value,p,sizeof value);
    if constexpr(std::is_unsigned_v<T>)if(value>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))fail(ErrorCode::out_of_bounds,path,"category code exceeds signed range");
    return static_cast<std::int64_t>(value);
}
std::int64_t category_code(const std::byte* p,DType t,std::string_view path){
    switch(t){
    case DType::int8:return signed_value<std::int8_t>(p,path);case DType::uint8:return signed_value<std::uint8_t>(p,path);
    case DType::int16:return signed_value<std::int16_t>(p,path);case DType::uint16:return signed_value<std::uint16_t>(p,path);
    case DType::int32:return signed_value<std::int32_t>(p,path);case DType::uint32:return signed_value<std::uint32_t>(p,path);
    case DType::int64:return signed_value<std::int64_t>(p,path);case DType::uint64:return signed_value<std::uint64_t>(p,path);
    default:fail(ErrorCode::unsupported_dtype,path,"category codes must be integers");}
}
void validate_codes(const Node& codes,std::size_t dictionary,std::size_t budget){
    const auto info=codes->session->backend->inspect(codes->path);
    if(info.group||info.strings||info.shape.size()!=1||codes->axes.size()!=1)fail(ErrorCode::shape_mismatch,codes->path,"category codes must be one-dimensional numeric arrays");
    if(info.dtype==DType::boolean||info.dtype==DType::float32||info.dtype==DType::float64)fail(ErrorCode::unsupported_dtype,codes->path,"category codes require integer dtype");
    if(codes->session->options.validation==Validation::none)return;
    const auto width=dtype_size(info.dtype);const auto cap=std::max<std::size_t>(1,budget/width);
    std::vector<std::byte> data;
    for(std::size_t i=0;i<codes->axes[0].size;){
        auto first=codes->axes[0].at(i);std::size_t run=1;auto limit=std::min(cap,codes->axes[0].size-i);
        while(run<limit && codes->axes[0].at(i+run)==first+run)++run;
        data.resize(run*width);codes->session->backend->read_numeric(codes->path,{first},{run},data.data());
        for(std::size_t j=0;j<run;++j){auto value=category_code(data.data()+j*width,info.dtype,codes->path);if(value < -1 || (value>=0&&static_cast<std::uint64_t>(value)>=dictionary))fail(ErrorCode::out_of_bounds,codes->path,"category code lies outside dictionary");}
        i+=run;
    }
}
void check_table(const Node& n,const NodeInfo& info){
    if(!info.group||n->axes.size()!=1)fail(ErrorCode::shape_mismatch,n->path,"dataframe must have a one-dimensional row index");
    auto index=string_attr(info,"_index",n->path);auto columns=strings_attr(info,"column-order",n->path);
    std::set<std::string> unique;std::vector<std::string> expected{index};
    for(const auto& column:columns){if(!unique.insert(column).second)fail(ErrorCode::unsupported_encoding,n->path,"duplicate or index-conflicting dataframe column");if(column!=index)expected.push_back(column);}
    require_children(n,expected);
    auto index_info=n->session->backend->inspect(child(n->path,index));
    if(encoding(index_info,child(n->path,index))!="string-array"||index_info.shape.size()!=1||!index_info.strings)fail(ErrorCode::unsupported_encoding,n->path,"dataframe index must be a string array");
    auto total=index_info.shape[0];
    for(const auto& column:columns){auto shape=logical_shape(*n->session,child(n->path,column));if(shape.size()!=1||shape[0]!=total)fail(ErrorCode::shape_mismatch,child(n->path,column),"column length differs from stored dataframe index");}
}
void check_aligned_shape(const Node& n,std::size_t first,std::optional<std::size_t> second={},bool matrix=false){
    auto shape=shape_of(n);if(shape.empty()||shape[0]!=first||(second&&(shape.size()<2||shape[1]!=*second))||(matrix&&shape.size()!=2))fail(ErrorCode::shape_mismatch,n->path,"stored element does not match aligned axes");
}
void compare_table_index(const BackedTable& table,const BackedTable& axis){compare_table_indexes(table,axis);}
void check_dataset(const BackedAnnDataView& source){
    auto no=source.n_obs(),nv=source.n_vars();
    const auto& dataset=detail::BackedAccess::get(source);
    const std::set<std::string> allowed{"obs","var","X","layers","obsm","varm","obsp","varp","uns","raw"};
    for(const auto& key:dataset->session->backend->children("/"))
        if(!allowed.contains(key))fail(ErrorCode::unsupported_encoding,child("/",key),"unknown AnnData root element cannot be exported");
    if(source.has_raw())for(const auto& key:dataset->session->backend->children("/raw"))
        if(key!="X"&&key!="var"&&key!="varm")fail(ErrorCode::unsupported_encoding,child("/raw",key),"unknown raw element cannot be exported");
    if(source.has_x())check_aligned_shape(detail::BackedAccess::get(source.x()),no,nv,true);
    for(const auto& key:source.layer_names())check_aligned_shape(detail::BackedAccess::get(source.layer(key)),no,nv,true);
    for(const auto& key:source.obsp_names())check_aligned_shape(detail::BackedAccess::get(source.obsp(key)),no,no);
    for(const auto& key:source.varp_names())check_aligned_shape(detail::BackedAccess::get(source.varp(key)),nv,nv);
    for(const auto& key:source.obsm_names()){auto item=source.obsm(key);check_aligned_shape(detail::BackedAccess::get(item),no);if(item.kind()==ElementKind::table)compare_table_index(item.table(),source.obs());}
    for(const auto& key:source.varm_names()){auto item=source.varm(key);check_aligned_shape(detail::BackedAccess::get(item),nv);if(item.kind()==ElementKind::table)compare_table_index(item.table(),source.var());}
    if(source.has_raw()){
        auto nr=source.raw_var().size();if(source.has_raw_x())check_aligned_shape(detail::BackedAccess::get(source.raw_x()),no,nr,true);
        for(const auto& key:source.raw_varm_names()){auto item=source.raw_varm(key);check_aligned_shape(detail::BackedAccess::get(item),nr);if(item.kind()==ElementKind::table)compare_table_index(item.table(),source.raw_var());}
    }
}
void check_export_metadata(Session& session,std::string_view path,bool same_store,const Node& selected_uns){
    const auto info=session.backend->inspect(path);
    if(!info.exists)fail(ErrorCode::out_of_bounds,path,"stored element does not exist");
    if(info.has_opaque_attrs&&!same_store)fail(ErrorCode::unsupported_dtype,path,"opaque backend attributes cannot be represented in a different store");
    if(info.attrs.contains("encoding-type"))(void)encoding(info,path);
    if(!info.group)return;
    // A selected view may remove an invalid palette. Only surviving values
    // belong to the logical export; inspect all other source subtrees normally.
    const auto names=path=="/uns" && selected_uns && selected_uns->slice_history
        ? color_keys(selected_uns) : session.backend->children(path);
    for(const auto& key:names)check_export_metadata(session,child(path,key),same_store,selected_uns);
}
class BorrowedBackend final:public Backend {
public:
    explicit BorrowedBackend(std::shared_ptr<Session> retained):retained_(std::move(retained)){}
    std::string location()const override{return retained_->backend->location();}
    std::string identity()const override{return retained_->backend->identity();}
    std::size_t numeric_workspace_bytes(const NodeInfo& info,bool writing)override{
        retained_->check_open();return retained_->backend->numeric_workspace_bytes(info,writing);
    }
    NodeInfo inspect(std::string_view p)override{retained_->check_open();return retained_->backend->inspect(p);}
    std::vector<std::string>children(std::string_view p)override{retained_->check_open();return retained_->backend->children(p);}
    void read_numeric(std::string_view p,const std::vector<std::size_t>& s,const std::vector<std::size_t>& c,void* d)override{retained_->check_open();retained_->backend->read_numeric(p,s,c,d);}
    std::vector<std::string>read_strings(std::string_view p,const std::vector<std::size_t>& s,const std::vector<std::size_t>& c)override{retained_->check_open();return retained_->backend->read_strings(p,s,c);}
    std::vector<std::string>read_strings_bounded(std::string_view p,const std::vector<std::size_t>& s,const std::vector<std::size_t>& c,std::size_t b)override{retained_->check_open();return retained_->backend->read_strings_bounded(p,s,c,b);}
    void write_numeric(std::string_view,const std::vector<std::size_t>&,const std::vector<std::size_t>&,const void*)override{readonly();}
    void write_strings(std::string_view,const std::vector<std::size_t>&,const std::vector<std::size_t>&,const std::vector<std::string>&)override{readonly();}
    void create_group(std::string_view,const Attributes&)override{readonly();}
    void create_numeric(std::string_view,DType,const std::vector<std::size_t>&,const Attributes&,ByteOrder)override{readonly();}
    void create_strings(std::string_view,const std::vector<std::size_t>&,const Attributes&)override{readonly();}
    void attributes(std::string_view,const Attributes&)override{readonly();}
    void erase(std::string_view)override{readonly();}
    void rename(std::string_view,std::string_view)override{readonly();}
    void flush()override{}
    void close()override{}
private:
    [[noreturn]] static void readonly(){fail(ErrorCode::invalid_argument,"validation","operation-local source is read-only");}
    std::shared_ptr<Session>retained_;
};
BackedAnnDataView validation_view(const BackedAnnDataView& source,Validation validation){
    const auto& original=detail::BackedAccess::get(source);
    if(original->session->options.validation==validation)return source;
    auto session=std::make_shared<Session>();session->backend=std::make_unique<BorrowedBackend>(original->session);
    session->options=original->session->options;session->options.validation=validation;
    session->revisions=original->session->revisions;
    session->dataset_revision=original->session->dataset_revision;
    auto dataset=std::make_shared<detail::BackedDataset>(*original);dataset->session=std::move(session);
    return BackedAnnData(std::move(dataset)).view();
}
class Sink final:public Backend{
    std::string location_,identity_;
public:
    Sink(std::string location,std::string identity):location_(std::move(location)),identity_(std::move(identity)){}
    std::string location()const override{return location_;}
    std::string identity()const override{return identity_;}
    NodeInfo inspect(std::string_view)override{return {};}
    std::vector<std::string>children(std::string_view)override{return {};}
    void read_numeric(std::string_view,const std::vector<std::size_t>&,const std::vector<std::size_t>&,void*)override{}
    void write_numeric(std::string_view,const std::vector<std::size_t>&,const std::vector<std::size_t>&,const void*)override{}
    std::vector<std::string>read_strings(std::string_view,const std::vector<std::size_t>&,const std::vector<std::size_t>&)override{return {};}
    void write_strings(std::string_view,const std::vector<std::size_t>&,const std::vector<std::size_t>&,const std::vector<std::string>&)override{}
    void create_group(std::string_view,const Attributes&)override{}
    void create_numeric(std::string_view,DType,const std::vector<std::size_t>&,const Attributes&,ByteOrder)override{}
    void create_strings(std::string_view,const std::vector<std::size_t>&,const Attributes&)override{}
    void attributes(std::string_view,const Attributes&)override{}
    void erase(std::string_view)override{}
    void rename(std::string_view,std::string_view)override{}
    void flush()override{}
    void close()override{}
};
}
void validate_utf8_string(std::string_view value,std::string_view path){validate_utf8(value,path);}
void validate_category_dictionary(const Node& dictionary,std::size_t scratch_bytes){
    check_node(dictionary);
    auto& session=*dictionary->session;
    const auto info=session.backend->inspect(dictionary->path);
    const auto code=encoding(info,dictionary->path);
    if(dictionary->axes.size()!=1)fail(ErrorCode::shape_mismatch,dictionary->path,"category dictionary must be one-dimensional");
    Node values=dictionary;
    bool is_string=code=="string-array";
    if(code=="nullable-integer"||code=="nullable-boolean"||code=="nullable-string-array"){
        if(!info.group)fail(ErrorCode::unsupported_encoding,dictionary->path,"nullable dictionary must be a group");
        require_children(dictionary,{"values","mask"});
        values=subnode(dictionary,"values",dictionary->axes);
        auto mask=subnode(dictionary,"mask",dictionary->axes);
        const auto v=session.backend->inspect(values->path), m=session.backend->inspect(mask->path);
        if(encoding(m,mask->path)!="array"||m.group||m.strings||m.dtype!=DType::boolean||m.shape.size()!=1||m.shape!=v.shape)
            fail(ErrorCode::shape_mismatch,mask->path,"nullable category mask must match its one-dimensional boolean values");
        const auto values_code=encoding(v,values->path);
        is_string=code=="nullable-string-array";
        if(is_string){
            if(values_code!="string-array"||!v.strings||v.group)fail(ErrorCode::unsupported_encoding,values->path,"nullable string dictionary requires string-array values");
            auto sentinel=info.attrs.find("na-value");
            if(sentinel!=info.attrs.end()){
                const auto value=string_attr(info,"na-value",dictionary->path);
                if(value!="NA"&&value!="NaN")fail(ErrorCode::unsupported_encoding,dictionary->path,"unsupported nullable string sentinel");
            }
        }else{
            if(values_code!="array"||v.group||v.strings)fail(ErrorCode::unsupported_encoding,values->path,"nullable numeric dictionary requires numeric values");
            if(code=="nullable-boolean" ? v.dtype!=DType::boolean : v.dtype==DType::boolean||v.dtype==DType::float32||v.dtype==DType::float64)
                fail(ErrorCode::unsupported_dtype,values->path,"nullable dictionary values have an incompatible dtype");
        }
        if(session.options.validation==Validation::full){
            auto budget=std::min(scratch_bytes,scratch_remaining(dictionary));
            auto cap=std::max<std::size_t>(1,std::min<std::size_t>(8192,budget/4));
            std::vector<std::byte> data;
            const auto& rows=dictionary->axes.front();
            for(std::size_t i=0;i<rows.size;){
                const auto first=rows.at(i);std::size_t run=1;
                while(run<cap&&run<rows.size-i&&first<=std::numeric_limits<std::size_t>::max()-run&&rows.at(i+run)==first+run)++run;
                data.resize(run);session.backend->read_numeric(mask->path,{first},{run},data.data());
                for(auto missing:data)if(missing!=std::byte{0})fail(ErrorCode::invalid_argument,dictionary->path,"category dictionaries cannot contain nulls");
                i+=run;
            }
        }
    }else if(code!="array"&&code!="string-array"){
        fail(ErrorCode::unsupported_encoding,dictionary->path,"dictionary must be a numeric or string column");
    }
    const auto v=session.backend->inspect(values->path);
    if(v.group||v.shape.size()!=1||v.strings!=is_string)fail(ErrorCode::unsupported_encoding,values->path,"invalid category dictionary values");
    if(session.options.validation==Validation::full){
        auto budget=std::min(scratch_bytes,scratch_remaining(values));
        if(is_string)require_unique_index(values,values->axes.front(),budget);
        else require_unique_categories(values,budget);
    }
}
void copy_node_impl(Backend& destination,std::string_view path,const Node& source,std::size_t scratch_bytes,
                    std::optional<std::size_t> string_scratch_bytes){
    check_node(source);
    if(source->empty_mapping){destination.create_group(path,attrs("dict","0.1.0"));return;}
    auto& session=*source->session;auto info=session.backend->inspect(source->path);auto code=encoding(info,source->path);
    const bool same_store=&destination==session.backend.get() || destination.identity()==session.backend->identity();
    if(info.has_opaque_attrs && !same_store)
        fail(ErrorCode::unsupported_dtype,source->path,"opaque backend attributes cannot be represented in a different store");
    const auto budget=std::min(scratch_bytes,scratch_remaining(source));
    if(source->color_override){copy_promoted_color(destination,path,source,budget);return;}
    if(budget<64)fail(ErrorCode::invalid_argument,source->path,"scratch budget must be at least 64 bytes");
    if(code=="numeric-scalar" && !info.shape.empty())fail(ErrorCode::shape_mismatch,source->path,"numeric-scalar requires rank zero");
    if(code=="string" && !info.shape.empty())fail(ErrorCode::shape_mismatch,source->path,"string scalar requires rank zero");
    if(code=="array"||code=="numeric-scalar"||code=="csr_matrix"||code=="csc_matrix"){copy_array(destination,path,source,budget);return;}
    if(code=="string-array"||code=="string"){copy_strings(destination,path,source,info,std::min(budget,string_scratch_bytes.value_or(budget)));return;}
    if(code=="null"){if(info.group)require_children(source,{});destination.create_numeric(path,DType::float64,{},info.attrs,ByteOrder::native);return;}
    if(!info.group)fail(ErrorCode::unsupported_encoding,source->path,"group encoding is stored as a dataset");
    if(code=="dataframe"){
        check_table(source,info);destination.create_group(path,info.attrs);
        const auto index=string_attr(info,"_index",source->path);
        copy_node(destination,child(path,index),subnode(source,index,source->axes),budget,string_scratch_bytes);
        for(const auto& column:strings_attr(info,"column-order",source->path))if(column!=index)copy_node(destination,child(path,column),subnode(source,column,source->axes),budget,string_scratch_bytes);
        return;
    }
    if(code=="categorical"){
        require_children(source,{"codes","categories"});auto ordered=info.attrs.find("ordered");if(ordered==info.attrs.end()||!std::holds_alternative<bool>(ordered->second))fail(ErrorCode::unsupported_encoding,source->path,"categorical ordered must be a boolean");
        auto categories=subnode(source,"categories");auto category_info=session.backend->inspect(categories->path);auto category_encoding=encoding(category_info,categories->path);
        (void)category_encoding;
        if(categories->axes.size()!=1)fail(ErrorCode::shape_mismatch,categories->path,"dictionary must be one-dimensional");
        if(source->prune_categories){copy_pruned_category(destination,path,source,budget);return;}
        validate_category_dictionary(categories,budget);
        auto codes=subnode(source,"codes",source->axes);
        validate_codes(codes,categories->axes[0].size,budget);
        destination.create_group(path,info.attrs);copy_node(destination,child(path,"codes"),codes,budget,string_scratch_bytes);copy_node(destination,child(path,"categories"),categories,budget,string_scratch_bytes);return;
    }
    if(code=="nullable-integer"||code=="nullable-boolean"||code=="nullable-string-array"){
        require_children(source,{"values","mask"});auto values=subnode(source,"values",source->axes);auto mask=subnode(source,"mask",source->axes);
        auto mask_info=session.backend->inspect(mask->path);auto values_info=session.backend->inspect(values->path);
        if(encoding(mask_info,mask->path)!="array"||mask_info.dtype!=DType::boolean||mask_info.shape.size()!=1||mask_info.shape!=values_info.shape)fail(ErrorCode::shape_mismatch,source->path,"nullable mask must be a matching boolean vector");
        auto values_code=encoding(values_info,values->path);
        if(code=="nullable-string-array"){if(values_code!="string-array")fail(ErrorCode::unsupported_encoding,values->path,"nullable string values require string-array");auto it=info.attrs.find("na-value");if(it!=info.attrs.end()){auto value=string_attr(info,"na-value",source->path);if(value!="NA"&&value!="NaN")fail(ErrorCode::unsupported_encoding,source->path,"unsupported nullable string sentinel");}}
        else if(values_code!="array"||values_info.strings||(code=="nullable-boolean"?values_info.dtype!=DType::boolean:values_info.dtype==DType::boolean||values_info.dtype==DType::float32||values_info.dtype==DType::float64))fail(ErrorCode::unsupported_dtype,values->path,"nullable values have an incompatible dtype");
        destination.create_group(path,info.attrs);copy_node(destination,child(path,"values"),values,budget,string_scratch_bytes);copy_node(destination,child(path,"mask"),mask,budget,string_scratch_bytes);return;
    }
    if(code=="dict"||code=="raw"||code=="anndata"){
        destination.create_group(path,info.attrs);
        const auto keys=session.backend->children(source->path);
        for(const auto& key:keys) {
            auto item=source->slice_history ? color_child(source,key) : subnode(source,key);
            if(item)copy_node(destination,child(path,key),item,budget,string_scratch_bytes);
        }
        return;
    }
    fail(ErrorCode::unsupported_encoding,source->path,"unsupported streamed encoding");
}
void copy_node(Backend& destination,std::string_view path,const Node& source,std::size_t scratch_bytes,
               std::optional<std::size_t> string_scratch_bytes){
    copy_node_impl(destination,path,source,scratch_bytes,string_scratch_bytes);
    if (!source->empty_mapping && (&destination == source->session->backend.get() || destination.identity() == source->session->backend->identity()))
        destination.copy_attributes(source->path,path);
}
void export_store(Backend& destination,const BackedAnnDataView& original,std::size_t scratch_bytes,Validation validation){
    const auto& retained=detail::BackedAccess::get(original);if(!retained||!retained->session)fail(ErrorCode::invalid_argument,"store","empty backed view");
    (void)original.n_obs();
    auto source=validation_view(original,validation);
    const auto& dataset=detail::BackedAccess::get(source);
    check_dataset(source);
    const auto budget=std::min(scratch_bytes,dataset->session->options.scratch_bytes);
    if(budget<64)fail(ErrorCode::invalid_argument,"store","scratch budget must be at least 64 bytes");
    const bool same_store=destination.identity()==dataset->session->backend->identity();
    Node selected_uns;
    if(dataset->session->backend->inspect("/uns").exists)selected_uns=detail::BackedAccess::get(source.uns());
    check_export_metadata(*dataset->session,"/",same_store,selected_uns);
    auto root=dataset->session->backend->inspect("/");
    if(root.has_opaque_attrs && !same_store)fail(ErrorCode::unsupported_dtype,"/","opaque backend root attributes cannot be exported");
    destination.create_group("/",root.attrs);
    auto copy=[&](std::string_view path,const auto& wrapper){copy_node(destination,path,detail::BackedAccess::get(wrapper),budget);};
    copy("/obs",source.obs());copy("/var",source.var());
    if(source.has_x())copy("/X",source.x());
    auto mapping=[&](std::string_view base,const std::vector<std::string>& keys,const auto& get){
        auto info=dataset->session->backend->inspect(base);if(info.has_opaque_attrs && !same_store)fail(ErrorCode::unsupported_dtype,base,"opaque mapping attributes cannot be exported");destination.create_group(base,info.exists?info.attrs:attrs("dict","0.1.0"));
        for(const auto& key:keys)copy(child(base,key),get(key));
    };
    mapping("/layers",source.layer_names(),[&](const auto& key){return source.layer(key);});
    mapping("/obsm",source.obsm_names(),[&](const auto& key){return source.obsm(key);});
    mapping("/varm",source.varm_names(),[&](const auto& key){return source.varm(key);});
    mapping("/obsp",source.obsp_names(),[&](const auto& key){return source.obsp(key);});
    mapping("/varp",source.varp_names(),[&](const auto& key){return source.varp(key);});
    if(dataset->session->backend->inspect("/uns").exists)copy("/uns",source.uns());else destination.create_group("/uns",attrs("dict","0.1.0"));
    if(source.has_raw()){
        auto info=dataset->session->backend->inspect("/raw");if(info.has_opaque_attrs && !same_store)fail(ErrorCode::unsupported_dtype,"/raw","opaque raw attributes cannot be exported");if(encoding(info,"/raw")!="raw")fail(ErrorCode::unsupported_encoding,"/raw","expected raw encoding");destination.create_group("/raw",info.attrs);
        copy("/raw/var",source.raw_var());if(source.has_raw_x())copy("/raw/X",source.raw_x());
        mapping("/raw/varm",source.raw_varm_names(),[&](const auto& key){return source.raw_varm(key);});
    }
    destination.flush();
}
} // namespace anndata::backed_detail

namespace anndata {
void BackedAnnData::validate_all() const {
    backed_detail::Sink sink(location(),impl_->session->backend->identity());
    backed_detail::export_store(sink,view(),impl_->session->options.scratch_bytes,Validation::full);
}
}
