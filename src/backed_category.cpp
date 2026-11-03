module;
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <variant>
#include <sys/types.h>
#include <unistd.h>

module anndata.backed;
import :storage;

namespace anndata::backed_detail {
namespace {
using Node = std::shared_ptr<detail::BackedNode>;
[[noreturn]] void fail(ErrorCode code, std::string_view path, std::string message) {
    throw Error(code, std::string(path), std::move(message));
}
[[noreturn]] void io_fail(std::string_view path, std::string_view operation) {
    auto reason = errno ? std::strerror(errno) : "temporary storage failure";
    fail(ErrorCode::io_error, path, std::string(operation) + ": " + reason);
}
struct TempFile {
    std::FILE* file = nullptr;
    int descriptor = -1;
    explicit TempFile(std::string_view path) {
        file = std::tmpfile();
        if (!file) io_fail(path, "create temporary category mapping");
        descriptor = ::fileno(file);
        if (descriptor < 0) {
            const auto saved = errno; std::fclose(file); file = nullptr; errno = saved;
            io_fail(path, "inspect temporary category mapping");
        }
        if (std::setvbuf(file, nullptr, _IONBF, 0) != 0) {
            const auto saved = errno; std::fclose(file); file = nullptr; errno = saved;
            io_fail(path, "configure temporary category mapping");
        }
    }
    ~TempFile() { if (file) std::fclose(file); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
};
void read_exact(int descriptor, void* p, std::size_t bytes, std::size_t offset, std::string_view path) {
    if (offset > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) ||
        bytes > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) - offset)
        fail(ErrorCode::invalid_argument, path, "temporary category offset exceeds file range");
    std::size_t done = 0;
    while (done < bytes) {
        auto result = ::pread(descriptor, static_cast<char*>(p) + done, bytes - done, static_cast<off_t>(offset + done));
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) io_fail(path, "read temporary category mapping");
        done += static_cast<std::size_t>(result);
    }
}
void write_exact(int descriptor, const void* p, std::size_t bytes, std::size_t offset, std::string_view path) {
    if (offset > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) ||
        bytes > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) - offset)
        fail(ErrorCode::invalid_argument, path, "temporary category offset exceeds file range");
    std::size_t done = 0;
    while (done < bytes) {
        auto result = ::pwrite(descriptor, static_cast<const char*>(p) + done, bytes - done, static_cast<off_t>(offset + done));
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) io_fail(path, "write temporary category mapping");
        done += static_cast<std::size_t>(result);
    }
}
struct Bitset {
    std::size_t bytes = 0, page_size = 0;
    std::string path;
    std::vector<unsigned char> memory, page;
    std::unique_ptr<TempFile> file;
    std::size_t page_begin = std::numeric_limits<std::size_t>::max(), page_count = 0;
    bool dirty = false;
    Bitset(std::size_t bits, std::size_t budget, std::string_view p):path(p) {
        bytes = bits / 8 + (bits % 8 != 0);
        if (bytes <= budget / 8) memory.assign(bytes, 0);
        else {
            file = std::make_unique<TempFile>(path);
            if (bytes > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) ||
                ::ftruncate(file->descriptor, static_cast<off_t>(bytes)) != 0)
                io_fail(path, "size temporary category bitmap");
            page_size = std::max<std::size_t>(1, std::min<std::size_t>(4096, budget / 16));
            page.resize(page_size);
        }
    }
    ~Bitset() { try { flush(); } catch (...) {} }
    void flush() {
        if (!dirty) return;
        write_exact(file->descriptor, page.data(), page_count, page_begin, path); dirty = false;
    }
    unsigned char& byte(std::size_t i) {
        if (!file) return memory[i];
        const auto begin = (i / page_size) * page_size;
        if (begin != page_begin) {
            flush(); page_begin = begin; page_count = std::min(page_size, bytes - begin);
            read_exact(file->descriptor, page.data(), page_count, begin, path);
        }
        return page[i - page_begin];
    }
    void set(std::size_t i) { byte(i / 8) |= static_cast<unsigned char>(1u << (i % 8)); if (file) dirty = true; }
    bool get(std::size_t i) { return (byte(i / 8) & (1u << (i % 8))) != 0; }
};
template<class F> decltype(auto) visit_integer(DType dtype, std::string_view path, F&& f) {
    switch(dtype) {
    case DType::int8:return f.template operator()<std::int8_t>();
    case DType::uint8:return f.template operator()<std::uint8_t>();
    case DType::int16:return f.template operator()<std::int16_t>();
    case DType::uint16:return f.template operator()<std::uint16_t>();
    case DType::int32:return f.template operator()<std::int32_t>();
    case DType::uint32:return f.template operator()<std::uint32_t>();
    case DType::int64:return f.template operator()<std::int64_t>();
    case DType::uint64:return f.template operator()<std::uint64_t>();
    default:fail(ErrorCode::unsupported_dtype,path,"categorical codes require an integer dtype");
    }
}
template<class T> std::int64_t decode_code(const std::byte* p, std::string_view path) {
    T value;std::memcpy(&value,p,sizeof value);
    if constexpr(std::is_unsigned_v<T>)
        if (static_cast<std::uint64_t>(value)>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            fail(ErrorCode::out_of_bounds,path,"categorical code exceeds supported range");
    auto result=static_cast<std::int64_t>(value);
    if(result < -1)fail(ErrorCode::out_of_bounds,path,"categorical code below missing sentinel");
    return result;
}
template<class T> void encode_code(std::byte* p,std::int64_t value,std::string_view path) {
    if constexpr(std::is_unsigned_v<T>) {
        if(value<0 || static_cast<std::uint64_t>(value)>std::numeric_limits<T>::max())
            fail(ErrorCode::out_of_bounds,path,"recoded categorical value does not fit dtype");
    } else if(value<std::numeric_limits<T>::min() || value>std::numeric_limits<T>::max())
        fail(ErrorCode::out_of_bounds,path,"recoded categorical value does not fit dtype");
    T result=static_cast<T>(value);std::memcpy(p,&result,sizeof result);
}
struct CodeSource {
    Node category, codes;
    NodeInfo code_info;
    Axis rows;
    std::size_t categories = 0;
};
CodeSource code_source(const Node& category) {
    check_node(category);
    auto& session=*category->session;
    if(encoding(session.backend->inspect(category->path),category->path)!="categorical")
        fail(ErrorCode::unsupported_encoding,category->path,"expected a categorical column");
    auto categories_path=child(category->path,"categories");
    auto cat_shape=logical_shape(session,categories_path);
    if(cat_shape.size()!=1)fail(ErrorCode::shape_mismatch,categories_path,"category dictionary must have rank one");
    auto codes_path=child(category->path,"codes");
    auto code_info=session.backend->inspect(codes_path);
    if(!code_info.exists||code_info.group||code_info.strings||code_info.shape.size()!=1 ||
       encoding(code_info,codes_path)!="array")
        fail(ErrorCode::unsupported_encoding,codes_path,"category codes must be a one-dimensional numerical array");
    (void)visit_integer(code_info.dtype,codes_path,[]<class T>{return sizeof(T);});
    Axis rows=category->axes.empty()?Axis{code_info.shape[0]}:category->axes.at(0);
    if(category->axes.size()>1)fail(ErrorCode::shape_mismatch,category->path,"category selection must have rank one");
    auto codes=node(category->session,codes_path,{rows});
    return {category,std::move(codes),std::move(code_info),std::move(rows),cat_shape[0]};
}
template<class F> void scan_codes(const CodeSource& source,std::size_t budget,F&& visit) {
    const auto width=dtype_size(source.code_info.dtype);
    const auto cap=std::max<std::size_t>(1,std::min<std::size_t>(65536,budget/(4*width)));
    std::vector<std::byte> buffer;
    auto& backend=*source.category->session->backend;
    visit_integer(source.code_info.dtype,source.codes->path,[&]<class T>{
        for(std::size_t i=0;i<source.rows.size;){
            const auto first=source.rows.at(i);std::size_t run=1;
            while(run<cap && run<source.rows.size-i && first<=std::numeric_limits<std::size_t>::max()-run && source.rows.at(i+run)==first+run)++run;
            buffer.resize(run*width);backend.read_numeric(source.codes->path,{first},{run},buffer.data());
            for(std::size_t j=0;j<run;++j)visit(decode_code<T>(buffer.data()+j*width,source.codes->path));
            i+=run;
        }
    });
}
}

struct CategoryPlan {
    std::size_t source_count = 0, selected_count = 0, budget = 0;
    std::string path;
    std::vector<std::size_t> positions;
    std::unique_ptr<TempFile> positions_file;
    mutable std::vector<std::size_t> page;
    mutable std::size_t page_begin=std::numeric_limits<std::size_t>::max(),page_count=0;
    std::shared_ptr<CategoryPlan> outer, before, after;
};

std::size_t scratch_remaining(const Node& node) {
    check_node(node);
    std::size_t remaining = node->session->options.scratch_bytes;
    auto charge = [&](auto&& self, const std::shared_ptr<CategoryPlan>& plan) -> void {
        if (!plan) return;
        auto allocation = [&](std::size_t elements) {
            if (elements > remaining / sizeof(std::size_t))
                fail(ErrorCode::invalid_argument,node->path,"retained category selections exhaust the scratch budget");
            remaining -= elements * sizeof(std::size_t);
        };
        allocation(plan->positions.capacity());
        allocation(plan->page.capacity());
        self(self,plan->outer);self(self,plan->before);self(self,plan->after);
    };
    for (const auto& axis : node->axes) charge(charge,axis.category_positions);
    charge(charge,node->code_recode);
    if (node->color_override) charge(charge,node->color_override->axis.category_positions);
    if (!remaining) fail(ErrorCode::invalid_argument,node->path,"retained category selections exhaust the scratch budget");
    return remaining;
}

std::size_t category_count(const std::shared_ptr<CategoryPlan>& plan) {
    if(!plan)fail(ErrorCode::invalid_argument,"categorical","empty category plan");return plan->selected_count;
}
std::size_t category_source_count(const std::shared_ptr<CategoryPlan>& plan) {
    if(!plan)fail(ErrorCode::invalid_argument,"categorical","empty category plan");return plan->source_count;
}
std::size_t category_position(const std::shared_ptr<CategoryPlan>& plan,std::size_t i) {
    if(!plan||i>=plan->selected_count)fail(ErrorCode::out_of_bounds,"categorical","selected category position out of bounds");
    if(plan->after) {
        auto result=category_position(plan->after,i);
        if(plan->before) {
            if(result>static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))fail(ErrorCode::out_of_bounds,plan->path,"category position exceeds signed range");
            auto mapped=category_recode(plan->before,static_cast<std::int64_t>(result));
            if(mapped<0)fail(ErrorCode::invalid_argument,plan->path,"nested selected categories are not a subset of the previous selection");
            result=static_cast<std::size_t>(mapped);
        }
        return plan->outer?category_position(plan->outer,result):result;
    }
    if(!plan->positions_file)return plan->positions[i];
    auto capacity=plan->page.size(),begin=(i/capacity)*capacity;
    if(begin!=plan->page_begin){
        plan->page_begin=begin;plan->page_count=std::min(capacity,plan->selected_count-begin);
        if(begin>std::numeric_limits<std::size_t>::max()/sizeof(std::size_t))fail(ErrorCode::invalid_argument,plan->path,"category mapping offset overflow");
        read_exact(plan->positions_file->descriptor,plan->page.data(),plan->page_count*sizeof(std::size_t),begin*sizeof(std::size_t),plan->path);
    }
    return plan->page[i-begin];
}
std::int64_t category_recode(const std::shared_ptr<CategoryPlan>& plan,std::int64_t old) {
    if(!plan)fail(ErrorCode::invalid_argument,"categorical","empty category plan");
    if(old==-1)return -1;
    if(old<0||static_cast<std::uint64_t>(old)>=plan->source_count)
        fail(ErrorCode::out_of_bounds,plan->path,"categorical code is outside dictionary bounds");
    auto target=static_cast<std::size_t>(old);std::size_t lo=0,hi=plan->selected_count;
    while(lo<hi){auto mid=lo+(hi-lo)/2;if(category_position(plan,mid)<target)lo=mid+1;else hi=mid;}
    if(lo==plan->selected_count || category_position(plan,lo)!=target)return -1;
    if(lo>static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        fail(ErrorCode::out_of_bounds,plan->path,"selected category index exceeds signed range");
    return static_cast<std::int64_t>(lo);
}
std::shared_ptr<CategoryPlan> compose_category_plans(const std::shared_ptr<CategoryPlan>& outer,
    const std::shared_ptr<CategoryPlan>& before,const std::shared_ptr<CategoryPlan>& after) {
    if(!after)fail(ErrorCode::invalid_argument,"categorical","composed category selection has no final plan");
    auto result=std::make_shared<CategoryPlan>();result->outer=outer;result->before=before;result->after=after;
    result->selected_count=category_count(after);
    result->source_count=outer?category_source_count(outer):(before?category_count(before):category_source_count(after));
    result->path=after->path;return result;
}
void recode_codes_buffer(const std::shared_ptr<CategoryPlan>& plan,DType dtype,void* data,std::size_t count,std::ptrdiff_t stride,std::string_view path){
    if(count && !data)fail(ErrorCode::invalid_argument,path,"categorical destination buffer is null");
    auto* bytes=static_cast<std::byte*>(data);
    visit_integer(dtype,path,[&]<class T>{
        for(std::size_t i=0;i<count;++i){
            auto* p=bytes+static_cast<std::ptrdiff_t>(i)*stride;auto old=decode_code<T>(p,path);auto mapped=category_recode(plan,old);
            if(old>=0&&mapped<0)fail(ErrorCode::invalid_argument,path,"source codes changed outside the selected dictionary; acquire a fresh view");
            encode_code<T>(p,mapped,path);
        }
    });
}
std::shared_ptr<CategoryPlan> make_category_plan(const Node& category,std::size_t scratch_bytes) {
    auto source=code_source(category);auto budget=std::min(scratch_bytes,category->session->options.scratch_bytes);
    if(budget<64)fail(ErrorCode::invalid_argument,category->path,"scratch budget must be at least 64 bytes");
    auto plan=std::make_shared<CategoryPlan>();plan->source_count=source.categories;plan->budget=budget;plan->path=category->path;
    if(source.rows.size <= budget/(8*sizeof(std::size_t))) {
        std::vector<std::size_t> positions; positions.reserve(source.rows.size);
        scan_codes(source,budget,[&](std::int64_t code){
            if(code<0)return;
            if(static_cast<std::uint64_t>(code)>=source.categories)fail(ErrorCode::out_of_bounds,source.codes->path,"categorical code is outside dictionary bounds");
            positions.push_back(static_cast<std::size_t>(code));
        });
        std::sort(positions.begin(),positions.end());positions.erase(std::unique(positions.begin(),positions.end()),positions.end());
        plan->selected_count=positions.size();plan->positions=std::move(positions);return plan;
    }
    std::unique_ptr<Bitset> used;
    scan_codes(source,budget,[&](std::int64_t code){
        if(code<0)return;
        if(static_cast<std::uint64_t>(code)>=source.categories)fail(ErrorCode::out_of_bounds,source.codes->path,"categorical code is outside dictionary bounds");
        if(!used)used=std::make_unique<Bitset>(source.categories,budget,category->path);
        used->set(static_cast<std::size_t>(code));
    });
    if(!used)return plan;
    for(std::size_t i=0;i<source.categories;++i)if(used->get(i))++plan->selected_count;
    if(plan->selected_count<=budget/(8*sizeof(std::size_t))) {
        plan->positions.reserve(plan->selected_count);
        for(std::size_t i=0;i<source.categories;++i)if(used->get(i))plan->positions.push_back(i);
    } else {
        plan->positions_file=std::make_unique<TempFile>(category->path);
        auto capacity=std::max<std::size_t>(1,std::min<std::size_t>(512,budget/(16*sizeof(std::size_t))));
        plan->page.resize(capacity);std::size_t filled=0,written=0;
        for(std::size_t i=0;i<source.categories;++i)if(used->get(i)){
            plan->page[filled++]=i;
            if(filled==capacity){write_exact(plan->positions_file->descriptor,plan->page.data(),filled*sizeof(std::size_t),written*sizeof(std::size_t),plan->path);written+=filled;filled=0;}
        }
        if(filled)write_exact(plan->positions_file->descriptor,plan->page.data(),filled*sizeof(std::size_t),written*sizeof(std::size_t),plan->path);
    }
    return plan;
}

void copy_recoded_codes(Backend& destination,std::string_view path,const Node& codes,
                        std::size_t scratch_bytes,const std::shared_ptr<CategoryPlan>& plan) {
    check_node(codes);auto& session=*codes->session;auto info=session.backend->inspect(codes->path);
    if(!plan||info.group||info.strings||info.shape.size()!=1||codes->axes.size()!=1||encoding(info,codes->path)!="array")
        fail(ErrorCode::unsupported_encoding,codes->path,"recoded codes require a one-dimensional numerical array");
    if(info.has_opaque_attrs && destination.identity()!=session.backend->identity())
        fail(ErrorCode::unsupported_dtype,codes->path,"opaque attributes cannot be represented in a different store");
    auto width=dtype_size(info.dtype);auto budget=std::min(scratch_bytes,session.options.scratch_bytes);
    auto capacity=std::max<std::size_t>(1,std::min<std::size_t>(65536,budget/(4*width)));
    destination.create_numeric(path,info.dtype,{codes->axes[0].size},info.attrs,info.byte_order);
    std::vector<std::byte> buffer;
    visit_integer(info.dtype,codes->path,[&]<class T>{
        for(std::size_t i=0;i<codes->axes[0].size;){
            auto first=codes->axes[0].at(i);std::size_t run=1;
            while(run<capacity && run<codes->axes[0].size-i && first<=std::numeric_limits<std::size_t>::max()-run && codes->axes[0].at(i+run)==first+run)++run;
            buffer.resize(run*width);session.backend->read_numeric(codes->path,{first},{run},buffer.data());
            for(std::size_t j=0;j<run;++j){
                auto old=decode_code<T>(buffer.data()+j*width,codes->path);auto mapped=category_recode(plan,old);
                if(old>=0 && mapped<0)fail(ErrorCode::invalid_argument,codes->path,"source codes changed outside the selected dictionary; acquire a fresh view");
                encode_code<T>(buffer.data()+j*width,mapped,codes->path);
            }
            destination.write_numeric(path,{i},{run},buffer.data());i+=run;
        }
    });
    if(destination.identity()==session.backend->identity())destination.copy_attributes(codes->path,path);
}
void copy_pruned_category(Backend& destination,std::string_view path,const Node& category,std::size_t scratch_bytes,
                          std::shared_ptr<CategoryPlan> plan) {
    auto source=code_source(category);
    const auto budget=std::min(scratch_bytes,category->session->options.scratch_bytes);
    if(budget<128)fail(ErrorCode::invalid_argument,category->path,"scratch budget must be at least 128 bytes for categorical transfer");
    if(!plan)plan=make_category_plan(category,budget);
    const auto transfer_budget=budget/2;
    auto group_info=category->session->backend->inspect(category->path);
    if(group_info.has_opaque_attrs && destination.identity()!=category->session->backend->identity())
        fail(ErrorCode::unsupported_dtype,category->path,"opaque attributes cannot be represented in a different store");
    auto dict=node(category->session,child(category->path,"categories"),{Axis{category_count(plan),0,1,{},plan}});
    validate_category_dictionary(dict,transfer_budget);
    destination.create_group(path,group_info.attrs);
    copy_recoded_codes(destination,child(path,"codes"),source.codes,transfer_budget,plan);
    // Validation and code packing have returned before this copy. Numeric
    // children reserve the backend workspace against the global remainder;
    // keep the existing narrower string staging cap for nullable dictionaries.
    copy_node(destination,child(path,"categories"),dict,budget,transfer_budget);
    if(destination.identity()==category->session->backend->identity())destination.copy_attributes(category->path,path);
}
} // namespace anndata::backed_detail
