module;
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
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
#include <utility>
#include <vector>
#include <variant>
// Temporary-run offsets use the POSIX large-file API (fseeko/ftello).
#include <sys/types.h>

module anndatapp.backed;
import :storage;

namespace anndatapp::backed_detail {
namespace {

[[noreturn]] void fail(ErrorCode code, std::string_view path, std::string message) {
    throw Error(code, std::string(path), std::move(message));
}

[[noreturn]] void io_fail(std::string_view path, std::string_view operation) {
    const auto reason = errno ? std::strerror(errno) : "temporary run operation failed";
    fail(ErrorCode::io_error, path, std::string(operation) + ": " + reason);
}

void seek_file(std::FILE* file, off_t offset, std::string_view path) {
    if (offset < 0 || ::fseeko(file, offset, SEEK_SET) != 0)
        io_fail(path, "seek temporary uniqueness run");
}

off_t tell_file(std::FILE* file, std::string_view path) {
    const auto offset = ::ftello(file);
    if (offset < 0) io_fail(path, "locate temporary uniqueness record");
    return offset;
}

struct TempRun {
    std::FILE* file = nullptr;
    std::size_t records = 0;

    TempRun() = default;
    explicit TempRun(std::FILE* value) : file(value) {}
    ~TempRun() { if (file) std::fclose(file); }
    TempRun(const TempRun&) = delete;
    TempRun& operator=(const TempRun&) = delete;
    TempRun(TempRun&& other) noexcept
        : file(std::exchange(other.file, nullptr)), records(std::exchange(other.records, 0)) {}
    TempRun& operator=(TempRun&& other) noexcept {
        if (this != &other) {
            if (file) std::fclose(file);
            file = std::exchange(other.file, nullptr);
            records = std::exchange(other.records, 0);
        }
        return *this;
    }
};

TempRun make_run(std::string_view path) {
    errno = 0;
    auto* file = std::tmpfile();
    if (!file) io_fail(path, "create temporary uniqueness run");
    return TempRun(file);
}

void write_exact(std::FILE* file, const void* data, std::size_t bytes, std::string_view path) {
    if (bytes && std::fwrite(data, 1, bytes, file) != bytes)
        io_fail(path, "write temporary uniqueness run");
}

void read_exact(std::FILE* file, void* data, std::size_t bytes, std::string_view path) {
    if (bytes && std::fread(data, 1, bytes, file) != bytes)
        io_fail(path, "read temporary uniqueness run");
}

void finish_run(TempRun& run, std::string_view path) {
    if (std::fflush(run.file) != 0) io_fail(path, "flush temporary uniqueness run");
    seek_file(run.file, 0, path);
}

void write_string(TempRun& run, const std::string& value, std::string_view path) {
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
    const auto length = static_cast<std::uint64_t>(value.size());
    write_exact(run.file, &length, sizeof(length), path);
    write_exact(run.file, value.data(), value.size(), path);
    ++run.records;
}

struct Head {
    std::FILE* file = nullptr;
    std::size_t remaining = 0;
    off_t next_header = 0;
    off_t data_offset = 0;
    std::uint64_t length = 0;
    bool valid = false;
};

void load_head(Head& head, std::string_view path) {
    if (!head.remaining) {
        head.valid = false;
        return;
    }
    seek_file(head.file, head.next_header, path);
    read_exact(head.file, &head.length, sizeof(head.length), path);
    head.data_offset = tell_file(head.file, path);
    if (head.length > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max() - head.data_offset))
        fail(ErrorCode::io_error, path, "temporary uniqueness record exceeds file offset range");
    head.next_header = head.data_offset + static_cast<off_t>(head.length);
    --head.remaining;
    head.valid = true;
}

void copy_record(TempRun& destination, const Head& source, std::string_view path) {
    write_exact(destination.file, &source.length, sizeof(source.length), path);
    seek_file(source.file, source.data_offset, path);
    std::array<char, 4096> buffer{};
    std::uint64_t remaining = source.length;
    while (remaining) {
        const auto amount = static_cast<std::size_t>(std::min<std::uint64_t>(remaining, buffer.size()));
        read_exact(source.file, buffer.data(), amount, path);
        write_exact(destination.file, buffer.data(), amount, path);
        remaining -= amount;
    }
    ++destination.records;
}

int compare_records(const Head& left, const Head& right, std::string_view path) {
    const auto common = std::min(left.length, right.length);
    std::array<char, 256> left_buffer{}, right_buffer{};
    std::uint64_t offset = 0;
    while (offset < common) {
        const auto amount = static_cast<std::size_t>(std::min<std::uint64_t>(common - offset, left_buffer.size()));
        if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            left.data_offset > std::numeric_limits<off_t>::max() - static_cast<off_t>(offset) ||
            right.data_offset > std::numeric_limits<off_t>::max() - static_cast<off_t>(offset))
            fail(ErrorCode::io_error, path, "temporary uniqueness record exceeds file offset range");
        seek_file(left.file, left.data_offset + static_cast<off_t>(offset), path);
        read_exact(left.file, left_buffer.data(), amount, path);
        seek_file(right.file, right.data_offset + static_cast<off_t>(offset), path);
        read_exact(right.file, right_buffer.data(), amount, path);
        const int order = std::memcmp(left_buffer.data(), right_buffer.data(), amount);
        if (order) return order < 0 ? -1 : 1;
        offset += amount;
    }
    if (left.length < right.length) return -1;
    if (left.length > right.length) return 1;
    return 0;
}

struct MergeResult {
    bool unique = true;
    TempRun run;
};

MergeResult merge_two(TempRun left, TempRun right, std::string_view path) {
    auto output = make_run(path);
    Head a{left.file, left.records}, b{right.file, right.records};
    load_head(a, path);
    load_head(b, path);
    while (a.valid && b.valid) {
        const int order = compare_records(a, b, path);
        if (!order) return {false, {}};
        if (order < 0) {
            copy_record(output, a, path);
            load_head(a, path);
        } else {
            copy_record(output, b, path);
            load_head(b, path);
        }
    }
    while (a.valid) {
        copy_record(output, a, path);
        load_head(a, path);
    }
    while (b.valid) {
        copy_record(output, b, path);
        load_head(b, path);
    }
    finish_run(output, path);
    return {true, std::move(output)};
}

struct RunLevels {
    std::vector<std::optional<TempRun>> levels;
    bool unique = true;
};

void add_run(RunLevels& levels, TempRun value, std::string_view path) {
    std::size_t level = 0;
    while (level < levels.levels.size() && levels.levels[level]) {
        auto merged = merge_two(std::move(*levels.levels[level]), std::move(value), path);
        levels.levels[level].reset();
        if (!merged.unique) {
            levels.unique = false;
            return;
        }
        value = std::move(merged.run);
        ++level;
    }
    if (level == levels.levels.size()) levels.levels.emplace_back();
    levels.levels[level].emplace(std::move(value));
}

TempRun write_sorted_run(std::vector<std::string>& values, std::string_view path, bool& unique) {
    std::sort(values.begin(), values.end());
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i - 1] == values[i]) {
            unique = false;
            return {};
        }
    }
    auto run = make_run(path);
    for (const auto& value : values) write_string(run, value, path);
    finish_run(run, path);
    return run;
}

bool levels_unique(const RunLevels& levels, std::string_view path) {
    std::vector<Head> heads;
    heads.reserve(levels.levels.size());
    for (const auto& run : levels.levels) {
        if (!run) continue;
        heads.push_back(Head{run->file, run->records});
        load_head(heads.back(), path);
    }

    std::optional<Head> previous;
    while (true) {
        std::size_t best = heads.size();
        for (std::size_t i = 0; i < heads.size(); ++i) {
            if (!heads[i].valid) continue;
            if (best == heads.size() || compare_records(heads[i], heads[best], path) < 0)
                best = i;
        }
        if (best == heads.size()) return true;
        if (previous && compare_records(*previous, heads[best], path) == 0) return false;
        previous = heads[best];
        load_head(heads[best], path);
    }
}

std::size_t next_run_capacity(std::size_t scratch, std::size_t remaining) {
    // Reserve at most half of scratch for string objects up front; actual
    // variable-length payload capacities are accounted as values arrive.
    const auto slots = std::max<std::size_t>(1, scratch / (2 * sizeof(std::string)));
    return std::min(slots, remaining);
}

class UniqueAccumulator {
public:
    UniqueAccumulator(std::string_view path, std::size_t scratch, std::size_t count)
        : path_(path), scratch_(scratch), capacity_(next_run_capacity(scratch, count)) {
        initial_capacity_ = std::min<std::size_t>(capacity_, 256);
        if (initial_capacity_) values_.reserve(initial_capacity_);
    }

    bool add(std::string&& value, std::size_t transient_overhead) {
        if (!unique_) return false;
        const auto incoming = value.capacity() + 1;
        if (values_.size() == values_.capacity() && !grow_for_value(incoming, transient_overhead)) {
            if (!flush_values()) return false;
        }
        if (!fits(incoming, transient_overhead)) {
            if (!flush_values()) return false;
            if (!fits(incoming, transient_overhead)) {
                auto run = make_run(path_);
                write_string(run, value, path_);
                finish_run(run, path_);
                add_run(levels_, std::move(run), path_);
                unique_ = levels_.unique;
                return unique_;
            }
        }
        values_.push_back(std::move(value));
        value_bytes_ += incoming;
        return true;
    }

    bool finish() {
        if (!unique_) return false;
        if (levels_.levels.empty()) {
            std::sort(values_.begin(), values_.end());
            for (std::size_t i = 1; i < values_.size(); ++i)
                if (values_[i - 1] == values_[i]) return false;
            return true;
        }
        if (!flush_values()) return false;
        return levels_unique(levels_, path_);
    }

private:
    bool fits(std::size_t incoming, std::size_t transient_overhead) const {
        std::size_t used = sizeof(values_);
        auto add_size = [&](std::size_t amount) {
            if (amount > std::numeric_limits<std::size_t>::max() - used) return false;
            used += amount;
            return true;
        };
        if (values_.capacity() > (std::numeric_limits<std::size_t>::max() - used) / sizeof(std::string)) return false;
        if (!add_size(values_.capacity() * sizeof(std::string)) || !add_size(value_bytes_) ||
            !add_size(transient_overhead) || !add_size(incoming)) return false;
        return used <= scratch_;
    }

    bool grow_for_value(std::size_t incoming, std::size_t transient_overhead) {
        const auto current = values_.capacity();
        if (current >= capacity_) return false;
        const auto doubled = current > std::numeric_limits<std::size_t>::max() / 2
            ? std::numeric_limits<std::size_t>::max() : std::max<std::size_t>(1, current * 2);
        const auto requested = std::min(capacity_, doubled);
        std::size_t peak = sizeof(values_);
        const auto max = std::numeric_limits<std::size_t>::max();
        if (current > (max - peak) / sizeof(std::string)) return false;
        peak += current * sizeof(std::string);
        if (requested > (max - peak) / sizeof(std::string)) return false;
        peak += requested * sizeof(std::string);
        auto add_size = [&](std::size_t amount) {
            if (amount > max - peak) return false;
            peak += amount;
            return true;
        };
        if (!add_size(value_bytes_) || !add_size(transient_overhead) || !add_size(incoming) || peak > scratch_)
            return false;
        values_.reserve(requested);
        return fits(incoming, transient_overhead);
    }

    void reinitialize_values() {
        std::vector<std::string>().swap(values_);
        if (initial_capacity_) values_.reserve(initial_capacity_);
        value_bytes_ = 0;
    }

    bool flush_values() {
        if (values_.empty()) return true;
        bool unique = true;
        auto run = write_sorted_run(values_, path_, unique);
        if (!unique) {
            unique_ = false;
            return false;
        }
        reinitialize_values();
        add_run(levels_, std::move(run), path_);
        unique_ = levels_.unique;
        return unique_;
    }

    std::string path_;
    std::size_t scratch_ = 0;
    std::size_t capacity_ = 0;
    std::size_t initial_capacity_ = 0;
    std::vector<std::string> values_;
    std::size_t value_bytes_ = 0;
    RunLevels levels_;
    bool unique_ = true;
};

void reject_nonunique(const std::shared_ptr<detail::BackedNode>& index, bool full_physical_axis) {
    if (full_physical_axis) index->revision->unique_index = false;
    fail(ErrorCode::invalid_argument, index->path,
         "label-list indexing requires unique matching axis names");
}

std::string numeric_key(const std::byte* bytes, DType dtype, std::string_view path) {
    const auto width = dtype_size(dtype);
    std::string key(width, '\0');
    std::memcpy(key.data(), bytes, width);
    if (dtype == DType::boolean) {
        key[0] = std::to_integer<unsigned char>(bytes[0]) ? '\1' : '\0';
    } else if (dtype == DType::float32) {
        float value;
        std::memcpy(&value, bytes, sizeof(value));
        if (std::isnan(value)) fail(ErrorCode::invalid_argument, path, "category dictionaries cannot contain NaN");
        if (value == 0) std::fill(key.begin(), key.end(), '\0');
    } else if (dtype == DType::float64) {
        double value;
        std::memcpy(&value, bytes, sizeof(value));
        if (std::isnan(value)) fail(ErrorCode::invalid_argument, path, "category dictionaries cannot contain NaN");
        if (value == 0) std::fill(key.begin(), key.end(), '\0');
    }
    return key;
}

} // namespace

void require_unique_index(const std::shared_ptr<detail::BackedNode>& index, const Axis& selected, std::size_t scratch_bytes) {
    check_node(index);
    auto& session = *index->session;
    const auto shape = logical_shape(session, index->path);
    if (shape.size() != 1)
        fail(ErrorCode::shape_mismatch, index->path, "table index must be one-dimensional");
    const auto budget = scratch_bytes ? std::min(scratch_bytes,session.options.scratch_bytes) : session.options.scratch_bytes;
    const auto source_length = shape.front();
    const bool full_physical_axis = selected.all(source_length);
    if (full_physical_axis && index->revision->unique_index) {
        if (!*index->revision->unique_index) reject_nonunique(index, true);
        return;
    }

    UniqueAccumulator unique(index->path, budget, selected.size);
    const auto batch_budget = std::max<std::size_t>(1, budget / 2);
    const auto batch_items = std::max<std::size_t>(1, std::min<std::size_t>(8192, batch_budget / (2 * sizeof(std::string))));
    for (std::size_t i = 0; i < selected.size;) {
        std::size_t requested = 1;
        while (requested < batch_items && requested < selected.size - i &&
               selected.at(i + requested) == selected.at(i) + requested) ++requested;
        auto decoded = session.backend->read_strings_bounded(index->path, {selected.at(i)}, {requested}, batch_budget);
        if (decoded.empty() || decoded.size() > requested)
            fail(ErrorCode::io_error, index->path, "index transfer returned an incorrect length");
        std::size_t transient = sizeof(decoded) + decoded.capacity() * sizeof(std::string);
        for (const auto& value : decoded) {
            if (value.capacity() == std::numeric_limits<std::size_t>::max() || transient > std::numeric_limits<std::size_t>::max() - (value.capacity() + 1))
                fail(ErrorCode::invalid_argument, index->path, "decoded index staging size overflows");
            transient += value.capacity() + 1;
        }
        for (auto& value : decoded) {
            if(session.options.validation==Validation::full)validate_utf8_string(value,index->path);
            const auto incoming = value.capacity() + 1;
            if (!unique.add(std::move(value), transient)) reject_nonunique(index, full_physical_axis);
            transient -= incoming;
        }
        i += decoded.size();
    }
    if (!unique.finish()) reject_nonunique(index, full_physical_axis);
    if (full_physical_axis) index->revision->unique_index = true;
}

void require_unique_categories(const std::shared_ptr<detail::BackedNode>& categories, std::size_t scratch_bytes) {
    check_node(categories);
    auto& session = *categories->session;
    const auto info = session.backend->inspect(categories->path);
    if (encoding(info, categories->path) != "array" || info.group || info.strings)
        fail(ErrorCode::unsupported_encoding, categories->path,
             "numeric category dictionary must be a plain numeric array");
    if (info.shape.size() != 1 || categories->axes.size() != 1)
        fail(ErrorCode::shape_mismatch, categories->path,
             "category dictionary must be one-dimensional");

    const auto budget = scratch_bytes ? std::min(scratch_bytes,session.options.scratch_bytes) : session.options.scratch_bytes;
    const auto width = dtype_size(info.dtype);
    if (!width) fail(ErrorCode::unsupported_dtype, categories->path, "category dictionary has an invalid numeric dtype");
    const auto& axis = categories->axes.front();
    UniqueAccumulator unique(categories->path, budget, axis.size);
    const auto max_rows = std::max<std::size_t>(1, std::min<std::size_t>(8192,
        budget / std::max<std::size_t>(1, 8 * width)));
    std::vector<std::byte> buffer;
    for (std::size_t i = 0; i < axis.size;) {
        const auto first = axis.at(i);
        std::size_t run = 1;
        while (run < max_rows && run < axis.size - i) {
            if (first > std::numeric_limits<std::size_t>::max() - run ||
                axis.at(i + run) != first + run) break;
            ++run;
        }
        if (run > std::numeric_limits<std::size_t>::max() / width)
            fail(ErrorCode::shape_mismatch, categories->path, "category read exceeds addressable range");
        buffer.resize(run * width);
        session.backend->read_numeric(categories->path, {first}, {run}, buffer.data());
        for (std::size_t j = 0; j < run; ++j) {
            auto key = numeric_key(buffer.data() + j * width, info.dtype, categories->path);
            if (!unique.add(std::move(key), sizeof(std::string) + buffer.capacity()))
                fail(ErrorCode::invalid_argument, categories->path,
                     "category dictionaries must contain unique values");
        }
        i += run;
    }
    if (!unique.finish())
        fail(ErrorCode::invalid_argument, categories->path,
             "category dictionaries must contain unique values");
}

namespace {
class IndexCursor {
public:
    explicit IndexCursor(const BackedTable& table,std::size_t budget):budget_(budget){
        const auto& t=detail::BackedAccess::get(table);check_node(t);
        auto info=t->session->backend->inspect(t->path);
        index_=node(t->session,child(t->path,string_attr(info,"_index",t->path)),t->axes);
        if(index_->axes.size()!=1)fail(ErrorCode::shape_mismatch,t->path,"table index must be one-dimensional");
    }
    const std::string& at(std::size_t i){
        if(i>=index_->axes[0].size)fail(ErrorCode::out_of_bounds,index_->path,"table row is out of bounds");
        if(i<first_||i-first_>=values_.size()){
            const auto& axis=index_->axes[0];auto cap=std::max<std::size_t>(1,std::min<std::size_t>(8192,budget_/(2*sizeof(std::string))));
            std::size_t requested=1;
            while(requested<cap && requested<axis.size-i && axis.at(i+requested)==axis.at(i)+requested)++requested;
            values_.clear();values_.shrink_to_fit();
            values_=index_->session->backend->read_strings_bounded(index_->path,{axis.at(i)},{requested},budget_);
            if(values_.empty()||values_.size()>requested)fail(ErrorCode::io_error,index_->path,"index transfer returned an incorrect length");
            first_=i;
        }
        return values_[i-first_];
    }
private:
    std::shared_ptr<detail::BackedNode> index_;
    std::size_t budget_=0,first_=0;
    std::vector<std::string> values_;
};
}
void compare_table_indexes(const BackedTable& left,const BackedTable& right){
    const auto size=left.size();if(size!=right.size())fail(ErrorCode::shape_mismatch,"dataframe","aligned table length differs from its axis");
    const auto& a=detail::BackedAccess::get(left);const auto& b=detail::BackedAccess::get(right);
    auto budget=std::min(a->session->options.scratch_bytes,b->session->options.scratch_bytes)/2;
    IndexCursor first(left,budget),second(right,budget);
    for(std::size_t i=0;i<size;++i)if(first.at(i)!=second.at(i))fail(ErrorCode::shape_mismatch,a->path,"dataframe index differs from its aligned axis");
}
void compare_table_indexes(const BackedTable& left,const TableView& right){
    if(left.size()!=right.size())fail(ErrorCode::shape_mismatch,"dataframe","aligned table length differs from its axis");
    const auto& a=detail::BackedAccess::get(left);IndexCursor cursor(left,a->session->options.scratch_bytes);
    for(std::size_t i=0;i<right.size();++i)if(cursor.at(i)!=right.name(i))fail(ErrorCode::shape_mismatch,a->path,"dataframe index differs from its aligned axis");
}

} // namespace anndatapp::backed_detail
