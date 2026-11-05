#!/usr/bin/env python3
"""Inventory first-party source, include sites, and type declarations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


SOURCE_DIRS = ("src", "tests", "benchmarks", "examples")
EXTENSIONS = {".c", ".cc", ".cpp", ".cppm", ".cxx", ".h", ".hh", ".hpp", ".hxx"}

CPP_HEADERS = {
    "algorithm", "array", "atomic", "barrier", "bit", "cassert", "cerrno", "charconv",
    "chrono", "ctime", "cmath", "condition_variable", "cstddef", "cstdint", "cstdio", "cstdlib",
    "cstring", "deque", "exception", "filesystem", "fstream", "functional", "future",
    "iomanip", "initializer_list", "iostream", "iterator", "limits", "map", "memory", "mutex",
    "new", "numeric", "optional", "set", "span", "stdexcept", "string", "string_view",
    "system_error", "thread", "tuple", "type_traits", "unordered_map", "unordered_set", "utility", "variant",
    "vector",
}
C_HEADERS = {
    "assert.h", "complex.h", "ctype.h", "errno.h", "fenv.h", "float.h", "inttypes.h",
    "iso646.h", "limits.h", "locale.h", "math.h", "setjmp.h", "signal.h", "stdalign.h",
    "stdarg.h", "stdatomic.h", "stdbool.h", "stddef.h", "stdint.h", "stdio.h", "stdlib.h",
    "stdnoreturn.h", "string.h", "tgmath.h", "threads.h", "time.h", "uchar.h", "wchar.h",
    "wctype.h",
}
BACKEND_HEADERS = {"hdf5.h", "blosc.h", "zlib.h", "zstd.h"}
EXTERNAL_HEADERS = {"nlohmann/json.hpp", "openssl/evp.h"}
PLATFORM_PREFIXES = ("sys/", "linux/", "asm/", "mach/", "net/", "arpa/")
PLATFORM_HEADERS = {
    "dlfcn.h", "fcntl.h", "unistd.h", "pthread.h", "windows.h", "io.h", "sys/types.h",
    "sys/stat.h", "sys/mman.h", "sys/resource.h", "sys/ioctl.h", "sys/syscall.h",
}

# Representative declarations provided by headers commonly used in this tree.
# This deliberately stays finite: unmatched identifiers remain unresolved.
STD_SYMBOLS = {
    "algorithm": r"std::(?:sort|stable_sort|find|find_if|find_if_not|remove|remove_if|transform|min|max|copy|copy_n|copy_backward|lower_bound|upper_bound|all_of|any_of|fill|fill_n|clamp|equal|unique|lexicographical_compare|max_element|min_element|is_sorted)\b",
    "array": r"std::array\b",
    "atomic": r"std::(?:atomic(?:_flag)?|memory_order_[a-z_]+)\b",
    "barrier": r"std::barrier\b",
    "bit": r"std::(?:bit_cast|endian)\b",
    "cerrno": r"\berrno\b",
    "charconv": r"std::(?:to_chars|from_chars|to_chars_result|from_chars_result)\b",
    "chrono": r"std::chrono\b",
    "ctime": r"std::(?:tm|time_t|clock_t|time|clock|difftime|mktime|asctime|ctime|gmtime|localtime|strftime)\b",
    "cmath": r"std::(?:isfinite|isnan|fabs|sqrt|pow|ceil|floor|round|trunc|fmod|copysign|signbit|abs|isinf)\b",
    "condition_variable": r"std::(?:condition_variable|condition_variable_any)\b",
    "cstddef": r"std::(?:size_t|ptrdiff_t|byte|max_align_t)\b",
    "cstdint": r"std::(?:u?int(?:8|16|32|64|_fast8|_fast16|_fast32|_fast64|_least8|_least16|_least32|_least64|max|ptr|_ptr)_t|uintptr_t|intptr_t)\b",
    "cstdio": r"std::(?:FILE|printf|fprintf|snprintf|fopen|fclose|fread|fwrite|rename|remove|tmpfile|tmpnam)\b",
    "cstdlib": r"std::(?:abort|malloc|free|realloc|getenv|strtol|strtoul|strtod|exit|quick_exit|system|rand|srand|abs)\b",
    "cstring": r"std::(?:memcpy|memset|memcmp|memmove|strlen|strcmp|strchr|strerror)\b",
    "deque": r"std::deque\b",
    "exception": r"std::(?:exception|terminate|uncaught_exceptions)\b",
    "filesystem": r"std::filesystem\b",
    "fstream": r"std::(?:fstream|ifstream|ofstream)\b",
    "functional": r"std::(?:function|less|invoke|hash|reference_wrapper)\b",
    "future": r"std::(?:future|promise|async|packaged_task)\b",
    "iomanip": r"std::(?:setprecision|setw|setfill|fixed|hex|dec|scientific)\b",
    "initializer_list": r"std::initializer_list\b",
    "iostream": r"std::(?:cout|cerr|clog|cin|ios_base|ostream|istream)\b",
    "iterator": r"std::(?:istreambuf_iterator|ostreambuf_iterator|distance|next|prev|back_inserter|front_inserter|begin|end)\b",
    "limits": r"std::numeric_limits\b",
    "map": r"std::(?:map|multimap)\b",
    "memory": r"std::(?:shared_ptr|unique_ptr|weak_ptr|make_shared|make_unique|allocator|addressof|construct_at|destroy_at|align)\b",
    "mutex": r"std::(?:mutex|recursive_mutex|lock_guard|unique_lock|scoped_lock|shared_mutex)\b",
    "new": r"std::(?:nothrow|nothrow_t|bad_alloc|align_val_t|new_handler|set_new_handler|get_new_handler|launder)\b|\bnew\s*\(",
    "numeric": r"std::(?:iota|accumulate|inner_product|reduce|inclusive_scan|exclusive_scan|gcd|lcm)\b",
    "optional": r"std::(?:optional|nullopt)\b",
    "set": r"std::(?:set|multiset)\b",
    "span": r"std::span\b",
    "stdexcept": r"std::(?:logic_error|runtime_error|invalid_argument|out_of_range|length_error|overflow_error|domain_error|range_error|underflow_error)\b",
    "system_error": r"std::(?:errc|error_code|error_condition|system_error)\b",
    "string": r"std::(?:basic_string|string)\b",
    "string_view": r"std::(?:basic_string_view|string_view)\b",
    "thread": r"std::(?:thread|jthread|this_thread)\b",
    "tuple": r"std::(?:tuple|make_tuple)\b",
    "type_traits": r"std::(?:is_same|is_integral|is_unsigned|is_signed|is_const|is_volatile|is_convertible|is_trivially_copyable|is_floating_point|remove_cv|remove_const|remove_volatile|conditional_t|enable_if_t|common_type_t|underlying_type_t|make_unsigned_t|decay_t)(?:_v|_t)?\b",
    "unordered_map": r"std::unordered_map\b",
    "unordered_set": r"std::unordered_set\b",
    "utility": r"std::(?:move|forward|pair|swap|exchange|move_if_noexcept|as_const)\b",
    "variant": r"std::(?:variant|visit|get|get_if|holds_alternative|monostate|bad_variant_access|in_place_index|in_place_type)\b",
    "vector": r"std::vector\b",
}

C_SYMBOLS = {
    "assert.h": r"\bassert\s*\(", "errno.h": r"\berrno\b", "stddef.h": r"\b(?:size_t|ptrdiff_t|offsetof)\b",
    "stdint.h": r"\b(?:u?int(?:8|16|32|64)_t|INT(?:8|16|32|64)_MAX|UINT(?:8|16|32|64)_MAX)\b",
    "limits.h": r"\b(?:CHAR_BIT|CHAR_MAX|CHAR_MIN|INT_MAX|INT_MIN|LONG_MAX|LONG_MIN|SHRT_MAX|SHRT_MIN|UCHAR_MAX|UINT_MAX|ULONG_MAX|USHRT_MAX)\b",
    "inttypes.h": r"\b(?:PRI|SCN)[a-zA-Z0-9_]+\b",
    "stdarg.h": r"\b(?:va_list|va_start|va_arg|va_copy|va_end)\b",
    "stdatomic.h": r"\b(?:atomic_[A-Za-z0-9_]+|memory_order_[A-Za-z0-9_]+)\b",
    "stdio.h": r"\b(?:FILE|printf|fprintf|snprintf|fopen|fclose|fread|fwrite|rename|remove)\b",
    "stdlib.h": r"\b(?:abort|malloc|free|realloc|getenv|strtol|strtoul|strtod|exit|system|rand|srand)\b",
    "string.h": r"\b(?:memcpy|memset|memcmp|memmove|strlen|strcmp|strchr|strerror)\b",
    "math.h": r"\b(?:isfinite|isnan|fabs|sqrt|pow|ceil|floor|round|trunc|fmod|copysign|signbit)\b",
    "time.h": r"\b(?:time|time_t|timespec|clock_gettime|nanosleep)\b",
}

SYSTEM_SYMBOLS = {
    "dlfcn.h": r"\b(?:dlopen|dlsym|dlclose|dlerror)\b",
    "fcntl.h": r"\b(?:fcntl|open|O_[A-Z0-9_]+|FD_CLOEXEC)\b",
    "unistd.h": r"\b(?:read|write|close|unlink|ftruncate|fsync|lseek|syscall|getpid|access|pipe|pread|pwrite)\b",
    "sys/types.h": r"\b(?:off_t|ssize_t|pid_t|mode_t)\b",
    "sys/stat.h": r"\b(?:stat|fstat|S_IS[A-Z]+|struct stat)\b",
    "sys/mman.h": r"\b(?:mmap|munmap|mprotect|PROT_[A-Z]+|MAP_[A-Z_]+|msync)\b",
    "sys/resource.h": r"\b(?:getrusage|rusage|RUSAGE_[A-Z_]+)\b",
    "sys/ioctl.h": r"\bioctl\b",
    "sys/syscall.h": r"\b(?:SYS_[A-Za-z0-9_]+|__NR_[A-Za-z0-9_]+)\b",
    "linux/perf_event.h": r"\b(?:perf_event_attr|PERF_[A-Z0-9_]+|PERF_COUNT_[A-Z0-9_]+)\b",
    "pthread.h": r"\bpthread_[A-Za-z0-9_]+\b",
    "windows.h": r"\b(?:HANDLE|DWORD|CreateFile|CloseHandle|GetLastError)\b",
}

BACKEND_SYMBOLS = {
    "hdf5.h": r"\bH5[A-Z][A-Za-z0-9_]*\b",
    "blosc.h": r"\bblosc_[A-Za-z0-9_]+\b",
    "zlib.h": r"\b(?:deflate|inflate|compress|uncompress|z_stream|Z_STREAM_END|Z_OK|Z_[A-Z0-9_]+)\b",
    "zstd.h": r"\bZSTD_[A-Za-z0-9_]+\b",
    "nlohmann/json.hpp": r"\bnlohmann::json\b",
    "openssl/evp.h": r"\bEVP_[A-Za-z0-9_]+\b",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mask_comments_and_literals(text: str) -> str:
    """Replace comments and quoted literals with spaces while keeping lines."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if text[k] != "\n":
                    out[k] = " "
            i = j
            continue
        raw = re.match(r'R"([^ ()\\\t\r\n]{0,16})\(', text[i:])
        if raw:
            end = ")" + raw.group(1) + '"'
            j = text.find(end, i + raw.end())
            j = n if j < 0 else j + len(end)
            for k in range(i, j):
                if text[k] != "\n":
                    out[k] = " "
            i = j
            continue
        if text[i] in "\"'":
            quote, j = text[i], i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                elif text[j] == quote:
                    j += 1
                    break
                else:
                    j += 1
            for k in range(i, min(j, n)):
                if text[k] != "\n":
                    out[k] = " "
            i = j
            continue
        i += 1
    return "".join(out)


def mask_include_directives(code: str) -> str:
    """Keep line positions while preventing header names becoming evidence."""
    out = []
    for line in code.splitlines(keepends=True):
        if re.match(r"\s*#\s*include\b", line):
            out.append("\n" if line.endswith("\n") else "")
        else:
            out.append(line)
    return "".join(out)


def mask_preprocessor_directives(code: str) -> str:
    """Blank directive lines, including backslash-continued macro bodies."""
    out, continued = [], False
    for line in code.splitlines(keepends=True):
        directive = continued or bool(re.match(r"\s*#", line))
        if directive:
            out.append("\n" if line.endswith("\n") else "")
            continued = line.rstrip("\r\n").endswith("\\")
        else:
            out.append(line)
    return "".join(out)


def source_files(root: Path) -> list[Path]:
    return sorted(
        p for d in SOURCE_DIRS if (root / d).is_dir()
        for p in (root / d).rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def role_for(path: str) -> str:
    if path.startswith("src/"):
        if path.endswith(".cppm"):
            return "production_module_source"
        return "production_source_or_header"
    if path.startswith("tests/"):
        return "test_support"
    if path.startswith("benchmarks/"):
        return "benchmark_support"
    return "example"


def language_for(path: Path) -> str:
    return "c" if path.suffix.lower() == ".c" else "c++" if path.suffix.lower() in {".cc", ".cpp", ".cppm", ".cxx"} else "header"


def classify_header(header: str, delimiter: str, root: Path, includer: Path) -> tuple[str, str | None]:
    if delimiter == '"':
        found = (includer.parent / header).resolve()
        if not found.is_file():
            found = (root / header).resolve()
        return "local", found.relative_to(root.resolve()).as_posix() if found.is_file() and found.is_relative_to(root.resolve()) else None
    base = Path(header).name
    if header in CPP_HEADERS:
        return "standard", None
    if header in C_HEADERS:
        return "standard_c", None
    if header in BACKEND_HEADERS:
        return "backend", None
    if header in EXTERNAL_HEADERS:
        return "external", None
    if header.startswith(PLATFORM_PREFIXES) or header in PLATFORM_HEADERS:
        return "platform", None
    if base in CPP_HEADERS:
        return "standard", None
    if base in C_HEADERS:
        return "standard_c", None
    return "investigate", None


def symbol_evidence(category: str, header: str, code: str) -> list[dict[str, object]]:
    patterns: list[tuple[str, str]] = []
    if category == "standard" and header in STD_SYMBOLS:
        patterns.append((STD_SYMBOLS[header], f"std::<{header}-declaration>"))
    elif category == "standard_c" and Path(header).name in C_SYMBOLS:
        patterns.append((C_SYMBOLS[Path(header).name], f"<{Path(header).name}-declaration>"))
    elif category == "platform" and header in SYSTEM_SYMBOLS:
        patterns.append((SYSTEM_SYMBOLS[header], f"<{header}-declaration>"))
    elif category == "backend" and header in BACKEND_SYMBOLS:
        patterns.append((BACKEND_SYMBOLS[header], f"<{header}-declaration>"))
    elif category == "external" and header in BACKEND_SYMBOLS:
        patterns.append((BACKEND_SYMBOLS[header], f"<{header}-declaration>"))
    out = []
    for pattern, label in patterns:
        for match in list(re.finditer(pattern, code))[:6]:
            line = code.count("\n", 0, match.start()) + 1
            out.append({"symbol": match.group(0), "line": line, "provider": label})
    return out


def expected_header_for_symbol(symbol: str, language: str) -> str | None:
    if language == "c":
        token = symbol.split("::")[-1]
        for header, pattern in C_SYMBOLS.items():
            if re.search(pattern, token):
                return header
        return None
    name = symbol.removeprefix("std::")
    matches = []
    for header, pattern in STD_SYMBOLS.items():
        if re.fullmatch(pattern, symbol):
            matches.append(header)
    if len(matches) == 1:
        return matches[0]
    # A small explicit supplement handles identifiers whose representative
    # patterns also contain punctuation or broad alternatives.
    owners = {
        "byte": "cstddef", "size_t": "cstddef", "ptrdiff_t": "cstddef",
        "int8_t": "cstdint", "uint8_t": "cstdint", "int16_t": "cstdint", "uint16_t": "cstdint",
        "int32_t": "cstdint", "uint32_t": "cstdint", "int64_t": "cstdint", "uint64_t": "cstdint",
        "function": "functional", "less": "functional",
    }
    if name.startswith(("int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t")):
        return "cstdint"
    if name.startswith(("is_", "remove_", "add_", "conditional", "enable_if", "common_type", "underlying_type", "make_unsigned", "decay")):
        return "type_traits"
    return owners.get(name)


def declarations(path: str, text: str, code: str) -> list[dict[str, object]]:
    rows = []
    raw_lines = text.splitlines()
    code = mask_preprocessor_directives(code)

    def add(kind: str, name: str, start: int, form: str) -> None:
        line = code.count("\n", 0, start) + 1
        original = raw_lines[line - 1].strip() if line <= len(raw_lines) else ""
        rows.append({
            "source": path,
            "line": line,
            "kind": kind,
            "name": name,
            "form": form,
            "text": original[:220],
            "evidence": "syntax-pattern; forward declarations retained; not semantic proof",
        })

    type_re = re.compile(r"\b(?:enum\s+(?:class\s+|struct\s+)?|class\s+|struct\s+|union\s+)([A-Za-z_]\w*)\b")
    for match in type_re.finditer(code):
        prefix = code[code.rfind("\n", 0, match.start()) + 1:match.start()]
        if re.search(r"\bfriend\s*$", prefix):
            continue
        template = code.rfind("template", 0, match.start())
        close = code.find(">", template) if template >= 0 else -1
        if template >= 0 and close > match.start():
            continue
        tail = code[match.end():]
        next_token = re.match(r"\s*(final\b|\{|;|:)", tail)
        if not next_token:
            continue
        form = "forward_or_elaborated"
        token = next_token.group(1)
        if token == "{":
            form = "definition"
        elif token in {"final", ":"}:
            brace = tail.find("{")
            semi = tail.find(";")
            if brace >= 0 and (semi < 0 or brace < semi):
                form = "definition"
            elif semi < 0:
                continue
        keyword = match.group(0).split()[0]
        add("enum" if keyword == "enum" else keyword, match.group(1), match.start(), form)

    for match in re.finditer(r"\busing\s+([A-Za-z_]\w*)\s*=", code):
        add("alias", match.group(1), match.start(), "using")

    for match in re.finditer(r"\btypedef\b", code):
        start = match.start()
        depth = [0, 0, 0]
        end = None
        for i in range(start + len("typedef"), len(code)):
            c = code[i]
            if c == "(": depth[0] += 1
            elif c == ")": depth[0] -= 1
            elif c == "[": depth[1] += 1
            elif c == "]": depth[1] -= 1
            elif c == "{": depth[2] += 1
            elif c == "}": depth[2] -= 1
            elif c == ";" and not any(depth):
                end = i + 1
                break
        if end is None:
            continue
        statement = code[start:end]
        ptr = re.search(r"\(\s*\*\s*([A-Za-z_]\w*)", statement)
        names = re.findall(r"[A-Za-z_]\w*", statement[:statement.rfind(";")])
        name = ptr.group(1) if ptr else (names[-1] if names else "")
        if name and name != "typedef":
            add("typedef", name, start, "typedef")
    return sorted(rows, key=lambda row: (int(row["line"]), str(row["kind"]), str(row["name"])))


def module_exposure(path: str, code: str, symbol_lines: list[int]) -> str:
    if not path.endswith(".cppm"):
        return "implementation"
    if not re.search(r"(?m)^\s*export\s+module\b", code):
        return "implementation_partition_or_unit"
    if not symbol_lines:
        return "module_interface_symbol_use_not_found"
    lines = code.splitlines()
    export_depth = 0
    active = False
    public_lines: set[int] = set()
    for n, line in enumerate(lines, 1):
        if re.search(r"\bexport\s+namespace\b|^\s*export\s+(?:class|struct|enum|using|template)\b", line):
            active = True
        if active:
            public_lines.add(n)
        export_depth += line.count("{") - line.count("}") if active else 0
        if active and export_depth <= 0 and "{" in line:
            active = False
            export_depth = 0
    return "public_api_surface" if any(n in public_lines for n in symbol_lines) else "module_interface_nonexported_or_private_use"


def normalize_source_path(value: str, source_root: Path) -> str | None:
    value = value.replace("\\", "/")
    markers = ("build/thin-review-20261105/source/", "thin-review-20261105/source/", "source/")
    for marker in markers:
        if marker in value:
            value = value.rsplit(marker, 1)[1]
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(source_root.resolve()).as_posix()
        except ValueError:
            return None
    return candidate.as_posix().removeprefix("./")


def validate_ledger(path: Path, files: dict[str, dict[str, object]], source_root: Path,
                    include_ids: set[str]) -> dict[str, object]:
    data = json.loads(path.read_text())
    found, bad = [], []
    checked_sites = []
    path_keys = {"source", "source_path", "file", "file_path", "path"}
    line_keys = {"line", "line_start", "source_line"}

    def walk(value: object, where: str = "$", inherited_source: str | None = None) -> None:
        if isinstance(value, dict):
            explicit_source = next((value[k] for k in path_keys if isinstance(value.get(k), str)), None)
            own = explicit_source or inherited_source
            line = next((value[k] for k in line_keys if isinstance(value.get(k), int)), None)
            if own and (explicit_source is not None or line is not None or value.get("source_sha256")):
                normalized = normalize_source_path(own, source_root)
                if normalized and normalized in files:
                    entry = {"source": normalized, "line": line, "json_path": where}
                    if line is not None and (line < 1 or line > int(files[normalized]["line_count"])):
                        bad.append({**entry, "reason": f"line must be between 1 and {files[normalized]['line_count']}"})
                    elif value.get("source_sha256") and value["source_sha256"] != files[normalized]["sha256"]:
                        bad.append({**entry, "reason": "source_sha256 differs from scanned source"})
                    else:
                        found.append(entry)
                elif any(part in own.replace("\\", "/") for part in SOURCE_DIRS):
                    bad.append({"source": own, "line": line, "json_path": where, "reason": "source path is not in scanned tree"})
            site_id = value.get("site_id")
            if isinstance(site_id, str) and "#L" in site_id:
                source_part, line_part = site_id.rsplit("#L", 1)
                normalized = normalize_source_path(source_part, source_root)
                canonical = f"{normalized}#L{line_part}" if normalized else site_id
                if canonical in include_ids:
                    checked_sites.append({"site_id": canonical, "json_path": where})
                else:
                    bad.append({"site_id": site_id, "json_path": where, "reason": "include site id is not present in scanned source"})
            site_list = value.get("include_site_ids")
            if isinstance(site_list, list):
                for item in site_list:
                    if not isinstance(item, str):
                        bad.append({"json_path": where, "reason": "include_site_ids entries must be strings"})
                        continue
                    source_part, marker, line_part = item.rpartition("#L")
                    normalized = normalize_source_path(source_part, source_root) if marker else None
                    canonical = f"{normalized}#L{line_part}" if normalized else item
                    if canonical in include_ids:
                        checked_sites.append({"site_id": canonical, "json_path": where})
                    else:
                        bad.append({"site_id": item, "json_path": where, "reason": "include site id is not present in scanned source"})
            for k, child in value.items():
                walk(child, f"{where}.{k}", own)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, f"{where}[{i}]", inherited_source)

    walk(data)
    return {"ledger": str(path), "source_links": found, "include_site_links": checked_sites,
            "invalid_links": bad, "status": "ok" if not bad else "invalid"}


def inventory(source: Path) -> dict[str, object]:
    root = source.resolve()
    paths = source_files(root)
    file_rows, include_rows, type_rows, missing_rows, unused_rows = [], [], [], [], []
    totals: dict[str, int] = {}
    include_re = re.compile(r"(?m)^[ \t]*#[ \t]*include[ \t]*([<\"])([^>\"]+)[>\"]")
    macro_re = re.compile(r"(?m)^[ \t]*#[ \t]*include[ \t]+([^<\" \t][^\r\n]*)")
    for full in paths:
        rel = full.relative_to(root).as_posix()
        raw_bytes = full.read_bytes()
        text = raw_bytes.decode("utf-8", errors="replace")
        code = mask_comments_and_literals(text)
        evidence_code = mask_include_directives(code)
        language = language_for(full)
        file_includes = list(include_re.finditer(text))
        file_macro_includes = list(macro_re.finditer(text))
        source_lines = text.splitlines()
        file_rows.append({
            "path": rel,
            "language": language,
            "role": role_for(rel),
            "bytes": len(raw_bytes),
            "line_count": len(text.splitlines()),
            "sha256": sha256(raw_bytes),
            "include_sites": len(file_includes),
            "nonliteral_include_directives": len(file_macro_includes),
        })
        headers = set()
        for match in file_includes:
            delimiter, header = match.group(1), match.group(2).strip()
            line = text.count("\n", 0, match.start()) + 1
            if not source_lines[line - 1].lstrip().startswith(match.group(0).strip()):
                raise ValueError(f"include line mismatch at {rel}:{line}")
            category, local_target = classify_header(header, delimiter, root, full)
            evidence = symbol_evidence(category, header, evidence_code)
            symbol_lines = [int(item["line"]) for item in evidence]
            exposure = module_exposure(rel, evidence_code, symbol_lines)
            posix_time = list(re.finditer(r"\b(?:clock_gettime|timespec|CLOCK_[A-Z0-9_]+)\b", evidence_code))
            if category in {"standard", "standard_c", "platform", "backend", "external"}:
                disposition = "retain" if evidence else "remove_candidate" if category in {"standard", "standard_c"} else "investigate"
                rationale = "representative provided symbol appears in source" if evidence else "no mapped direct symbol found; lexical evidence cannot rule out macros, transitive needs, or configuration"
                if header in {"ctime", "time.h"} and posix_time and not evidence:
                    disposition = "investigate"
                    rationale = "POSIX time symbols occur, but the finite standard-symbol map found no ISO time declaration for this include"
            elif category == "local":
                disposition = "retain" if local_target else "investigate"
                rationale = "quoted first-party dependency resolves in frozen source tree" if local_target else "quoted target was not resolved by source-relative lookup"
            else:
                disposition = "investigate"
                rationale = "header class or providing declarations are not in the finite evidence map"
            site = {
                "site_id": f"{rel}#L{line}",
                "source": rel,
                "source_sha256": sha256(raw_bytes),
                "line": line,
                "directive": match.group(0).strip(),
                "delimiter": "angle" if delimiter == "<" else "quoted",
                "header": header,
                "classification": category,
                "local_target": local_target,
                "role": role_for(rel),
                "exposure": exposure,
                "symbol_evidence": evidence,
                "directness": "direct-symbol-evidence" if evidence else "resolved-local-file" if category == "local" and local_target else "unresolved-from-lexical-scan",
                "dependency_status": "direct_symbol_observed" if evidence else "local_include_resolved" if category == "local" and local_target else "transitive_or_unmapped_possible",
                "disposition": disposition,
                "rationale": rationale,
                "runtime_cost": "no direct runtime cost from the include directive itself",
                "build_cost": "may add parse, template, BMI, and rebuild dependencies; not measured",
            }
            include_rows.append(site)
            totals[category] = totals.get(category, 0) + 1
            if not evidence and category in {"standard", "standard_c"} and disposition == "remove_candidate":
                unused_rows.append({"site_id": site["site_id"], "source": rel, "line": line, "header": header, "candidate": disposition})

        # A lexical reference to a known standard symbol without its mapped
        # direct header is a candidate only; imports/transitive declarations
        # can satisfy the frozen translation unit today.
        imported = {m.group(2).strip() for m in file_includes}
        symbols = sorted(set(re.findall(r"\bstd::[A-Za-z_]\w*", evidence_code)))
        for symbol in symbols:
            owner = expected_header_for_symbol(symbol, language)
            if owner and owner not in imported:
                line = evidence_code.count("\n", 0, evidence_code.find(symbol)) + 1
                missing_rows.append({
                    "source": rel,
                    "source_sha256": sha256(raw_bytes),
                    "line": line,
                    "symbol": symbol,
                    "expected_header": owner,
                    "candidate": "add_direct",
                    "rationale": "known std symbol has no matching literal direct include; transitive/module visibility remains possible",
                })
        if re.search(r"\b(?:clock_gettime|timespec|CLOCK_[A-Z0-9_]+)\b", evidence_code) and "time.h" not in imported:
            match = re.search(r"\b(?:clock_gettime|timespec|CLOCK_[A-Z0-9_]+)\b", evidence_code)
            line = evidence_code.count("\n", 0, match.start()) + 1
            missing_rows.append({
                "source": rel,
                "source_sha256": sha256(raw_bytes),
                "line": line,
                "symbol": match.group(0),
                "expected_header": "time.h",
                "candidate": "add_direct",
                "rationale": "POSIX clock/time extension is used; C++ <ctime> does not guarantee clock_gettime, timespec, or CLOCK_* declarations",
            })
        type_rows.extend(declarations(rel, text, code))
    prod = [row for row in file_rows if str(row["path"]).startswith("src/")]
    support = [row for row in file_rows if not str(row["path"]).startswith("src/")]
    prod_classes: dict[str, int] = {}
    support_classes: dict[str, int] = {}
    for row in include_rows:
        key = str(row["classification"])
        counts = prod_classes if str(row["source"]).startswith("src/") else support_classes
        counts[key] = counts.get(key, 0) + 1
    return {
        "schema": "anndatapp-review-inventory-v1",
        "source_root": str(root),
        "scope": {"directories": list(SOURCE_DIRS), "extensions": sorted(EXTENSIONS), "preprocessor_conditionals": "all literal directives counted independent of active build configuration"},
        "summary": {
            "files": len(file_rows),
            "c_source_files": sum(row["language"] == "c" for row in file_rows),
            "cpp_source_files": sum(row["language"] == "c++" for row in file_rows),
            "headers": sum(row["language"] == "header" for row in file_rows),
            "production_files": len(prod),
            "production_module_sources": sum(row["language"] == "c++" and str(row["path"]).endswith(".cppm") for row in prod),
            "production_cpp_implementations": sum(row["language"] == "c++" and not str(row["path"]).endswith(".cppm") for row in prod),
            "production_headers": sum(row["language"] == "header" for row in prod),
            "support_files": len(support),
            "include_sites": len(include_rows),
            "nonliteral_include_directives": sum(int(row["nonliteral_include_directives"]) for row in file_rows),
            "include_sites_by_classification": dict(sorted(totals.items())),
            "production_include_sites": sum(row["source"].startswith("src/") for row in include_rows),
            "support_include_sites": sum(not row["source"].startswith("src/") for row in include_rows),
            "production_include_sites_by_classification": dict(sorted(prod_classes.items())),
            "support_include_sites_by_classification": dict(sorted(support_classes.items())),
            "declaration_rows": len(type_rows),
            "missing_direct_header_candidates": len(missing_rows),
            "unused_include_candidates": len(unused_rows),
        },
        "files": file_rows,
        "include_sites": include_rows,
        "missing_direct_header_candidates": missing_rows,
        "unused_include_candidates": unused_rows,
        "type_declarations": type_rows,
        "limitations": [
            "Include-site rows count source directives, not configuration-active preprocessing events.",
            "Symbol evidence uses a finite lexical map after comments and string literals are blanked; it is not compiler dependency proof.",
            "remove_candidate and add_direct are review candidates, not validated edits. Compilation against isolated copies is required to establish sufficiency.",
            "Declaration rows are syntax-pattern matches, not a complete C++ AST or semantic API inventory.",
            "No runtime or build timing was measured; include directives have no direct runtime cost, while parse and rebuild effects remain unmeasured.",
        ],
    }


def markdown_report(data: dict[str, object]) -> str:
    summary = data["summary"]
    assert isinstance(summary, dict)
    lines = [
        "# Include-site inventory",
        "",
        "This report inventories the frozen source tree by literal include occurrence. Each row carries source hash, line, class, representative symbol evidence, exposure, and a conservative disposition. `remove_candidate` means the finite lexical map found no mapped symbol; it is not proof that removing the include compiles.",
        "",
        f"Source root: `{data['source_root']}`  ",
        f"Source files: {summary['files']} ({summary['cpp_source_files']} C++ sources, {summary['c_source_files']} C sources, {summary['headers']} headers)  ",
        f"Include sites: {summary['include_sites']} total; {summary['production_include_sites']} under `src/`, {summary['support_include_sites']} under tests/benchmarks/examples  ",
        f"Declaration/type syntax rows: {summary['declaration_rows']}  ",
        f"Missing direct-header candidates: {summary['missing_direct_header_candidates']}; unused-include candidates: {summary['unused_include_candidates']}",
        "",
        "## Cost and evidence limits",
        "",
        "The include directive itself does not add runtime work. Header parsing, template instantiation, module BMI construction, and dependency-triggered rebuild cost can change; no cost was measured here. The classifier records direct symbol matches where its finite map recognizes the provider. Unmapped names, macro effects, transitive declarations, build-configuration branches, and module visibility remain uncertain. `retain` means a representative provided symbol or resolved local target was observed; it does not prove every transitive requirement is portable.",
        "",
        "## Per-occurrence assessment",
        "",
        "| Source | Line | Include | Class | Exposure | Evidence | Disposition |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in data["include_sites"]:
        assert isinstance(row, dict)
        ev = ", ".join(f"`{e['symbol']}@{e['line']}`" for e in row["symbol_evidence"]) or "—"
        lines.append(f"| `{row['source']}` | {row['line']} | `{row['directive']}` | {row['classification']} | {row['exposure']} | {ev} | {row['disposition']} |")
    lines += ["", "## Candidate direct-header gaps", "", "| Source | Line | Symbol | Expected direct header |", "|---|---:|---|---|"]
    for row in data["missing_direct_header_candidates"]:
        assert isinstance(row, dict)
        lines.append(f"| `{row['source']}` | {row['line']} | `{row['symbol']}` | `<{row['expected_header']}>` |")
    if not data["missing_direct_header_candidates"]:
        lines.append("| — | — | None identified by the finite map | — |")
    lines += ["", "## Declaration/type syntax inventory", "", "See `includes.json` → `type_declarations` for all per-file `class`, `struct`, `union`, `enum`, `using`, and `typedef` rows. These are source-linked syntax-pattern matches, not a complete semantic API inventory.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="frozen source root containing src/, tests/, benchmarks/, and examples/")
    parser.add_argument("--output", required=True, type=Path, help="inventory JSON destination")
    parser.add_argument("--markdown", type=Path, help="optional per-occurrence Markdown report destination")
    parser.add_argument("--validate-ledger", action="append", default=[], type=Path, help="validate source references in a subsystem JSON ledger (repeatable)")
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error(f"source root does not exist: {args.source}")
    data = inventory(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(data))
    if args.validate_ledger:
        files = {str(row["path"]): row for row in data["files"]}
        include_ids = {str(row["site_id"]) for row in data["include_sites"]}
        checks = [validate_ledger(p, files, args.source.resolve(), include_ids) for p in args.validate_ledger]
        print(json.dumps(checks, indent=2))
        return 1 if any(check["invalid_links"] for check in checks) else 0
    print(json.dumps(data["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
