# Standard-library and representation choices

The [include ledger](includes.md) accounts for every literal include site in the frozen production and supporting C/C++ files. Its symbol matching is lexical: it produces candidates for inspection, not a compiler-verified deletion list. The subsystem reviews assess the execution and representation behind those uses. This page records the architectural disposition of all 31 distinct production C++ standard headers.

| Header | Disposition and cost to examine |
| --- | --- |
| `algorithm` | Retain ordinary sort/search/min/max operations. Examine traversal and temporary storage at each call, not the header name. |
| `array` | Retain fixed-size local storage where it exposes bounds and avoids dynamic metadata. |
| `bit` | Retain explicit byte-order and representation operations; test actual file/native conversions. |
| `cerrno` | Retain error handling at OS boundaries, including partial I/O and interruption. |
| `cmath` | Retain the numerical predicates/calculations actually required; do not replace them with weaker assumptions. |
| `cstddef` | Retain explicit sizes, signed byte/element offsets and byte storage. Width and unit contracts matter. |
| `cstdint` | Retain fixed-width storage/index types, distinct from native C++ object identity. |
| `cstdio` | Retain bounded temporary-file mechanics where used. File lifetime and recovery must remain visible. |
| `cstdlib` | Retain required C allocation/release interoperability. Keep deleter and allocation-family matching explicit. |
| `cstring` | Retain direct byte copies and scalar decoding. Preserve overlap, alignment, object lifetime and extent preconditions. |
| `filesystem` | Retain private store/path operations; this facility does not make accompanying POSIX calls portable. |
| `fstream` | Retain ordinary file-buffer operations unless measured against an equivalent transfer. Avoid equating stream convenience with an established I/O bottleneck. |
| `functional` | Review actual erased callbacks, notably coordinate traversal, separately from header cost. Select a concrete callable once when it improves the complete implementation and measured execution. |
| `initializer_list` | Retain small caller syntax. Remove unnecessary intermediate owning vectors/strings behind that syntax where justified. |
| `limits` | Retain explicit overflow/range/capacity bounds. Moving checks out of loops requires a valid boundary proof. |
| `map` | Retain ordered metadata/session maps where their semantics earn the tree representation. Investigate repeated construction, lookups, and shared mutation rather than blanket replacement. |
| `memory` | Retain RAII and explicit retained lifetimes. Count control blocks, refcount traffic, duplicated owners and descriptor allocations at acquisition. Shared ownership supplies no payload synchronization. |
| `new` | Retain required allocation/object-lifetime operations, including record members; these are substantive C++ memory contracts. |
| `numeric` | Retain direct index generation/reduction where used. Examine work and overflow rather than introducing a custom algorithm layer. |
| `optional` | Retain meaningful absence. Investigate objects carrying many inactive optional payloads, especially recursive metadata; absence is distinct from empty data. |
| `set` | Retain required uniqueness/order behavior pending geometry-specific evidence for a smaller representation. |
| `span` | Retain the private bounded buffer descriptor where it removes pointer/length ambiguity without adding ownership. A wider public span framework is not implied. |
| `stdexcept` | Retain the existing contextual `Error` contract. Avoid exceptions as routine layout discovery where a direct branch suffices; do not silently remove diagnostic behavior. |
| `string` | Retain owned names/text. Remove unpack-and-repack staging from packed data paths if it preserves validation and reduces measured work. |
| `string_view` | Retain borrowed names/text with explicit byte lifetime; validate C-string/path boundaries before constructing or passing views to C APIs. |
| `type_traits` | Retain small compile-time type/native-layout checks. Keep specialization bounded and implementation details private. |
| `unordered_map` | Investigate sparse/gather lookup construction, per-node allocation and small-budget fallback complexity; preserve repeated selection positions. |
| `unordered_set` | Retain duplicate/membership validation where needed. Count repeated validation allocations before selecting another representation. |
| `utility` | Retain deliberate moves, aliases and forwarding. A move of an owning vector is different work from a deep copy. |
| `variant` | Retain actual tagged alternatives where appropriate, particularly private backend attributes. A variant replacement for metadata is a candidate requiring footprint, generated-code, lifetime and build evidence. |
| `vector` | Retain contiguous owned buffers and arbitrary-rank metadata where useful. Investigate per-call small vectors, copied requests, output-sized staging and duplicate schema lists. Preserve the rejected gather-buffer experiment's result. |

No production include was removed during this review. A facility can be justified in one file and unnecessary in another; the per-site ledger preserves that distinction. Compiler checks must resolve removal candidates and missing-direct-header candidates under the relevant module boundaries and backend configurations. Lexical matches can miss unqualified calls, macros, overload-provided types, or legitimate declarations reachable through imports; transitive availability on this host is not a portability proof.

The remaining production includes are private backend/JSON headers, private local headers, and POSIX headers. Keep storage dependencies behind compiled interfaces. Isolate platform operations only when doing so supplies a real target/capability boundary; a general filesystem or I/O framework would need its own justification.

The useful thin-library boundary is acquisition of a typed, lifetime-correct descriptor followed by ordinary pointer work. Wrapping every array in a new framework, removing retained views, or exporting large template implementations would add different costs. The strongest current candidates remove repeated acquisition, recoding, staging or dispatch while preserving that boundary; see the [ranked backlog](README.md).
