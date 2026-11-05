# Data API cleanup

This record resolves [data review D01–D08](review/data.md) against the frozen baseline in `build/v1-cleanup/baseline-source`. Numerical runtime and build-time effects were not measured in this change.

- **D01:** `Selection(const char*)` rejects null with `invalid_argument` at `index`. Existing string and duplicate-label behavior is unchanged. Passing a null pointer was previously undefined and must be removed from callers.
- **D02:** category pruning plans are immutable shared records. A newly selected descriptor retains its predecessor's plan for stale-code detection, reads each original code once while constructing its new plan, and then replaces the plan. A nonnull empty plan still means a pruned empty dictionary. A newly pruned result adds a plan allocation; subsequent descriptor copies share it instead of copying both vectors. Selected tables prune the new descriptor directly, rather than first allocating a second descriptor.
- **D03:** nonidentity categorical `codes()` and `copy()` materialize directly into one final array, with no intermediate `int64_t` vector and no subsequent second typed copy. The private array allocation bridge preserves the source's native C++ type, numerical dtype and serialization byte order. Identity-prefix paths continue to validate stale codes and deep-copy only when requested.
- **D04:** selected string-column copies build packed bytes and offsets directly, retaining validity, empty strings and `NA`/`NaN` metadata. Initializer-list string construction also packs directly. These operations preserve the prior trusted-input boundary: neither implicitly validates UTF-8 that was accepted with `Validation::none`; explicit `validate()` still checks it.
- **D05:** an all-row table selection returns the retained snapshot itself when pruning is disabled or there are no categorical columns. Snapshots otherwise remain eager: each captures a stable set of named bindings so later structural replacement cannot alter it, while payload edits remain visible. A shared copy-on-write binding registry would require a new mutation protocol and does not earn that complexity without workload evidence. Focused direct column acquisition remains available through `obs_column`, `var_column` and `Table::column`.
- **D06:** categorical validity uses `code`'s existing row resolution instead of resolving it twice.
- **D07:** the public declarations now describe copied name lists, possible pruning and recoding work, retained binding and borrowed string lifetimes, and caller responsibility for synchronization.
- **D08:** `ElementImpl` now stores one active scalar, owner, retained view, string or mapping in a private `std::variant`. It keeps the existing `ElementKind`, scalar dtype and byte-order fields and makes no additional allocation solely for the variant. `Table::with_index` moves names into the freshly created empty owner instead of allocating and discarding a second names owner. Public `Selection` and `ColumnImpl` representations are retained: their inactive fields make the tradeoff visible, but changing the public selection layout affects consumers and changing column storage would introduce another active-payload lifetime scheme without measured workload evidence.

The deterministic representation probe compiled the prior and candidate field layouts against the existing Clang 18/libstdc++ module BMI on this host. Its `sizeof` results in bytes were: `ElementImpl` equivalent 344 before and 56 with the variant; `ColumnViewImpl` equivalent 120 before and 80 with the shared-plan handle; `Selection` 120. These exclude shared control blocks, pointed-to payloads and allocator overhead; a pruned result now additionally allocates a plan. Sizes are implementation-specific, not runtime or workload measurements. The exact probe source and binary are preserved locally at `build/v1-cleanup/data-layout.cpp` (SHA-256 `848fde3f1e1b256a70b7942d3f7924641ae4ea93dccf4696afa40563b493e404`) and `build/v1-cleanup/data-layout` (SHA-256 `c96c568e9fc4114d311fc084752f3dc234bc64d9e7b898f06e3a2c26771098d0`); the compared variant enumerates the scalar, string, array, string-array, record, column, table, mapping and retained-view types in `src/data.cpp`.

Public owner/view APIs and biological alignment rules are unchanged. The private compiled `detail::empty_like` bridge is an implementation facility, not a consumer migration target. Retained views capture structural bindings, not frozen payloads: callers must avoid unsynchronized reads and writes to shared data, and any borrowed external buffer must outlive all views using it.

The probe source is reproduced below; build it against the baseline module and library with matching Clang and standard-library versions. It compares only object layouts.

```cpp
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>
import anndatapp.core;
using namespace anndatapp;
using Mapping = std::vector<std::pair<std::string, Element>>;
struct Existing {
    ElementKind kind; DType scalar_dtype; ByteOrder scalar_byte_order;
    bool boolean; std::int64_t si; std::uint64_t ui; double fp;
    std::string string;
    std::optional<Array> array; std::optional<ArrayView> view_array;
    std::optional<StringArray> string_array; std::optional<StringArrayView> view_string_array;
    std::optional<RecordArray> record_array;
    std::optional<Column> column; std::optional<ColumnView> view_column;
    std::optional<Table> table; std::optional<TableView> view_table;
    Mapping mapping;
};
using Payload = std::variant<std::monostate, bool, std::int64_t, std::uint64_t, double,
    std::string, Array, ArrayView, StringArray, StringArrayView, RecordArray,
    Column, ColumnView, Table, TableView, Mapping>;
struct Alternative { ElementKind kind; DType scalar_dtype; ByteOrder scalar_byte_order; Payload payload; };
struct OldPlan { std::shared_ptr<void> source; Axis rows; std::vector<std::size_t> category_rows; std::vector<std::int64_t> recode; bool pruned; };
struct NewPlan { std::shared_ptr<void> source; Axis rows; std::shared_ptr<const void> categories; };
int main() {
    std::cout << "Element existing=" << sizeof(Existing) << " variant=" << sizeof(Alternative)
              << " ColumnView existing=" << sizeof(OldPlan) << " shared-plan=" << sizeof(NewPlan)
              << " Selection=" << sizeof(Selection) << '\n';
}
```

A provisional isolated correctness check configured `build/v1-cleanup/data-check` with Clang 18, Debug, Ninja, both storage backends disabled and interoperability tests disabled, then built `test_data` at `-j2` and ran it successfully (`all data tests passed`). `ninja -n test_data` reported no remaining work immediately afterward. At that check the SHA-256 hashes were `src/data.cpp` `25d61932b968739751cc1ba3734be12d40bfc085be2d64a9c3f8e443349d73d2`, `src/anndatapp.data.cppm` `4eebf7aca9973363ec43ecb141d92063e136895e50d4fd59e55bbe37222d07a9`, and `tests/test_data.cpp` `1f6ab081e970f651e51f704a84288c57143398d39617f517841d8b836537f9d6`. This targeted check is not the final integrated build, sanitizer or storage interoperability result.
