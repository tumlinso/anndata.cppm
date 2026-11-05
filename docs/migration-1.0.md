# Cleanup migration notes

These changes define the documented source API for 1.x. They describe source changes made before 1.0; the compatibility fixes made during qualification add no further public API changes.

## Direct borrowed dense descriptors

Use the public factory instead of constructing a descriptor from `detail::DenseDimensions`:

```cpp
std::size_t shape[] = {rows, columns};
std::ptrdiff_t strides[] = {std::ptrdiff_t(columns), 1};
auto x = anndatapp::borrowed_dense(data, 2, shape, strides);
```

The factory copies rank/shape/stride metadata; only numerical bytes are borrowed. Strides count elements. Common ranks use inline metadata; larger ranks can allocate metadata. `Validation::none` is available for trusted callers. Keep the external data alive and respect constness and aliasing.

Retained `Array::borrow_dense` wrappers and checked retained acquisition remain available. Descriptor constructors accepting implementation structs are not consumer entry points.

## Focused record access

Acquire a single field directly from its owner with `member<T>(name)`, `text(name)`, `array(name)` or `string_array(name)` as appropriate. Use `view()` when a captured whole-record schema is required. A numerical interleaved member has byte strides; its `data()` is not a contiguous `T[]` across records. Retained bindings survive owner replacement without freezing payload contents.

## Package targets

The ordinary `anndatapp::anndatapp` target and existing imports remain available. A consumer requiring only the core can avoid optional storage discovery and interface compilation:

```cmake
find_package(anndatapp CONFIG REQUIRED COMPONENTS core)
target_link_libraries(my_target PRIVATE anndatapp::core)
```

Consumers still build their own compatible BMIs. Keep `CMAKE_CXX_EXTENSIONS OFF` to match the module toolchain. Internal dataset-state constructors and `detail` facade bridges are not part of the documented source-compatibility promise.

Change explicit package version requests from `find_package(anndatapp 0.1 ...)` to `find_package(anndatapp 1.0 ...)`. The package uses CMake's `SameMajorVersion` policy; an explicit 0.x requirement does not admit 1.0. Unversioned requests remain available.

## Explicit contracts

Null C-string selectors and malformed storage axes now receive contextual errors. Record validation-mode overrides validate through the storage adapters rather than rejecting supported records. Invalid roots are rejected before filesystem work, and failed category-page refills do not publish partial pages.

Prepare backed descriptors/contexts serially and execute independent contexts under the [concurrency contract](support.md). Failed stream requests remain outstanding; do not report them complete. Transfer allocation counters describe their documented staging events, not all allocations or process memory. Existing trusted-input, explicit-copy and adaptive-default behavior remain deliberate capabilities.
