# Coding style

Write in the compact, explicit style of classic C and Fortran numerical/HPC
software. Source should expose the algorithm and its relationship to memory:
loads, stores, indexing, traversal, allocation, copying, and calls should be easy
to reason about. This transparency matters independently of compilation time
and runtime performance.

- **Keep code short and locally understandable.** Keep the calculation and its
  relevant invariants together. Brevity helps human readers and agents retain
  useful context. Keep control flow visible when formatting nested operations.
  Use one-line guards and loop bodies when the operation fits naturally; keep
  complete mathematical expressions together and use ternaries for value
  choices. Compactness is the default, not an optional final formatting pass.
- **Use mathematical notation where it fits.** Names such as `i`, `j`, `k`, `n`,
  `x`, `p`, and `nnz` are welcome in focused algorithms. Give slot names, units,
  ownership boundaries, and byte-versus-element strides enough precision to
  prevent ambiguity. Long names are not a substitute for a clear calculation.
- **Prefer direct representations.** Use pointers, references, arrays, explicit
  dimensions and strides, and ordinary loops. Make mutability and ownership
  apparent. Document the work done by accessors that allocate, traverse, recode,
  or perform I/O.
- **Keep machinery minimal.** Standard-library facilities and dependencies need
  a concrete benefit that justifies their concepts and implementation burden.
  Prefer a straightforward loop or data structure when it expresses the work
  clearly. Avoid wrapper hierarchies, callback layers, and generic scaffolding
  introduced merely for extensibility. Replacing a container with a custom
  framework would need the same justification.
  Review the full representation: stored state, allocation count, duplicated
  ownership, and helper layers. A short call site alone is not compact code.
- **Use C++ deliberately.** Templates, RAII, references, and other C++ facilities
  are welcome when they express useful structure concisely and leave execution
  understandable. Keep template bodies small and implementation details behind
  compiled module boundaries.
- **Separate acquisition from computation.** Resolve types and layouts, retain
  ownership, and validate at appropriate boundaries. Provide explicit borrowed
  and trusted paths. Once numerical pointers and strides are acquired, leave
  the caller's loops free of library dispatch, repeated checks, and
  synchronization. Callers control execution and parallelism.

Preserve C++20 named modules, AnnData semantics, explicit copies, retained-view
lifetimes, and private storage dependencies. Apply this style to new and touched
code through focused changes. Performance claims still require measurements.

Follow the written guidance in
[Cellerator's coding_style.md](/home/tumlinson/Cellerator/coding_style.md), not
its existing implementation as an example. For this library, small compiled
module interfaces, fast imports, and consumer recompilation time remain very
high priorities. Keep implementation changes behind those boundaries.
