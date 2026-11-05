# Include review and reproducible inventory

The frozen review covered **99 C/C++ files and 991 include sites**: 384 in the 36 production files and 607 in supporting code. Of those sites, 847 use C++ standard headers, 35 C standard headers, 21 backend headers, 74 platform headers, 12 local headers and 2 other external headers. All 991 directive locations and source hashes were checked. The syntax inventory recorded 379 type/alias declarations.

The architectural assessment of all 31 distinct production C++ standard headers is in [library-choices.md](library-choices.md). Per-subsystem findings account for the representation and execution behind those facilities. Include presence alone establishes no runtime overhead.

The full per-occurrence JSON and rendered table are generated artifacts, preserved locally under `build/thin-review-20261105/generated-inventory/` and excluded from this documentation commit. Regenerate an inventory with the committed [scanner](../../tools/review_inventory.py):

```sh
python3 tools/review_inventory.py --source . \
  --output build/include-review/includes.json \
  --markdown build/include-review/includes.md
```

To reproduce the original source identity, use `--source build/thin-review-20261105/source` while that preserved snapshot is available, and compare input hashes with [source-manifest.json](source-manifest.json). Running against another checkout produces an inventory of that checkout; it does not recreate the original frozen evidence.

Each generated row records its source hash and line, header category, representative symbol evidence, public-interface exposure and conservative disposition. `remove_candidate` and missing-direct-header suggestions are lexical candidates, not compiler-verified edits. Macros, unqualified calls, imported declarations, overloads and conditional compilation can require deeper inspection. No production include was removed and no header-attributed build/runtime benefit was measured in the review.

The tool can also check source references in individual production ledgers:

```sh
python3 tools/review_inventory.py --source build/thin-review-20261105/source \
  --output build/include-review/includes.json \
  --validate-ledger docs/review/core-ledger.json
```

The original source manifest, compact coverage receipt, authored abstraction ledgers and findings remain committed. Full generated inventories and documentation-rendering output remain local.
