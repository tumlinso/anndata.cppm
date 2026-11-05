# Installed core-import build work

This single executable imports only the core. Configure against the installed frozen Debug package, use a fresh output tree, build and execute it, then count the provider's `.pcm` artifacts:

```sh
cmake -S docs/review/probes/core-import -B /tmp/anndatapp-core-import-review -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_PREFIX_PATH='/home/tumlinson/anndatapp/build/thin-review-20261105/source/build/debug/installed-test/prefix;/home/tumlinson/anndatapp/build/dependencies/usr'
timeout 30s cmake --build /tmp/anndatapp-core-import-review --parallel 2
/tmp/anndatapp-core-import-review/core_consumer
```

The original equivalent fixture lives in ignored `build/thin-review-20261105/core-consumer/`; its logs and artifact sizes are recorded in `../../validation.json`. That run compiled all ten enabled public BMIs. The supplied package provider builds its complete file set even for this core-only import. These are build-work and artifact-size observations, not a timing comparison. Paths and dependencies identify this host; use an explicitly installed prefix for another environment.
