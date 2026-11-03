#ifndef ANNDATA_RECORD_IO_PROBE_H
#define ANNDATA_RECORD_IO_PROBE_H

#include <stddef.h>

typedef void (*record_probe_enable_fn)(int);
typedef void (*record_probe_reset_fn)(void);
typedef size_t (*record_probe_json_fn)(char *, size_t);

#define RECORD_PROBE_ENABLE "anndata_record_probe_enable"
#define RECORD_PROBE_RESET  "anndata_record_probe_reset"
#define RECORD_PROBE_JSON   "anndata_record_probe_json"

#endif
