#pragma once

#include <hdf5.h>

namespace anndatapp_h5ad_detail {

// Keep native-file object addresses for hard-link cycle detection. HDF5 1.12
// changed the default info API to tokens; select the address API explicitly.
#if H5_VERSION_GE(1,12,0)
using ObjectInfo = H5O_info1_t;
inline herr_t object_info(hid_t object,ObjectInfo* info) { return H5Oget_info1(object,info); }
#else
using ObjectInfo = H5O_info_t;
inline herr_t object_info(hid_t object,ObjectInfo* info) { return H5Oget_info(object,info); }
#endif

} // namespace anndatapp_h5ad_detail
