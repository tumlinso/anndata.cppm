find_path(Blosc_INCLUDE_DIR NAMES blosc.h)
find_library(Blosc_LIBRARY NAMES blosc)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Blosc REQUIRED_VARS Blosc_LIBRARY Blosc_INCLUDE_DIR)
if(Blosc_FOUND AND NOT TARGET Blosc::Blosc)
  # Preserve shared dependency metadata, including non-system runtime paths.
  if(Blosc_LIBRARY MATCHES "\\.(so(\\.[0-9.]+)?|dylib)$")
    add_library(Blosc::Blosc SHARED IMPORTED GLOBAL)
  else()
    add_library(Blosc::Blosc UNKNOWN IMPORTED GLOBAL)
  endif()
  set_target_properties(Blosc::Blosc PROPERTIES
    IMPORTED_LOCATION "${Blosc_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Blosc_INCLUDE_DIR}")
endif()
mark_as_advanced(Blosc_INCLUDE_DIR Blosc_LIBRARY)
