#pragma once

#include <hdf5.h>
#include <cstddef>
#include <string_view>

namespace anndatapp_h5ad_detail {

// Borrowed bytes: NULLTERM ends at the first NUL; the other modes remove only
// their trailing padding. Interior NUL bytes remain text data under NULLPAD.
inline std::string_view fixed_text(const char* p,std::size_t n,H5T_str_t padding) noexcept {
    std::string_view text(p,n);
    if (padding == H5T_STR_NULLTERM) return text.substr(0,text.find('\0'));
    if (padding == H5T_STR_NULLPAD) while (!text.empty() && text.back() == '\0') text.remove_suffix(1);
    else if (padding == H5T_STR_SPACEPAD) while (!text.empty() && text.back() == ' ') text.remove_suffix(1);
    return text;
}

} // namespace anndatapp_h5ad_detail
