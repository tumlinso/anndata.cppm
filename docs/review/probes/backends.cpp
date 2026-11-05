#include "zarr_engine.hpp"
#include <fstream>
#include <iostream>
using namespace anndatapp_zarr;
void save(const std::filesystem::path& p, const Json& j) { std::ofstream(p)<<j; }
int main() {
    const auto root=std::filesystem::path("/tmp/anndatapp-backend-review/stores");
    std::filesystem::create_directories(root);
    Json results=Json::object();
    for (int format : {2,3}) {
        auto p=root/ (format==2 ? "v2" : "v3"); std::filesystem::create_directories(p);
        Json child = format==2 ? Json{{"zarr_format",2},{"shape",{1}},{"chunks",{1}},{"dtype","<i4"},{"compressor",nullptr},{"filters",nullptr},{"fill_value",0},{"order","C"}}
            : Json{{"zarr_format",3},{"node_type","array"},{"shape",{1}},{"data_type","int32"},{"chunk_grid",{{"name","regular"},{"configuration",{{"chunk_shape",{1}}}}}},{"chunk_key_encoding",{{"name","default"},{"configuration",{{"separator","/"}}}}},{"fill_value",0},{"codecs",Json::array({Json{{"name","bytes"},{"configuration",{{"endian","little"}}}}})},{"attributes",Json::object()}};
        if(format==2) {
            save(p/".zgroup",{{"zarr_format",2}});
            save(p/".zmetadata",{{"zarr_consolidated_format",1},{"metadata",{{".zgroup",{{"zarr_format",2}}},{"X/.zarray",child},{"X/.zattrs",Json::object()}}}});
        } else save(p/"zarr.json",{{"zarr_format",3},{"node_type","group"},{"attributes",Json::object()},{"consolidated_metadata",{{"kind","inline"},{"must_understand",false},{"metadata",{{"X",child}}}}}});
        Store s(p,true); Json r; r["before"]=s.inspect("X").exists;
        try { s.attributes("X",{{"test",1}}); r["exception"]=nullptr; }
        catch(const Error& e) { r["exception"]=e.what(); }
        r["after"]=s.inspect("X").exists; results["consolidated_v"+std::to_string(format)]=r;
    }
    auto p=root/"nul"; std::filesystem::create_directories(p); save(p/".zgroup",{{"zarr_format",2}});
    std::string nul=p.string(); nul.append("\0unrequested",12);
    try {
        Store s(std::filesystem::path(nul),false);
        results["root_with_embedded_nul_result"]=s.inspect("").exists;
    } catch (const std::exception& e) { results["root_with_embedded_nul_exception"]=e.what(); }
    std::cout<<results.dump(2)<<'\n';
}
