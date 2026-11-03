#include <cstddef>
#include <string_view>

import anndata.core;

int main() {
    using namespace anndata;
    struct Row { float score; Utf8Ref name; };
    Row rows[]={{1,{"gene",4}},{2,{"other",5}}};
    RecordLayout layout{sizeof(Row),alignof(Row),{
        {"score",RecordFieldKind::numeric,DType::float32,offsetof(Row,score)},
        {"name",RecordFieldKind::utf8_ref,DType::uint8,offsetof(Row,name)}}};
    auto record=RecordArray::borrow_interleaved(rows,{2},layout);
    auto x=record.edit_member<float>("score"); x[1]=3;
    auto copy=record.copy();
    return rows[1].score==3 && copy.view().string("name",0)=="gene"?0:1;
}
