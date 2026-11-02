import anndata.core;

int main() {
    auto data = anndata::AnnData::empty(3, 2);
    data.set_x(anndata::Array::zeros<float>({3, 2}));
    auto x = data.edit_x().dense<float>();
    x.data()[5] = 7.0f;
    return data.x().dense<float>().data()[5] == 7.0f ? 0 : 1;
}
