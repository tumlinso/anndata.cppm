import anndata.core;

int main() {
    auto data = anndata::AnnData::empty(3, 2);
    data.set_x(anndata::Array::zeros<float>({3, 2}));
    auto x = data.edit_x().dense<float>();
    x.data()[5] = 7.0f;
    data.set_obs_column("tag", anndata::Column::from_values<int>({1, 2, 3}));
    auto selected = data.x().select({anndata::Axis::stride(2, 1),
                                    anndata::Axis::gather({1})}).dense<float>();
    auto tag = data.obs_column("tag").values().dense<int>();
    auto retained = data.view().obs_column("tag").values().dense<int>();
    return selected.data() == x.data() + 5 && selected.data()[0] == 7.0f &&
           tag.data()[2] == 3 && retained.data() == tag.data() ? 0 : 1;
}
