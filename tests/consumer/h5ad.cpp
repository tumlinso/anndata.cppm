import anndata.h5ad;

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    auto data = anndata::AnnData::empty(2, 3);
    data.set_x(anndata::Array::zeros<float>({2, 3}));
    data.edit_x().dense<float>().data()[4] = 7.0f;
    anndata::write_h5ad(argv[1], data);
    auto loaded = anndata::read_h5ad(argv[1]);
    auto disk = anndata::open_h5ad(argv[1]);
    auto block = disk.x().read();
    if (block.view().dense<float>().data()[4] != 7.0f) return 3;
    return loaded.n_obs() == 2 && loaded.n_vars() == 3
        && loaded.x().dense<float>().data()[4] == 7.0f ? 0 : 2;
}
