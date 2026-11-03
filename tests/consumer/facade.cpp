import anndata;

int main(int argc, char** argv) {
    auto data = anndata::AnnData::empty(3, 2);
    if (data.n_obs() != 3 || data.n_vars() != 2) return 1;
#if ANNDATA_HAS_HDF5
    if (argc != 2) return 2;
    anndata::write_h5ad(argv[1], data);
    auto loaded = anndata::read_h5ad(argv[1]);
    return loaded.n_obs() == 3 && loaded.n_vars() == 2 ? 0 : 3;
#elif ANNDATA_HAS_ZARR
    if (argc != 2) return 2;
    anndata::write_zarr(argv[1], data);
    auto loaded = anndata::read_zarr(argv[1]);
    return loaded.n_obs() == 3 && loaded.n_vars() == 2 ? 0 : 3;
#else
    (void)argc;
    (void)argv;
    return 0;
#endif
}
