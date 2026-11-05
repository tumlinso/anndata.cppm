import anndatapp;

int main(int argc, char** argv) {
    auto data = anndatapp::AnnData::empty(3, 2);
    if (data.n_obs() != 3 || data.n_vars() != 2) return 1;
#if ANNDATAPP_HAS_HDF5
    if (argc != 2) return 2;
    anndatapp::write_h5ad(argv[1], data);
    auto loaded = anndatapp::read_h5ad(argv[1]);
    return loaded.n_obs() == 3 && loaded.n_vars() == 2 ? 0 : 3;
#elif ANNDATAPP_HAS_ZARR
    if (argc != 2) return 2;
    anndatapp::write_zarr(argv[1], data);
    auto loaded = anndatapp::read_zarr(argv[1]);
    return loaded.n_obs() == 3 && loaded.n_vars() == 2 ? 0 : 3;
#else
    (void)argc;
    (void)argv;
    return 0;
#endif
}
