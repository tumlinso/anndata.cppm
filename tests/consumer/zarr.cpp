import anndatapp.zarr;

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    auto data = anndatapp::AnnData::empty(2, 3);
    data.set_x(anndatapp::Array::zeros<float>({2, 3}));
    data.edit_x().dense<float>().data()[4] = 7.0f;
    anndatapp::write_zarr(argv[1], data);
    auto disk = anndatapp::open_zarr(argv[1]);
    auto loaded = disk.x().read();
    return disk.n_obs() == 2 && disk.n_vars() == 3
        && loaded.view().dense<float>().data()[4] == 7.0f ? 0 : 2;
}
