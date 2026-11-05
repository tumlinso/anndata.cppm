import anndatapp.h5ad;

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    auto data = anndatapp::AnnData::empty(2, 3);
    data.set_x(anndatapp::Array::zeros<float>({2, 3}));
    data.edit_x().dense<float>().data()[4] = 7.0f;
    anndatapp::write_h5ad(argv[1], data);
    auto loaded = anndatapp::read_h5ad(argv[1]);
    auto disk = anndatapp::open_h5ad(argv[1]);
    auto block = disk.x().read();
    if (block.view().dense<float>().data()[4] != 7.0f) return 3;
    return loaded.n_obs() == 2 && loaded.n_vars() == 3
        && loaded.x().dense<float>().data()[4] == 7.0f ? 0 : 2;
}
