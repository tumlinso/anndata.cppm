import anndatapp;

int main() {
    auto data = anndatapp::AnnData::empty(3, 2);
    return data.n_obs() == 3 && data.n_vars() == 2 ? 0 : 1;
}
