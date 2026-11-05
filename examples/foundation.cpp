import anndatapp;

int main() {
    auto data = anndatapp::AnnData::empty(3, 2);
    data.set_x(anndatapp::Array::empty<float>({3, 2}));
    auto x = data.edit_x().dense<float>();
    float* p = x.data();
    for (int i = 0; i < 6; ++i) p[i] = float(i);
    auto selected = data.slice(anndatapp::indices{0, 2}, anndatapp::all);
    auto independent = selected.copy();
    return independent.x().dense<float>().data()[2] == 4.0f ? 0 : 1;
}
