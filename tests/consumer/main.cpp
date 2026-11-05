import anndatapp.core;

int main() {
    auto data = anndatapp::AnnData::empty(3, 2);
    data.set_x(anndatapp::Array::zeros<float>({3, 2}));
    auto x = data.edit_x().dense<float>();
    x.data()[5] = 7.0f;
    data.set_obs_column("tag", anndatapp::Column::from_values<int>({1, 2, 3}));
    auto selected = data.x().select({anndatapp::Axis::stride(2, 1),
                                    anndatapp::Axis::gather({1})}).dense<float>();
    auto tag = data.obs_column("tag").values().dense<int>();
    auto retained = data.view().obs_column("tag").values().dense<int>();
    auto packed = anndatapp::Array::empty<float>({1, 1});
    data.x().select({anndatapp::Axis::stride(2, 1), anndatapp::Axis::gather({1})}).copy_into(packed.edit());
    return selected.data() == x.data() + 5 && selected.data()[0] == 7.0f &&
           tag.data()[2] == 3 && retained.data() == tag.data() &&
           packed.view().dense<float>().data()[0] == 7.0f ? 0 : 1;
}
