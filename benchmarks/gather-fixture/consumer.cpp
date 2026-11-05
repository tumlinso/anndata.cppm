import anndatapp.core;

#include <cstddef>
#include <initializer_list>
#include <string_view>
#include <vector>

int main() {
    float scalar = 4.5f;
    const std::size_t positions[] = {2, 0};
    const auto axis = anndatapp::Axis::gather(positions, 2);
    auto data = anndatapp::AnnData::empty(3, 1);
    data.set_x(anndatapp::Array::borrow_dense(&scalar, {3, 1}, {0, 0},
                                            anndatapp::Validation::none));
    auto selected = data.slice(anndatapp::Selection(anndatapp::indices{2, 0}), anndatapp::all);
    const auto dense = selected.x().dense<float>();
    const auto obs = selected.obs();
    return axis.index(0) == 2 && axis.index(1) == 0 &&
           selected.n_obs() == 2 && dense.data() == &scalar &&
           dense.extent(0) == 2 && obs.name(0) == "2" && obs.name(1) == "0" ? 0 : 1;
}
