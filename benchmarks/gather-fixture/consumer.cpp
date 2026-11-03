import anndata.core;

#include <cstddef>
#include <initializer_list>
#include <string_view>
#include <vector>

int main() {
    float scalar = 4.5f;
    const std::size_t positions[] = {2, 0};
    const auto axis = anndata::Axis::gather(positions, 2);
    auto data = anndata::AnnData::empty(3, 1);
    data.set_x(anndata::Array::borrow_dense(&scalar, {3, 1}, {0, 0},
                                            anndata::Validation::none));
    auto selected = data.slice(anndata::Selection(anndata::indices{2, 0}), anndata::all);
    const auto dense = selected.x().dense<float>();
    const auto obs = selected.obs();
    return axis.index(0) == 2 && axis.index(1) == 0 &&
           selected.n_obs() == 2 && dense.data() == &scalar &&
           dense.extent(0) == 2 && obs.name(0) == "2" && obs.name(1) == "0" ? 0 : 1;
}
