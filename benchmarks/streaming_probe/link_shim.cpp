// Baseline d9869da's Clang 18 Release archive has an unresolved template move
// assignment in backed_writer.cpp. Emit the ordinary standard-library
// specialization in both probe binaries, without modifying the source snapshot.
#include <cstddef>
#include <memory>
#include <vector>
using IndexVector = const std::vector<std::size_t>;
template std::shared_ptr<IndexVector>&
std::shared_ptr<IndexVector>::operator=(std::shared_ptr<IndexVector>&&) noexcept;
