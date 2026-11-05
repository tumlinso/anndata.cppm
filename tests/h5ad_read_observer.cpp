// Darwin test-only interposition across the HDF5 shared-library boundary.
#include <hdf5.h>

namespace {
using Observer = void (*)(hid_t,hid_t,hid_t,herr_t);
Observer observer = nullptr;
herr_t observe_h5d_read(hid_t dataset,hid_t type,hid_t memory,hid_t file,hid_t transfer,void* data) {
    const auto status = H5Dread(dataset,type,memory,file,transfer,data);
    if (observer) observer(dataset,memory,transfer,status);
    return status;
}
__attribute__((used,section("__DATA,__interpose")))
const struct { const void* replacement; const void* original; } interpose = {
    reinterpret_cast<const void*>(&observe_h5d_read), reinterpret_cast<const void*>(&H5Dread)};
}
extern "C" void anndatapp_test_observe_h5dread(Observer callback) { observer = callback; }
