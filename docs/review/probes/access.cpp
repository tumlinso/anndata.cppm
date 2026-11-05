#include <array>
#include <barrier>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
import anndatapp;
using namespace anndatapp;
int main(int argc, char** argv) {
  if(argc != 3) return 2;
  if(std::string(argv[1])=="malformed") {
    try { auto x=open_zarr(argv[2]); std::cout<<"unexpected open "<<x.n_obs()<<"\n"; }
    catch(const std::exception& e) { std::cout<<"diagnostic: "<<e.what()<<"\n"; return 0; }
    return 3;
  }
  try {
    constexpr std::size_t n=16, workers=4;
    auto fixture=AnnData::empty(n,n);
    auto matrix=Array::empty<float>({n,n});
    for(std::size_t j=0;j<n*n;++j) matrix.edit().dense<float>().data()[j]=float(j);
    fixture.set_x(std::move(matrix));
    for(std::size_t w=0;w<workers;++w) fixture.set_layer("layer"+std::to_string(w),Array::zeros<float>({n,n}));
    ZarrWriteOptions o; o.mode=ZarrWriteMode::exclusive; o.format=std::string(argv[1])=="v2"?ZarrFormat::v2:ZarrFormat::v3;
    write_zarr(argv[2],fixture,o);
    BackedOptions opts; opts.mode=AccessMode::read_write;
    auto store=open_zarr(argv[2],opts);
    std::vector<TransferContext> readers, writers;
    for(std::size_t i=0;i<workers;++i) {readers.push_back(store.x().prepare_transfer());writers.push_back(store.layer("layer"+std::to_string(i)).prepare_transfer());}
    std::barrier gate{std::ptrdiff_t(workers)};
    std::vector<std::future<void>> jobs;
    for(std::size_t i=0;i<workers;++i) jobs.push_back(std::async(std::launch::async,[&,i] {
      std::array<float,n*n> values{}; gate.arrive_and_wait();
      for(int rep=0;rep<10;++rep) {
        readers[i].read_into({0,0},{n,n},DenseBuffer::from(values.data(),{n,n}));
        for(std::size_t j=0;j<n*n;++j) if(values[j]!=float(j)) throw std::runtime_error("same-session read mismatch");
        for(auto& x:values)x=float(i+1);
        writers[i].write_from({0,0},{n,n},DenseBuffer::from(values.data(),{n,n}));
      }
    }));
    for(auto& j:jobs) j.get();
    for(std::size_t i=0;i<workers;++i) {std::array<float,n*n> v{};writers[i].read_into({0,0},{n,n},DenseBuffer::from(v.data(),{n,n})); for(auto x:v)if(x!=float(i+1))throw std::runtime_error("write mismatch");}
    std::cout<<"same-session independent contexts passed "<<argv[1]<<"\n";
  } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
