#include <cstddef>
#include <vector>

import anndatapp.streaming;

int main() {
    anndatapp::StorageGeometry g;
    g.dtype = anndatapp::DType::float32;
    g.shape = g.source_shape = {17, 3}; g.axes.resize(2);
    g.parallel_reads = true;
    anndatapp::StreamOptions o;
    o.mode = anndatapp::StreamMode::manual;
    o.min_block_bytes = 4; o.target_block_bytes = 24; o.max_block_bytes = 48;
    o.max_in_flight = 2;
    anndatapp::BlockStream stream(g, o);
    std::vector<unsigned> visits(51);
    while (auto block = stream.next()) {
        for (std::size_t i = 0; i < block->count[0]; ++i)
            for (std::size_t j = 0; j < block->count[1]; ++j)
                ++visits[(block->start[0] + i)*3 + block->start[1] + j];
        stream.complete(*block, block->estimated_bytes);
    }
    for (auto n : visits) if (n != 1) return 1;
    return stream.finished() ? 0 : 1;
}
