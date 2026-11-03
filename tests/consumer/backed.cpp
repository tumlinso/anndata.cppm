import anndata.backed;

int main() {
    anndata::BackedOptions options;
    anndata::BackedAnnData closed;
    return !closed.is_open() && options.scratch_bytes == 64u * 1024u * 1024u ? 0 : 1;
}
