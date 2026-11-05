import anndatapp.backed;

int main() {
    anndatapp::BackedOptions options;
    anndatapp::BackedAnnData closed;
    return !closed.is_open() && options.scratch_bytes == 64u * 1024u * 1024u ? 0 : 1;
}
