import anndatapp.core;
int main() {
    auto x = anndatapp::Array::zeros<float>({2, 2});
    return x.view().dense<float>().data()[0] != 0;
}
