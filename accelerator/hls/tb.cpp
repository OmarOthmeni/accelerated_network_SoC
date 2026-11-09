// C-simulation testbench (runs with g++ or Vitis HLS csim).
// Runs cnn_top on the whole test set and requires the int32 logits to match the
// Python integer golden model exactly; also reports top-1 accuracy.
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "cnn.h"
#include "weights.h"

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif

template <typename T>
static std::vector<T> load(const char *name, size_t count) {
    std::vector<T> v(count);
    char path[512];
    snprintf(path, sizeof path, "%s/%s", DATA_DIR, name);
    FILE *f = fopen(path, "rb");
    if (!f || fread(v.data(), sizeof(T), count, f) != count) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(2);
    }
    fclose(f);
    return v;
}

int main() {
    const int n = N_TEST_IMAGES;
    auto images = load<uint8_t>("test_images.bin", (size_t)n * IMG_H * IMG_W);
    auto labels = load<int32_t>("test_labels.bin", n);
    auto golden = load<int32_t>("golden_logits.bin", (size_t)n * N_OUT);
    std::vector<int32_t> logits((size_t)n * N_OUT);

    cnn_top(images.data(), logits.data(), n);

    int mismatches = 0, correct = 0;
    for (int i = 0; i < n; i++) {
        int best = 0;
        for (int k = 0; k < N_OUT; k++) {
            if (logits[i * N_OUT + k] != golden[i * N_OUT + k]) mismatches++;
            if (logits[i * N_OUT + k] > logits[i * N_OUT + best]) best = k;
        }
        correct += (best == labels[i]);
    }
    printf("images: %d | logit mismatches vs golden: %d | top-1 accuracy: %.4f\n",
           n, mismatches, (double)correct / n);
    if (mismatches) {
        printf("FAIL: HLS kernel is not bit-exact with the golden model\n");
        return 1;
    }
    printf("PASS: bit-exact with the golden model\n");
    return 0;
}
