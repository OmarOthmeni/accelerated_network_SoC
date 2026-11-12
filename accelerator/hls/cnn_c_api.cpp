// C entry point so the CPU build of the kernel can be loaded from Python (ctypes) by
// demo/board_server.py --backend cpu. Not used by Vitis HLS.
#include "cnn.h"

extern "C" void cnn_classify(const uint8_t *images, int32_t *logits, int n_images) {
    cnn_top(images, logits, n_images);
}
