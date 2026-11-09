#pragma once
#include <stdint.h>

#define IMG_H 64
#define IMG_W 64
#define N_OUT 24

// Classify n_images 64x64 uint8 images stored back to back in `images`;
// writes N_OUT int32 logits per image to `logits` (argmax = predicted class).
void cnn_top(const uint8_t *images, int32_t *logits, int n_images);
