// INT8 CNN accelerator for the Zynq-7020 (Vitis HLS).
//
// Network (see model/net.py): 4 x [conv3x3 + ReLU + maxpool2] with 16/32/64/64 channels,
// global average pooling, fully connected -> 24 classes. All weights live in on-chip ROMs.
//
// Arithmetic is bit-exact with golden_infer() in model/quantize.py:
//   acc  = bias + sum(w_int8 * a_uint8)                      int32
//   a    = clamp((acc * M + 2^(S-1)) >> S, 0, 255)            uint8 (requantize + ReLU)
//   pool = max over 2x2
//
// Plain C++ integer types are used so the same file compiles with g++ (C simulation)
// and with Vitis HLS; the pragmas are ignored by g++.

#include "cnn.h"
#include "weights.h"

// output channels computed in parallel per conv layer (MACs per cycle = PO * 9)
#define PO 8

static inline uint8_t requant(int32_t acc, int32_t m, uint8_t s) {
#pragma HLS INLINE
    int64_t r = ((int64_t)acc * (int64_t)m + ((int64_t)1 << (s - 1))) >> s;
    return r < 0 ? 0 : (r > 255 ? 255 : (uint8_t)r);
}

// conv3x3 (zero padding) + requant/ReLU + 2x2 maxpool, fused:
// for each pooled output pixel, the 4 conv outputs underneath are computed and maxed,
// so the full-resolution conv output never has to be stored.
template <int CIN, int COUT, int H, int W>
static void conv_relu_pool(const uint8_t in[CIN][H][W],
                           uint8_t out[COUT][H / 2][W / 2],
                           const int8_t w[COUT][CIN][3][3],
                           const int32_t b[COUT], const int32_t m[COUT], const uint8_t s[COUT]) {
CO_TILE:
    for (int co0 = 0; co0 < COUT; co0 += PO) {
    PY:
        for (int py = 0; py < H / 2; py++) {
        PX:
            for (int px = 0; px < W / 2; px++) {
                uint8_t best[PO];
#pragma HLS ARRAY_PARTITION variable=best complete
                for (int p = 0; p < PO; p++) best[p] = 0;
            SUB:
                for (int sub = 0; sub < 4; sub++) {
                    const int y = 2 * py + (sub >> 1);
                    const int x = 2 * px + (sub & 1);
                    int32_t acc[PO];
#pragma HLS ARRAY_PARTITION variable=acc complete
                    for (int p = 0; p < PO; p++) acc[p] = b[co0 + p];
                CI:
                    for (int ci = 0; ci < CIN; ci++) {
#pragma HLS PIPELINE II=1
                        uint8_t win[3][3];
#pragma HLS ARRAY_PARTITION variable=win complete dim=0
                        for (int ky = 0; ky < 3; ky++)
                            for (int kx = 0; kx < 3; kx++) {
                                const int yy = y + ky - 1, xx = x + kx - 1;
                                win[ky][kx] = (yy >= 0 && yy < H && xx >= 0 && xx < W) ? in[ci][yy][xx] : 0;
                            }
                        for (int p = 0; p < PO; p++) {
                            int32_t sum = 0;
                            for (int ky = 0; ky < 3; ky++)
                                for (int kx = 0; kx < 3; kx++)
                                    sum += (int32_t)w[co0 + p][ci][ky][kx] * (int32_t)win[ky][kx];
                            acc[p] += sum;
                        }
                    }
                    for (int p = 0; p < PO; p++) {
                        const uint8_t v = requant(acc[p], m[co0 + p], s[co0 + p]);
                        if (v > best[p]) best[p] = v;
                    }
                }
                for (int p = 0; p < PO; p++) out[co0 + p][py][px] = best[p];
            }
        }
    }
}

static void classify_one(const uint8_t img[IMG_H][IMG_W], int32_t out[N_OUT]) {
    static uint8_t a0[1][64][64];
    static uint8_t a1[16][32][32];
    static uint8_t a2[32][16][16];
    static uint8_t a3[64][8][8];
    static uint8_t a4[64][4][4];
// 3x3 windows: rows/columns y-1, y, y+1 fall in 3 different banks
#pragma HLS ARRAY_PARTITION variable=a0 cyclic factor=3 dim=2
#pragma HLS ARRAY_PARTITION variable=a0 cyclic factor=3 dim=3
#pragma HLS ARRAY_PARTITION variable=a1 cyclic factor=3 dim=2
#pragma HLS ARRAY_PARTITION variable=a1 cyclic factor=3 dim=3
#pragma HLS ARRAY_PARTITION variable=a2 cyclic factor=3 dim=2
#pragma HLS ARRAY_PARTITION variable=a2 cyclic factor=3 dim=3
#pragma HLS ARRAY_PARTITION variable=a3 cyclic factor=3 dim=2
#pragma HLS ARRAY_PARTITION variable=a3 cyclic factor=3 dim=3
// PO output channels read their weights in the same cycle
#pragma HLS ARRAY_PARTITION variable=W0 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=W0 complete dim=3
#pragma HLS ARRAY_PARTITION variable=W0 complete dim=4
#pragma HLS ARRAY_PARTITION variable=W1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=W1 complete dim=3
#pragma HLS ARRAY_PARTITION variable=W1 complete dim=4
#pragma HLS ARRAY_PARTITION variable=W2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=W2 complete dim=3
#pragma HLS ARRAY_PARTITION variable=W2 complete dim=4
#pragma HLS ARRAY_PARTITION variable=W3 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=W3 complete dim=3
#pragma HLS ARRAY_PARTITION variable=W3 complete dim=4

    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++) {
#pragma HLS PIPELINE II=1
            a0[0][y][x] = img[y][x];
        }

    conv_relu_pool<1, 16, 64, 64>(a0, a1, W0, B0, M0, S0);
    conv_relu_pool<16, 32, 32, 32>(a1, a2, W1, B1, M1, S1);
    conv_relu_pool<32, 64, 16, 16>(a2, a3, W2, B2, M2, S2);
    conv_relu_pool<64, 64, 8, 8>(a3, a4, W3, B3, M3, S3);

    // global average pooling (sum; the 1/16 is folded into the FC scale) + fully connected
    int32_t g[64];
GAP:
    for (int c = 0; c < 64; c++) {
#pragma HLS PIPELINE II=1
        int32_t sum = 0;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) sum += a4[c][i][j];
        g[c] = sum;
    }
FC:
    for (int k = 0; k < N_OUT; k++) {
        int32_t acc = FCB[k];
        for (int c = 0; c < 64; c++) {
#pragma HLS PIPELINE II=1
            acc += (int32_t)FCW[k][c] * g[c];
        }
        out[k] = acc;
    }
}

void cnn_top(const uint8_t *images, int32_t *logits, int n_images) {
#pragma HLS INTERFACE m_axi port=images offset=slave bundle=gmem0 depth=4096 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=logits offset=slave bundle=gmem1 depth=24
#pragma HLS INTERFACE s_axilite port=images
#pragma HLS INTERFACE s_axilite port=logits
#pragma HLS INTERFACE s_axilite port=n_images
#pragma HLS INTERFACE s_axilite port=return

IMAGES:
    for (int n = 0; n < n_images; n++) {
        uint8_t img[IMG_H][IMG_W];
        int32_t out[N_OUT];
    LOAD:
        for (int i = 0; i < IMG_H * IMG_W; i++) {
#pragma HLS PIPELINE II=1
            img[i / IMG_W][i % IMG_W] = images[(long)n * IMG_H * IMG_W + i];
        }
        classify_one(img, out);
    STORE:
        for (int k = 0; k < N_OUT; k++) {
#pragma HLS PIPELINE II=1
            logits[(long)n * N_OUT + k] = out[k];
        }
    }
}
