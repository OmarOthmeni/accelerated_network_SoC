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
//
// Design (v2). Parallelism is a compile-time choice (PO1..PO4 below):
//   default  PO = 1/1/1/1 : low power, ~1.4 M cycles/image (36 images/s at 50 MHz)
//   -DPO1=4 -DPO2=8 -DPO3=8 -DPO4=4 : high throughput, 223 k cycles/image (448 images/s
//                                     at 100 MHz), 192 DSP / 51 k LUT (fills a Zynq-7020)
// - Each conv layer is one flat II=1 pipeline over (pooled pixel, 2x2 sub-pixel, input
//   channel), so the pipeline never drains between pixels (v1 drained after every
//   sub-pixel, which made layer 2 alone take 53% of the latency).
// - Parallel output channels (PO) are chosen per layer so the 4 layers take about the
//   same number of cycles: 1 MAC row of 9 multipliers per output channel.
// - Requantization (a 32x32-bit multiply) is time-shared: one unit per layer requantizes
//   the PO accumulators of the previous sub-pixel one per cycle, while the next
//   sub-pixel accumulates (needs CIN >= PO). Layer 1 (CIN = 1) requantizes in parallel.
// - Weights are reshaped so the 9 taps of a (co, ci) pair are one 72-bit ROM word.

#include "cnn.h"
#include "weights.h"

// parallel output channels per layer (MACs per cycle = PO * 9); override with -DPOn=...
#ifndef PO1
#define PO1 1
#endif
#ifndef PO2
#define PO2 1
#endif
#ifndef PO3
#define PO3 1
#endif
#ifndef PO4
#define PO4 1
#endif

static inline uint8_t requant(int32_t acc, int32_t m, uint8_t s) {
#pragma HLS INLINE
    int64_t r = ((int64_t)acc * (int64_t)m + ((int64_t)1 << (s - 1))) >> s;
    return r < 0 ? 0 : (r > 255 ? 255 : (uint8_t)r);
}

// 3x3 window of channel ci centred on (y, x), zero padding
template <int CIN, int H, int W>
static inline void window(const uint8_t in[CIN][H][W], int ci, int y, int x, uint8_t win[3][3]) {
#pragma HLS INLINE
    for (int ky = 0; ky < 3; ky++)
        for (int kx = 0; kx < 3; kx++) {
            const int yy = y + ky - 1, xx = x + kx - 1;
            win[ky][kx] = (yy >= 0 && yy < H && xx >= 0 && xx < W) ? in[ci][yy][xx] : 0;
        }
}

// 9-tap dot product; FABRIC=true maps the multipliers to LUTs instead of DSP slices
template <bool FABRIC>
static inline int32_t dot9(const int8_t k[3][3], const uint8_t win[3][3]) {
#pragma HLS INLINE
    int32_t sum = 0;
    for (int ky = 0; ky < 3; ky++)
        for (int kx = 0; kx < 3; kx++) {
            int32_t prod;
            if (FABRIC) {
                prod = (int32_t)k[ky][kx] * (int32_t)win[ky][kx];
#pragma HLS BIND_OP variable=prod op=mul impl=fabric
            } else {
                prod = (int32_t)k[ky][kx] * (int32_t)win[ky][kx];
            }
            sum += prod;
        }
    return sum;
}

// Layer 1 (CIN = 1): one sub-pixel per cycle, PO channels requantized in parallel.
template <int COUT, int H, int W, int PO, bool FABRIC>
static void conv1_relu_pool(const uint8_t in[1][H][W], uint8_t out[COUT][H / 2][W / 2],
                            const int8_t w[COUT][1][3][3], const int32_t b[COUT],
                            const int32_t m[COUT], const uint8_t s[COUT]) {
    constexpr int WP = W / 2, NPIX = (H / 2) * (W / 2);
CO_TILE:
    for (int co0 = 0; co0 < COUT; co0 += PO) {
        uint8_t best[PO];
#pragma HLS ARRAY_PARTITION variable=best complete
    PIX_SUB:
        for (int it = 0; it < NPIX * 4; it++) {
#pragma HLS PIPELINE II=1
            const int pix = it >> 2, sub = it & 3;
            const int py = pix / WP, px = pix % WP;
            uint8_t win[3][3];
#pragma HLS ARRAY_PARTITION variable=win complete dim=0
            window<1, H, W>(in, 0, 2 * py + (sub >> 1), 2 * px + (sub & 1), win);
            for (int p = 0; p < PO; p++) {
                const int32_t acc = b[co0 + p] + dot9<FABRIC>(w[co0 + p][0], win);
                const uint8_t v = requant(acc, m[co0 + p], s[co0 + p]);
                const uint8_t prev = (sub == 0) ? (uint8_t)0 : best[p];
                best[p] = v > prev ? v : prev;
                if (sub == 3) out[co0 + p][py][px] = best[p];
            }
        }
    }
}

// Layers 2-4 (CIN >= PO): flat pipeline over (pixel, sub-pixel, input channel); the
// accumulators of a finished sub-pixel are held and requantized one channel per cycle
// during the first PO cycles of the next sub-pixel (plus a PO-cycle drain at the end).
template <int CIN, int COUT, int H, int W, int PO, bool FABRIC>
static void conv_relu_pool(const uint8_t in[CIN][H][W], uint8_t out[COUT][H / 2][W / 2],
                           const int8_t w[COUT][CIN][3][3], const int32_t b[COUT],
                           const int32_t m[COUT], const uint8_t s[COUT]) {
    static_assert(CIN >= PO, "time-shared requantization needs CIN >= PO");
    constexpr int WP = W / 2, NPIX = (H / 2) * (W / 2), TOTAL = NPIX * 4 * CIN;
CO_TILE:
    for (int co0 = 0; co0 < COUT; co0 += PO) {
        int32_t acc[PO], hold[PO];
        uint8_t best[PO];
#pragma HLS ARRAY_PARTITION variable=acc complete
#pragma HLS ARRAY_PARTITION variable=hold complete
#pragma HLS ARRAY_PARTITION variable=best complete
        int ci = 0, sub = 0, pix = 0;  // current MAC position
        int hsub = 0, hpix = 0;        // sub-pixel whose accumulators are in hold[]
        bool held = false;
    FLAT:
        for (int it = 0; it < TOTAL + PO; it++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=hold inter false
#pragma HLS DEPENDENCE variable=best inter false
            // 1) requantize one held channel (reads hold[] before it is overwritten below)
            const int r = (it < TOTAL) ? ci : it - TOTAL;
            if (held && r < PO) {
                const uint8_t v = requant(hold[r], m[co0 + r], s[co0 + r]);
                const uint8_t prev = (hsub == 0) ? (uint8_t)0 : best[r];
                const uint8_t nb = v > prev ? v : prev;
                best[r] = nb;
                if (hsub == 3) out[co0 + r][hpix / WP][hpix % WP] = nb;
            }
            // 2) one input channel of the current sub-pixel for PO output channels
            if (it < TOTAL) {
                const int py = pix / WP, px = pix % WP;
                uint8_t win[3][3];
#pragma HLS ARRAY_PARTITION variable=win complete dim=0
                window<CIN, H, W>(in, ci, 2 * py + (sub >> 1), 2 * px + (sub & 1), win);
                for (int p = 0; p < PO; p++) {
                    const int32_t a = ((ci == 0) ? b[co0 + p] : acc[p]) + dot9<FABRIC>(w[co0 + p][ci], win);
                    acc[p] = a;
                    if (ci == CIN - 1) hold[p] = a;
                }
                if (ci == CIN - 1) {
                    held = true;
                    hsub = sub;
                    hpix = pix;
                    ci = 0;
                    if (sub == 3) {
                        sub = 0;
                        pix++;
                    } else {
                        sub++;
                    }
                } else {
                    ci++;
                }
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
// 3x3 windows: rows/columns y-1, y, y+1 fall in different banks. Factor 4 (not 3) because
// every map size is a power of two: bank = index & 3 and address = index >> 2, instead of
// /3 and %3 address logic on 9 reads per cycle.
#pragma HLS ARRAY_PARTITION variable=a0 cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=a0 cyclic factor=4 dim=3
#pragma HLS ARRAY_PARTITION variable=a1 cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=a1 cyclic factor=4 dim=3
#pragma HLS ARRAY_PARTITION variable=a2 cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=a2 cyclic factor=4 dim=3
#pragma HLS ARRAY_PARTITION variable=a3 cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=a3 cyclic factor=4 dim=3
// weights: the 9 taps of one (co, ci) pair form one ROM word; PO channels per cycle
#pragma HLS ARRAY_RESHAPE variable=W0 complete dim=3
#pragma HLS ARRAY_RESHAPE variable=W0 complete dim=4
#pragma HLS ARRAY_RESHAPE variable=W1 complete dim=3
#pragma HLS ARRAY_RESHAPE variable=W1 complete dim=4
#pragma HLS ARRAY_RESHAPE variable=W2 complete dim=3
#pragma HLS ARRAY_RESHAPE variable=W2 complete dim=4
#pragma HLS ARRAY_RESHAPE variable=W3 complete dim=3
#pragma HLS ARRAY_RESHAPE variable=W3 complete dim=4
#if PO1 > 1
// layer 1 writes and reads PO1 channels per cycle
#pragma HLS ARRAY_PARTITION variable=a1 cyclic factor=PO1 dim=1
#pragma HLS ARRAY_PARTITION variable=W0 cyclic factor=PO1 dim=1
#pragma HLS ARRAY_PARTITION variable=B0 cyclic factor=PO1 dim=1
#pragma HLS ARRAY_PARTITION variable=M0 cyclic factor=PO1 dim=1
#pragma HLS ARRAY_PARTITION variable=S0 cyclic factor=PO1 dim=1
#endif
#if PO2 > 1
#pragma HLS ARRAY_PARTITION variable=W1 cyclic factor=PO2 dim=1
#pragma HLS ARRAY_PARTITION variable=B1 cyclic factor=PO2 dim=1
#endif
#if PO3 > 1
#pragma HLS ARRAY_PARTITION variable=W2 cyclic factor=PO3 dim=1
#pragma HLS ARRAY_PARTITION variable=B2 cyclic factor=PO3 dim=1
#endif
#if PO4 > 1
#pragma HLS ARRAY_PARTITION variable=W3 cyclic factor=PO4 dim=1
#pragma HLS ARRAY_PARTITION variable=B3 cyclic factor=PO4 dim=1
#endif

    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++) {
#pragma HLS PIPELINE II=1
            a0[0][y][x] = img[y][x];
        }

    // All MACs on DSP slices (FABRIC=false): about 200 DSP and fewer LUTs, which fits the
    // Zynq-7020 better than LUT multipliers (set FABRIC=true on a layer to trade DSP for LUT).
#if PO1 > 1
    conv1_relu_pool<16, 64, 64, PO1, false>(a0, a1, W0, B0, M0, S0);
#else
    conv_relu_pool<1, 16, 64, 64, 1, false>(a0, a1, W0, B0, M0, S0);  // 1 channel: time-shared path
#endif
    conv_relu_pool<16, 32, 32, 32, PO2, false>(a1, a2, W1, B1, M1, S1);
    conv_relu_pool<32, 64, 16, 16, PO3, false>(a2, a3, W2, B2, M2, S2);
    conv_relu_pool<64, 64, 8, 8, PO4, false>(a3, a4, W3, B3, M3, S3);

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
