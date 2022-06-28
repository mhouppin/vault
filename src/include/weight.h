#ifndef WEIGHT_H
#define WEIGHT_H

#include <stdio.h>
#include <stdint.h>

#define WG_PREC 21

#define WG_ONE ((weight_t)1 << WG_PREC)

typedef int32_t weight_t;

// Loads a single weight from the given file.
int wload(FILE *fp, weight_t *w);

// Saves a single weight from the given file.
int wsave(FILE *fp, weight_t w);

// Computes (min + (max - min) * rate), with rate being a fixed point
// integer in the [0, 1) interval.
static inline weight_t wrate(weight_t minValue, weight_t maxValue, uint16_t rate) {
    return minValue + (weight_t)((int64_t)(maxValue - minValue) * (int64_t)rate / 65536);
}

static inline double wnormalize(weight_t value) {
    return (double)value / (double)WG_ONE;
}

#endif
