#ifndef TRAINING_H
#define TRAINING_H

#include <stdint.h>
#include "dataset.h"
#include "network.h"

enum
{
    TRAIN_SHOW_CONF  = 1 << 0,
    TRAIN_SHOW_EPOCH = 1 << 1,
    TRAIN_SHOW_BATCH = 1 << 2,
    TRAIN_SHOW_LOSS  = 1 << 3,
    TRAIN_SHOW_TIME  = 1 << 4,
    TRAIN_SHOW_SAVES = 1 << 5,
    TRAIN_SHOW_ALL   = (1 << 6) - 1,
};

typedef struct _TrainParams
{
    int epochs;
    double learningRate;
    size_t batchSize;
    double momentum;
    double velocity;
    int threads;
    int saveEvery;
    const char *nameFormat;
    void (*callbackAfterEpoch)(Network *, Dataset *, void *);
    void (*callbackAfterBatch)(Network *, Dataset *, void *);
    void *callbackUserData;
}
TrainParams;

#define NN_TP_DEFAULT ((TrainParams){100, 0.001, 1, 0.9, 0.999, 1, 1, "network_%03d.nn", NULL, NULL, NULL})

int nn_train(Network *nn, Dataset *d, const char *datafile, TrainParams tp, uint32_t debug);

#endif
