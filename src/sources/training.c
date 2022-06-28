#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "training.h"

typedef struct _NN_Worker
{
    pthread_t thread;
    const Network *nn;
    const weight_t *inputArray;
    const weight_t *outputArray;
    size_t entryCount;
    size_t totalLayerSize;
    size_t totalWeightSize;
    weight_t *entryInput;
    weight_t *cpuBuffer;
    weight_t *nValues;
    weight_t *error;
    weight_t *gradient;
}
NN_Worker;

typedef struct _NN_Allocator
{
    size_t maxInputSize;
    size_t maxOutputSize;
    char *tempInputDecoder;
    char *tempOutputDecoder;
    weight_t *batchInputMemory;
    weight_t *batchOutputMemory;
}
NN_Allocator;

void *nn_worker_thread(void *ptr)
{
    NN_Worker *worker = ptr;
    const Network *nn = worker->nn;
    const size_t nnInputSize = nn->layerSizes[0];
    const size_t nnOutputSize = nn->layerSizes[nn->layers];

    // Reset the gradient for all neurons in the network.

    for (size_t o = 0; o < worker->totalWeightSize; ++o)
        worker->gradient[o] = 0.0;

    for (size_t entryIdx = 0; entryIdx < worker->entryCount; ++entryIdx)
    {
        const weight_t *curEntryInput = worker->inputArray + entryIdx * nnInputSize;
        const weight_t *curEntryOutput = worker->outputArray + entryIdx * nnOutputSize;

        // Here we basically do as in the nn_compute() function, but we keep
        // all hidden neuron values for backpropagation.

        memcpy(worker->entryInput, curEntryInput, nnInputSize * sizeof(weight_t));

        // Save the input values in the nValues buffer since we will overwrite them
        // in the entryInput buffer after the first inference.

        memcpy(worker->nValues, worker->entryInput, nnInputSize * sizeof(weight_t));

        // Keep track of the offset in the nValues buffer.

        size_t nOffset = nnInputSize;

        for (size_t l = 0; l < nn->layers; ++l)
        {
            // Preload some constant values to simplify further calculations.

            const size_t inputSize = nn->layerSizes[l];
            const size_t outputSize = nn->layerSizes[l + 1];
            weight_t *const weights = nn->weights + nn->layerOffsets[l];

            wforwardprop(worker->cpuBuffer, worker->entryInput, weights, outputSize, inputSize);

            // Save the neuron values prior to the activation function and
            // adjust the nValues offset.

            memcpy(worker->nValues + nOffset, worker->cpuBuffer, sizeof(weight_t) * outputSize);
            nOffset += outputSize;

            // Then apply the activation function and pass the data back in the
            // entryInput buffer for the next layer.

            nn->activations[l](worker->cpuBuffer, worker->entryInput, outputSize);
        }

        // Get the error for the output layer of the network.

        for (size_t outputIdx = 0; outputIdx < nnOutputSize; ++outputIdx)
            worker->cpuBuffer[outputIdx] = worker->entryInput[outputIdx] - curEntryOutput[outputIdx];

        nOffset -= nnOutputSize;
        nn->derivatives[nn->layers - 1](worker->nValues + nOffset, worker->error + nOffset, nnOutputSize);

        whadamard(worker->error + nOffset, worker->cpuBuffer, nnOutputSize);

        // Then backprop the error for each layer of the network.

        for (size_t l = nn->layers - 1; l > 0; --l)
        {
            const size_t inputSize = nn->layerSizes[l];
            const size_t outputSize = nn->layerSizes[l + 1];
            weight_t *const weights = nn->weights + nn->layerOffsets[l];

            wbackprop(worker->cpuBuffer, worker->error + nOffset, weights, inputSize, outputSize);

            nOffset -= inputSize;
            nn->derivatives[l - 1](worker->nValues + nOffset, worker->error + nOffset, inputSize);

            whadamard(worker->error + nOffset, worker->cpuBuffer, inputSize);
        }

        nOffset = worker->totalLayerSize;

        // Then compute the gradient for each weight of the network.

        for (size_t l = nn->layers; l > 0; --l)
        {
            const size_t inputSize = nn->layerSizes[l - 1];
            const size_t outputSize = nn->layerSizes[l];
            weight_t *gradient = worker->gradient + nn->layerOffsets[l - 1];

            nOffset -= outputSize;

            nn->activations[l - 1](worker->nValues + nOffset - inputSize, worker->cpuBuffer, inputSize);

            wgradupdate(gradient, worker->error + nOffset, worker->cpuBuffer, inputSize, outputSize);
        }
    }

    return (NULL);
}

int nn_train_check_range(double value, const char *valueName)
{
    if (!isfinite(value))
    {
        fprintf(stderr, "nn_train(): error: invalid %s (%lg)\n", valueName, value);
        return (-1);
    }
    if (value < 0)
    {
        fprintf(stderr, "nn_train(): error: negative %s (%lg)\n", valueName, value);
        return (-1);
    }
    if (value > 1)
        fprintf(stderr, "nn_train(): warning: %s exceeds 1 (%lg)\n", valueName, value);

    return 0;
}

int nn_train(Network *nn, Dataset *d, const char *datafile, TrainParams tp, uint32_t debug)
{
    if (nn_train_check_range(tp.learningRate, "learning rate"))
        return (-1);

    if (nn_train_check_range(tp.momentum, "momentum"))
        return (-1);

    if (nn_train_check_range(tp.velocity, "velocity"))
        return (-1);

    if (tp.saveEvery != 0 && tp.nameFormat == NULL)
    {
        fprintf(stderr, "nn_train(): error: saveEvery is set to %d but no name format was given\n", tp.saveEvery);
        return (-1);
    }

    FILE *f = NULL;

    if (datafile != NULL)
    {
        f = fopen(datafile, "rb");

        if (f == NULL)
        {
            perror("nn_train(): error");
            return (-1);
        }
    }

    int ret = 0;

    if (tp.batchSize == 0)
        tp.batchSize = 1;

    if (tp.threads <= 0)
        tp.threads = 1;

    const size_t nnInputSize = nn->layerSizes[0];
    const size_t nnOutputSize = nn->layerSizes[nn->layers];

    size_t maxLayerSize = nnInputSize;
    size_t totalLayerSize = nnInputSize;
    size_t totalWeightSize = nn->layerOffsets[nn->layers - 1] + (nn->layerSizes[nn->layers - 1] + 1) * nn->layerSizes[nn->layers];

    for (size_t i = 1; i <= nn->layers; ++i)
    {
        totalLayerSize += nn->layerSizes[i];
        if (maxLayerSize < nn->layerSizes[i])
            maxLayerSize = nn->layerSizes[i];
    }

    NN_Allocator alloc = {};
    // DatasetEntry *batch = malloc(sizeof(DatasetEntry) * tp.batchSize);
    NN_Worker *workerList = malloc(sizeof(NN_Worker) * tp.threads);
    alloc.batchInputMemory = malloc(sizeof(weight_t) * nnInputSize * tp.batchSize);
    alloc.batchOutputMemory = malloc(sizeof(weight_t) * nnOutputSize * tp.batchSize);
    double *mGrad = malloc(sizeof(double) * totalWeightSize);
    double *vGrad = malloc(sizeof(double) * totalWeightSize);

    if (workerList == NULL || alloc.batchInputMemory == NULL || alloc.batchOutputMemory == NULL || mGrad == NULL || vGrad == NULL)
    {
        perror("nn_train(): error");
        ret = -2;
        goto initial_alloc_fail;
    }

    for (size_t i = 0; i < totalWeightSize; ++i)
        mGrad[i] = vGrad[i] = 0;

    for (int i = 0; i < tp.threads; ++i)
    {
        NN_Worker *cur = workerList + i;

        cur->nn = nn;
        cur->totalLayerSize = totalLayerSize;
        cur->totalWeightSize = totalWeightSize;

        cur->entryInput = malloc(sizeof(weight_t) * maxLayerSize);
        cur->error = malloc(sizeof(weight_t) * totalLayerSize);
        cur->nValues = malloc(sizeof(weight_t) * totalLayerSize);
        cur->cpuBuffer = malloc(sizeof(weight_t) * (maxLayerSize + 1));
        cur->gradient = malloc(sizeof(weight_t) * totalWeightSize);

        if (cur->entryInput == NULL || cur->error == NULL || cur->nValues == NULL || cur->cpuBuffer == NULL || cur->gradient == NULL)
        {
            perror("nn_train(): error");
            ret = -2;

            for (int k = 0; k <= i; ++k)
            {
                cur = workerList + k;

                free(cur->entryInput);
                free(cur->error);
                free(cur->nValues);
                free(cur->cpuBuffer);
                free(cur->gradient);
            }

            goto initial_alloc_fail;
        }
    }

    size_t datasetSize = d->entryCount;

    if (f != NULL)
    {
        size_t inSize, outSize;
        while (fread(&inSize, sizeof(size_t), 1, f) == 1)
        {
            if (fread(&outSize, sizeof(size_t), 1, f) != 1
                || fseek(f, (long)(inSize + outSize), SEEK_CUR) != 0)
            {
                perror("nn_train(): error");
                ret = -1;
                goto nn_allocator_or_file_fail;
            }
            ++datasetSize;

            alloc.maxInputSize = (inSize > alloc.maxInputSize) ? inSize : alloc.maxInputSize;
            alloc.maxOutputSize = (outSize > alloc.maxOutputSize) ? outSize : alloc.maxOutputSize;
        }
    }

    alloc.tempInputDecoder = malloc(alloc.maxInputSize * sizeof(weight_t));
    alloc.tempOutputDecoder = malloc(alloc.maxOutputSize * sizeof(weight_t));

    if (alloc.tempInputDecoder == NULL || alloc.tempOutputDecoder == NULL)
    {
        perror("nn_train(): error");
        ret = -2;
        free(alloc.tempInputDecoder);
        free(alloc.tempOutputDecoder);

        goto nn_allocator_or_file_fail;
    }

    size_t batchCount = (datasetSize - 1) / tp.batchSize + 1;

    if (debug & TRAIN_SHOW_CONF)
    {
        printf("Traning session parameters:\n");
        printf(" - Epochs:        %d\n", tp.epochs);
        printf(" - Learning rate: %lg\n", tp.learningRate);
        printf(" - Batch size:    %lu\n", (unsigned long)tp.batchSize);
        printf(" - Dataset size:  %lu\n", (unsigned long)datasetSize);
        printf(" - Momentum:      %lg\n", tp.momentum);
        printf(" - Velocity:      %lg\n", tp.velocity);
        printf(" - Threads:       %d\n", tp.threads);
        printf(" - Checkpoints:   ");

        if (tp.saveEvery == 0)      printf("None\n\n");
        else if (tp.saveEvery == 1) printf("Every epoch\n");
        else                        printf("Every %d epochs\n", tp.saveEvery);
        if (tp.saveEvery != 0)      printf(" - CKP format:    \"%s\"\n\n", tp.nameFormat);

        fflush(stdout);
    }

    for (int epoch = 0; epoch < tp.epochs; ++epoch)
    {
        if (f != NULL)
            rewind(f);

        if (debug & TRAIN_SHOW_EPOCH)
        {
            printf("Epoch %d/%d\n", epoch + 1, tp.epochs);
            fflush(stdout);
        }

        for (size_t batchIdx = 0; batchIdx < batchCount; ++batchIdx)
        {
            if (debug & TRAIN_SHOW_BATCH)
            {
                int p = (int)((batchIdx + 1) * 40 / batchCount);
                const char *strSharp = "########################################";
                fputc('\r', stderr);
                fflush(stderr);
                printf("[%-40.*s] Batch %lu/%lu", p, strSharp, (unsigned long)batchIdx + 1, (unsigned long)batchCount);
                fflush(stdout);
            }

            size_t batchStart = batchIdx * tp.batchSize;
            size_t batchEnd = batchStart + tp.batchSize;
            size_t batchFill = 0;

            if (batchStart < d->entryCount)
            {
                batchFill = (batchEnd <= d->entryCount) ? tp.batchSize : d->entryCount - batchStart;

                if (d->decode)
                    for (size_t i = 0; i < batchFill; ++i)
                        d->decode(d->entries + batchStart + i, alloc.batchInputMemory + i * nnInputSize, alloc.batchOutputMemory + i * nnOutputSize);
                else
                    for (size_t i = 0; i < batchFill; ++i)
                    {
                        memcpy(alloc.batchInputMemory  + i * nnInputSize,  d->entries[batchStart + i].inData,  nnInputSize  * sizeof(weight_t));
                        memcpy(alloc.batchOutputMemory + i * nnOutputSize, d->entries[batchStart + i].outData, nnOutputSize * sizeof(weight_t));
                    }
            }

            if (batchFill < tp.batchSize && f)
                while (batchFill < tp.batchSize)
                {
                    DatasetEntry tmp;

                    tmp.inData  = alloc.tempInputDecoder;
                    tmp.outData = alloc.tempOutputDecoder;

                    if (fread(&tmp.inSize, sizeof(size_t), 1, f) != 1)
                        break ;

                    if (fread(&tmp.outSize, sizeof(size_t), 1, f) != 1
                        || fread(tmp.inData, 1, tmp.inSize, f) != tmp.inSize
                        || fread(tmp.outData, 1, tmp.outSize, f) != tmp.outSize)
                    {
                        fputs("nn_train(): error: dataset file corrupted\n", stderr);
                        goto in_loop_fail;
                    }

                    if (d->decode)
                        d->decode(&tmp, alloc.batchInputMemory + batchFill * nnInputSize, alloc.batchOutputMemory + batchFill * nnOutputSize);
                    else
                    {
                        memcpy(alloc.batchInputMemory  + batchFill * nnInputSize,  tmp.inData,  nnInputSize  * sizeof(weight_t));
                        memcpy(alloc.batchOutputMemory + batchFill * nnOutputSize, tmp.outData, nnOutputSize * sizeof(weight_t));
                    }

                    ++batchFill;
                }

            for (int threadIdx = 0; threadIdx < tp.threads; ++threadIdx)
            {
                NN_Worker *cur = workerList + threadIdx;
                size_t start = batchFill * (size_t)threadIdx / tp.threads;
                size_t end = batchFill * (size_t)(threadIdx + 1) / tp.threads;

                cur->inputArray  = alloc.batchInputMemory  + start * nnInputSize;
                cur->outputArray = alloc.batchOutputMemory + start * nnOutputSize;
                cur->entryCount = end - start;

                if (threadIdx && pthread_create(&cur->thread, NULL, &nn_worker_thread, cur))
                {
                    perror("nn_train(): error");
                    ret = -2;
                    goto in_loop_fail;
                }
            }
            nn_worker_thread(workerList);

            for (int threadIdx = 1; threadIdx < tp.threads; ++threadIdx)
            {
                pthread_join(workerList[threadIdx].thread, NULL);

                // Accumulate gradients in the first worker.

                for (size_t weightIdx = 0; weightIdx < totalWeightSize; ++weightIdx)
                    workerList->gradient[weightIdx] += workerList[threadIdx].gradient[weightIdx];
            }

            // Update all the weights in the network.

            for (size_t weightIdx = 0; weightIdx < totalWeightSize; ++weightIdx)
            {
                weight_t grad = workerList->gradient[weightIdx] / (weight_t)batchFill;

                mGrad[weightIdx] = mGrad[weightIdx] * tp.momentum + (double)grad               * (1.0 - tp.momentum);
                vGrad[weightIdx] = vGrad[weightIdx] * tp.velocity + pow(wnormalize(grad), 2.0) * (1.0 - tp.velocity);

                nn->weights[weightIdx] -= mGrad[weightIdx] * tp.learningRate / sqrt(vGrad[weightIdx] + 1e-8);
            }

            if (tp.callbackAfterBatch != NULL)
                tp.callbackAfterBatch(nn, d, tp.callbackUserData);
        }

        if (debug & TRAIN_SHOW_BATCH)
        {
            putchar('\n');
            fflush(stdout);
        }

        if (tp.callbackAfterEpoch != NULL)
            tp.callbackAfterEpoch(nn, d, tp.callbackUserData);

        if (debug & TRAIN_SHOW_LOSS)
        {
            double totalLoss = 0.0;

            for (size_t i = 0; i < d->entryCount; ++i)
            {
                DatasetEntry *cur = d->entries + i;

                if (d->decode == NULL)
                {
                    memcpy(workerList->entryInput, cur->inData, nnInputSize * sizeof(weight_t));
                    memcpy(alloc.batchOutputMemory, cur->outData, nnOutputSize * sizeof(weight_t));
                }
                else
                    d->decode(cur, workerList->entryInput, alloc.batchOutputMemory);

                nn_compute(nn, workerList->entryInput, workerList->cpuBuffer);
                for (size_t o = 0; o < nnOutputSize; ++o)
                {
                    weight_t t = alloc.batchOutputMemory[o];
                    weight_t p = workerList->cpuBuffer[o];
                    totalLoss += pow(wnormalize(p - t), 2);
                }
            }

            printf("Current loss: [%lg]\n", totalLoss / d->entryCount);
            fflush(stdout);
        }

        if (tp.saveEvery && epoch % tp.saveEvery == tp.saveEvery - 1)
        {
            char filename[4096];

            sprintf(filename, tp.nameFormat, epoch + 1);

            if (debug & TRAIN_SHOW_SAVES)
            {
                printf("Saving network to '%s'\n", filename);
                fflush(stdout);
            }
            nn_save(nn, filename);
        }
    }

in_loop_fail:

nn_allocator_or_file_fail:

    for (int i = 0; i < tp.threads; ++i)
    {
        NN_Worker *cur = workerList + i;

        free(cur->entryInput);
        free(cur->error);
        free(cur->nValues);
        free(cur->cpuBuffer);
        free(cur->gradient);
    }

initial_alloc_fail:

    free(alloc.batchInputMemory);
    free(alloc.batchOutputMemory);
    free(workerList);
    free(mGrad);
    free(vGrad);
    if (f != NULL) fclose(f);
    return (ret);
}
