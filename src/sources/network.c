#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "activation.h"
#include "matrix.h"
#include "network.h"

// Network file structure:
// Layer count
// Neurons per layer
// Activation function IDs
// Weights

int integer_load(FILE *fp, uint32_t *u)
{
    unsigned char buffer[4];

    if (fread(buffer, 1, 4, fp) != 4)
        return -1;

    *u = ((uint32_t)buffer[0] <<  0)
       | ((uint32_t)buffer[1] <<  8)
       | ((uint32_t)buffer[2] << 16)
       | ((uint32_t)buffer[3] << 24);

    return 0;
}

int integer_save(FILE *fp, uint32_t u)
{
    fputc((u >>  0) & 0xFF, fp);
    fputc((u >>  8) & 0xFF, fp);
    fputc((u >> 16) & 0xFF, fp);
    fputc((u >> 24) & 0xFF, fp);

    return ferror(fp);
}

void nn_load_alloc_fail(const char *filename, const char *varName)
{
    fprintf(stderr, "nn_load(\"%s\"): Unable to allocate %s data: %s\n",
        filename, varName, strerror(ENOMEM));
}

int nn_allocate_cpu_buffers(Network *nn)
{
    size_t maxLayerSize = nn->layerSizes[0];

    for (size_t i = 1; i <= nn->layers; ++i)
        if (maxLayerSize < nn->layerSizes[i])
            maxLayerSize = nn->layerSizes[i];

    nn->cpuInput = malloc(sizeof(weight_t) * maxLayerSize);
    nn->cpuOutput = malloc(sizeof(weight_t) * maxLayerSize);

    if (nn->cpuInput == NULL || nn->cpuOutput == NULL)
        return (-1);
    return (0);
}

void nn_destroy(Network *nn)
{
    free(nn->layerSizes);
    free(nn->weights);
    free(nn->layerOffsets);
    free(nn->activationIds);
    free(nn->activations);
    free(nn->derivatives);
    free(nn->cpuInput);
    free(nn->cpuOutput);
    memset(nn, 0, sizeof(Network));
}

int nn_create(Network *nn, size_t layers, const size_t layerSizes[], int activationIds[])
{
    memset(nn, 0, sizeof(Network));

    nn->layers = layers;

    nn->layerSizes = malloc(sizeof(size_t) * (nn->layers + 1));
    if (nn->layerSizes == NULL)
    {
        perror("nn_create(): Unable to allocate layer size data");
        goto create_error;
    }

    nn->activationIds = malloc(sizeof(int) * nn->layers);
    nn->activations = malloc(sizeof(Activation) * nn->layers);
    nn->derivatives = malloc(sizeof(Activation) * nn->layers);
    if (nn->activationIds == NULL || nn->activations == NULL || nn->derivatives == NULL)
    {
        perror("nn_create(): Unable to allocate activation function data");
        goto create_error;
    }

    nn->layerOffsets = malloc(sizeof(size_t) * nn->layers);
    if (nn->layerOffsets == NULL)
    {
        perror("nn_create(): Unable to allocate layer offset data");
        goto create_error;
    }

    for (size_t i = 0; i < nn->layers + 1; ++i)
    {
        if (layerSizes[i] == 0)
        {
            fprintf(stderr, "nn_create(): Layer %"PRIu32 " is zero-sized\n", (uint32_t)i);
            goto create_error;
        }
        nn->layerSizes[i] = layerSizes[i];
    }

    nn->layerOffsets[0] = 0;
    for (size_t i = 1; i < nn->layers; ++i)
        nn->layerOffsets[i] = nn->layerOffsets[i - 1] + (nn->layerSizes[i - 1] + 1) * nn->layerSizes[i];

    for (size_t i = 0; i < nn->layers; ++i)
    {
        if (activationIds[i] > ACTIVATION_COUNT || activationIds[i] < 0)
        {
            fprintf(stderr, "nn_create(): Activation id %d doesn't exist\n", activationIds[i]);
            goto create_error;
        }
        nn->activationIds[i] = activationIds[i];
        nn->activations[i] = ActivationList[activationIds[i]].function;
        nn->derivatives[i] = ActivationList[activationIds[i]].derivative;
    }

    if (nn_allocate_cpu_buffers(nn))
    {
        perror("nn_create(): Unable to allocate internal buffers");
        goto create_error;
    }

    size_t weightCount = nn->layerOffsets[nn->layers - 1]
        + (nn->layerSizes[nn->layers - 1] + 1) * nn->layerSizes[nn->layers];

    nn->weights = malloc(sizeof(weight_t) * weightCount);
    if (nn->weights == NULL)
    {
        perror("nn_create(): Unable to allocate weights data");
        goto create_error;
    }

    for (size_t i = 0; i < weightCount; ++i)
        nn->weights[i] = 0;

    return (0);

create_error:
    nn_destroy(nn);
    return -1;
}

int nn_load(Network *nn, const char *filename)
{
    memset(nn, 0, sizeof(Network));
    FILE *f = fopen(filename, "rb");

    if (f == NULL)
        return -1;

    uint32_t sizeReader;

    if (integer_load(f, &sizeReader))
    {
        fprintf(stderr, "nn_load(\"%s\"): Unable to read layer count variable\n",
            filename);
        goto load_error;
    }

    nn->layers = sizeReader;

    nn->layerSizes = malloc(sizeof(size_t) * (nn->layers + 1));
    if (nn->layerSizes == NULL)
    {
        nn_load_alloc_fail(filename, "layer size");
        goto load_error;
    }

    nn->activationIds = malloc(sizeof(int) * nn->layers);
    nn->activations = malloc(sizeof(Activation) * nn->layers);
    nn->derivatives = malloc(sizeof(Activation) * nn->layers);
    if (nn->activationIds == NULL || nn->activations == NULL || nn->derivatives == NULL)
    {
        nn_load_alloc_fail(filename, "activation function");
        goto load_error;
    }

    nn->layerOffsets = malloc(sizeof(size_t) * nn->layers);
    if (nn->layerOffsets == NULL)
    {
        nn_load_alloc_fail(filename, "layer offset");
        goto load_error;
    }

    for (size_t i = 0; i < nn->layers + 1; ++i)
    {
        if (integer_load(f, &sizeReader))
        {
            fprintf(stderr, "nn_load(\"%s\"): Unable to read layer %"PRIu32 " size\n",
                filename, (uint32_t)i);
            goto load_error;
        }
        if (sizeReader == 0)
        {
            fprintf(stderr, "nn_load(\"%s\"): Layer %"PRIu32 " is zero-sized\n",
                filename, (uint32_t)i);
            goto load_error;
        }
        nn->layerSizes[i] = sizeReader;
    }

    nn->layerOffsets[0] = 0;
    for (size_t i = 1; i < nn->layers; ++i)
        nn->layerOffsets[i] = nn->layerOffsets[i - 1] + (nn->layerSizes[i - 1] + 1) * nn->layerSizes[i];

    for (size_t i = 0; i < nn->layers; ++i)
    {
        if (integer_load(f, &sizeReader))
        {
            fprintf(stderr, "nn_load(\"%s\"): Unable to read layer %"PRIu32 " activation id\n",
                filename, (uint32_t)i);
            goto load_error;
        }

        nn->activationIds[i] = (int)(int32_t)sizeReader;

        if (nn->activationIds[i] > ACTIVATION_COUNT || nn->activationIds[i] < 0)
        {
            fprintf(stderr, "nn_load(\"%s\"): Activation id %d doesn't exist\n",
                filename, nn->activationIds[i]);
            goto load_error;
        }
        nn->activations[i] = ActivationList[nn->activationIds[i]].function;
        nn->derivatives[i] = ActivationList[nn->activationIds[i]].derivative;
    }

    if (nn_allocate_cpu_buffers(nn))
    {
        nn_load_alloc_fail(filename, "internal buffers");
        goto load_error;
    }

    size_t weightCount = nn->layerOffsets[nn->layers - 1]
        + (nn->layerSizes[nn->layers - 1] + 1) * nn->layerSizes[nn->layers];

    nn->weights = malloc(sizeof(weight_t) * weightCount);
    if (nn->weights == NULL)
    {
        nn_load_alloc_fail(filename, "weights");
        goto load_error;
    }

    for (size_t i = 0; i < weightCount; ++i)
    {
        if (wload(f, &nn->weights[i]))
        {
            size_t l;
            for (l = 1; nn->layerOffsets[l] <= i && l < nn->layers; ++l);
            --l;

            size_t inputIndex = (i - nn->layerOffsets[l]) / nn->layerSizes[l];
            size_t outputIndex = (i - nn->layerOffsets[l]) % nn->layerSizes[l];

            fprintf(stderr, "nn_load(\"%s\"): Unable to read layer %"PRIu32 " weight (%"PRIu32 ", %"PRIu32 ")\n",
                filename, (uint32_t)l, (uint32_t)inputIndex, (uint32_t)outputIndex);
            goto load_error;
        }
    }

    char c;
    if (fread(&c, 1, 1, f) == 1)
        fprintf(stderr, "nn_load(\"%s\"): Warning: garbage data after final weight (value %d)\n", filename, (int)c);

    fclose(f);
    return 0;

load_error:
    nn_destroy(nn);
    fclose(f);
    return -1;
}

int nn_save(Network *nn, const char *filename)
{
    FILE *f = fopen(filename, "wb");

    if (f == NULL)
        return -1;

    if (integer_save(f, (uint32_t)nn->layers))
    {
        fprintf(stderr, "nn_save(\"%s\"): Unable to write layer count variable\n",
            filename);
        goto save_error;
    }

    for (size_t i = 0; i < nn->layers + 1; ++i)
    {
        if (integer_save(f, (uint32_t)nn->layerSizes[i]))
        {
            fprintf(stderr, "nn_save(\"%s\"): Unable to write layer %"PRIu32 " size\n",
                filename, (uint32_t)i);
            goto save_error;
        }
    }

    for (size_t i = 0; i < nn->layers; ++i)
    {
        if (integer_save(f, (uint32_t)(int32_t)nn->activationIds[i]))
        {
            fprintf(stderr, "nn_save(\"%s\"): Unable to write layer %"PRIu32 " activation id\n",
                filename, (uint32_t)i);
            goto save_error;
        }
    }

    for (size_t l = 0; l < nn->layers; ++l)
    {
        const size_t inputSize = nn->layerSizes[l] + 1;
        const size_t outputSize = nn->layerSizes[l + 1];

        for (size_t n = 0; n < inputSize; ++n)
        {
            for (size_t w = 0; w < outputSize; ++w)
            {
                if (wsave(f, nn->weights[nn->layerOffsets[l] + outputSize * n + w]))
                {
                    fprintf(stderr, "nn_save(\"%s\"): Unable to write layer %"PRIu32 " weight (%"PRIu32 ", %"PRIu32 ")\n",
                        filename, (uint32_t)l, (uint32_t)n, (uint32_t)w);
                    goto save_error;
                }
            }
        }
    }

    fclose(f);
    return 0;

save_error:
    fclose(f);
    return -1;
}

void nn_set_layer_activation(Network *nn, size_t layer, int activationId)
{
    nn->activationIds[layer] = activationId;
    nn->activations[layer] = ActivationList[activationId].function;
    nn->derivatives[layer] = ActivationList[activationId].derivative;
}

void nn_compute(Network *restrict nn, const weight_t *restrict inputs,
    weight_t *restrict outputs)
{
    memcpy(nn->cpuInput, inputs, sizeof(weight_t) * nn->layerSizes[0]);

    for (size_t l = 0; l < nn->layers; ++l)
    {
        // Preload some constant values to simplify further calculations

        const size_t inputSize = nn->layerSizes[l];
        const size_t outputSize = nn->layerSizes[l + 1];
        weight_t *const weights = nn->weights + nn->layerOffsets[l];

        wforwardprop(nn->cpuOutput, nn->cpuInput, weights, outputSize, inputSize);

        // Then apply the activation function and pass the data back in the
        // input buffer for the next layer.

        nn->activations[l](nn->cpuOutput, nn->cpuInput, outputSize);
    }

    // Copy back the data from the input buffer to the user-given array.

    memcpy(outputs, nn->cpuInput, sizeof(weight_t) * nn->layerSizes[nn->layers]);
}

void nn_const_compute(const Network *restrict nn, weight_t *restrict ioBuffer,
    weight_t *restrict cpuBuffer)
{
    for (size_t l = 0; l < nn->layers; ++l)
    {
        // Preload some constant values to simplify further calculations

        const size_t inputSize = nn->layerSizes[l];
        const size_t outputSize = nn->layerSizes[l + 1];
        weight_t *const weights = nn->weights + nn->layerOffsets[l];

        wforwardprop(cpuBuffer, ioBuffer, weights, outputSize, inputSize);

        // Then apply the activation function and pass the data back in the
        // I/O buffer for the next layer.

        nn->activations[l](cpuBuffer, ioBuffer, outputSize);
    }
}

void nn_init_all_weights(Network *nn, weight_t minValue, weight_t maxValue, int seed)
{
    for (size_t l = 0; l < nn->layers; ++l)
        nn_init_layer_weights(nn, minValue, maxValue, seed, l);
}

void nn_init_layer_weights(Network *nn, weight_t minValue, weight_t maxValue, int seed, size_t layer)
{
    // This function internally uses a Xorshift implementation to yield
    // unsigned 16-bit values, then converted into the requested range.
    uint64_t state = (uint64_t)(unsigned int)seed + ((uint64_t)layer * UINT32_MAX);

    if (state == 0) state = 1;

    const size_t inputSize = nn->layerSizes[layer] + 1;
    const size_t outputSize = nn->layerSizes[layer + 1];

    for (size_t n = 0; n < inputSize; ++n)
    {
        for (size_t w = 0; w < outputSize; ++w)
        {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            uint16_t index = (uint16_t)state;

            nn->weights[nn->layerOffsets[layer] + outputSize * n + w] = wrate(minValue, maxValue, index);
        }
    }
}
