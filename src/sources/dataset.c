#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dataset.h"

void dataset_init(Dataset *d, size_t inputSize, size_t outputSize)
{
    d->inputSize = inputSize;
    d->outputSize = outputSize;
    d->entries = NULL;
    d->entryCount = 0;
    d->entryMaxCount = 0;
    d->decode = NULL;
}

void dataset_set_data_decoder(Dataset *d, decoder_t decode)
{
    d->decode = decode;
}

int dataset_add_entry(Dataset *d, const void *inData, const void *outData, size_t inSize, size_t outSize)
{
    if (d->entryCount == d->entryMaxCount)
    {
        // Use a quadratic growth scheme for the entry buffer.

        size_t newSize = (d->entryMaxCount == 0) ? 16
            : d->entryMaxCount + (int)sqrt(d->entryMaxCount) * 2;
        void *newPtr = realloc(d->entries, sizeof(DatasetEntry) * newSize);

        if (newPtr == NULL)
        {
            perror("dataset_add_entry(): error");
            return (-1);
        }

        d->entryMaxCount = newSize;
        d->entries = newPtr;
    }

    DatasetEntry *cur = d->entries + d->entryCount;

    cur->inSize = (d->decode == NULL) ? sizeof(weight_t) * d->inputSize : inSize;
    cur->outSize = (d->decode == NULL) ? sizeof(weight_t) * d->outputSize : outSize;
    cur->inData = malloc(cur->inSize);
    cur->outData = malloc(cur->outSize);

    if (cur->inData == NULL || cur->outData == NULL)
    {
        perror("dataset_add_entry(): error");
        free(cur->inData);
        free(cur->outData);
        return (-1);
    }

    memcpy(cur->inData, inData, cur->inSize);
    memcpy(cur->outData, outData, cur->outSize);
    d->entryCount++;
    return (0);
}

int dataset_push_entries(Dataset *d, const char *filename)
{
    FILE *f = fopen(filename, "ab");

    if (f == NULL)
    {
        perror("dataset_push_entries(): unable to open file");
        return (-1);
    }

    for (size_t i = 0; i < d->entryCount; ++d)
    {
        DatasetEntry *cur = d->entries + i;

        if (fwrite(&cur->inSize, sizeof(size_t), 1, f) != 1
            || fwrite(&cur->outSize, sizeof(size_t), 1, f) != 1
            || fwrite(cur->inData, 1, cur->inSize, f) != cur->inSize
            || fwrite(cur->outData, 1, cur->outSize, f) != cur->outSize)
        {
            perror("dataset_push_entries(): unable to write entry to file");
            fclose(f);
            return (-2);
        }
    }

    for (size_t i = 0; i < d->entryCount; ++i)
    {
        free(d->entries[i].inData);
        free(d->entries[i].outData);
    }
    free(d->entries);

    d->entries = NULL;
    d->entryCount = 0;
    d->entryMaxCount = 0;
    fclose(f);
    return (0);
}

void dataset_destroy(Dataset *d)
{
    for (size_t i = 0; i < d->entryCount; ++i)
    {
        free(d->entries[i].inData);
        free(d->entries[i].outData);
    }
    free(d->entries);
    d->inputSize = 0;
    d->outputSize = 0;
    d->entries = NULL;
    d->entryCount = 0;
    d->entryMaxCount = 0;
    d->decode = NULL;
}
