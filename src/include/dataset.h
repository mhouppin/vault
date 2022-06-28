#ifndef DATASET_H
#define DATASET_H

#include <stddef.h>
#include "weight.h"

typedef struct _DatasetEntry
{
    void *inData;
    void *outData;
    size_t inSize;
    size_t outSize;
}
DatasetEntry;

typedef void (*decoder_t)(const DatasetEntry *, weight_t *, weight_t *);

typedef struct _Dataset
{
    size_t inputSize;
    size_t outputSize;
    DatasetEntry *entries;
    size_t entryCount;
    size_t entryMaxCount;
    decoder_t decode;
}
Dataset;

// Initializes a new dataset with the given parameters.
void dataset_init(Dataset *d, size_t inputSize, size_t outputSize);

// Sets the decoding function for all entries.
void dataset_set_data_decoder(Dataset *d, decoder_t decode);

// Adds a new entry to the dataset. If no decoding function is set, inSize
// and outSize will be ignored. Returns zero if the entry has been
// successfully added, non-zero integer otherwise.
int dataset_add_entry(Dataset *d, const void *inData, const void *outData, size_t inSize, size_t outSize);

// Writes the current entries from the dataset to the given file, and empties
// the entry buffer. Returns zero if successful, non-zero integer otherwise.
// (For error returns less than or equal to -2, the errors are not recoverable
// for now, this behavior might be fixed in the future.)
// The binary format of the file might not be suitable for transfer between
// different systems.
int dataset_push_entries(Dataset *d, const char *filename);

// Frees all memory allocated by the dataset and resets it to an unused state.
void dataset_destroy(Dataset *d);

#endif
