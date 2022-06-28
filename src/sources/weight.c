#include "weight.h"

int integer_load(FILE *fp, uint32_t *u);
int integer_save(FILE *fp, uint32_t u);

int wload(FILE *fp, weight_t *w)
{
    return integer_load(fp, (uint32_t *)w);
}

int wsave(FILE *fp, weight_t w)
{
    return integer_save(fp, (uint32_t)w);
}