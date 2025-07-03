#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_SAMPLES 1000
#define FEATURES 20            // Num de colunas
#define K 5                     // numero K no KNN

typedef struct {
    double features[FEATURES];
    char label;                 // 'P' ou 'H'
} DataPoint;

void loadCSV(const char *filename);
void computeMinMax(DataPoint *data, int size, double *min, double *max);
void normalize(DataPoint *data, int size, double *min, double *max);
double euclideanDistance(DataPoint a, DataPoint b);
int compare(const void *a, const void *b);
char classify(DataPoint test_point, DataPoint *data, int size);
void evaluateModel();

