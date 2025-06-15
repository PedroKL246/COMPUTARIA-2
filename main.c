#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_SAMPLES 1000
#define FEATURES 5  // Ajuste aqui para o número de colunas de features
#define K 5         // Valor de K no KNN

typedef struct {
    double features[FEATURES];
    char label;  // 'P' ou 'H'
} DataPoint;

DataPoint dataset[MAX_SAMPLES];
int total_samples = 0;

// Funções para Normalização
void computeMinMax(DataPoint *data, int size, double *min, double *max) {
    for (int f = 0; f < FEATURES; f++) {
        min[f] = data[0].features[f];
        max[f] = data[0].features[f];
        for (int i = 1; i < size; i++) {
            if (data[i].features[f] < min[f]) min[f] = data[i].features[f];
            if (data[i].features[f] > max[f]) max[f] = data[i].features[f];
        }
        if (max[f] == min[f]) max[f] = min[f] + 1.0;  // Protege contra divisão por zero
    }
}

void normalize(DataPoint *data, int size, double *min, double *max) {
    for (int i = 0; i < size; i++) {
        for (int f = 0; f < FEATURES; f++) {
            data[i].features[f] = (data[i].features[f] - min[f]) / (max[f] - min[f]);
        }
    }
}

// Função para calcular distância Euclidiana
double euclideanDistance(DataPoint a, DataPoint b) {
    double sum = 0;
    for (int f = 0; f < FEATURES; f++) {
        sum += pow(a.features[f] - b.features[f], 2);
    }
    return sqrt(sum);
}

// Estrutura auxiliar para armazenar distâncias
typedef struct {
    double distance;
    char label;
} Neighbor;

// Comparador para qsort
int compare(const void *a, const void *b) {
    Neighbor *n1 = (Neighbor *)a;
    Neighbor *n2 = (Neighbor *)b;
    if (n1->distance < n2->distance) return -1;
    else if (n1->distance > n2->distance) return 1;
    else return 0;
}

// Função KNN
char classify(DataPoint test_point, DataPoint *data, int size) {
    Neighbor neighbors[size];

    for (int i = 0; i < size; i++) {
        neighbors[i].distance = euclideanDistance(test_point, data[i]);
        neighbors[i].label = data[i].label;
    }

    qsort(neighbors, size, sizeof(Neighbor), compare);

    int countP = 0, countH = 0;
    for (int i = 0; i < K; i++) {
        if (neighbors[i].label == 'P') countP++;
        else if (neighbors[i].label == 'H') countH++;
    }

    return (countP > countH) ? 'P' : 'H';
}

// Leitura do CSV
void loadCSV(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Erro ao abrir o arquivo.\n");
        exit(1);
    }

    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        char *token;
        token = strtok(line, ",");
        int feature_index = 0;

        while (token != NULL && feature_index < FEATURES) {
            dataset[total_samples].features[feature_index++] = atof(token);
            token = strtok(NULL, ",");
        }

        if (token != NULL) {
            dataset[total_samples].label = token[0];  // Assume 'P' ou 'H'
        }

        total_samples++;
    }

    fclose(file);
}

// Avaliação de acurácia (Leave-One-Out)
void evaluateModel() {
    int correct = 0;

    for (int i = 0; i < total_samples; i++) {
        // Deixa i como teste e o resto como treino
        DataPoint test_point = dataset[i];

        DataPoint train_set[MAX_SAMPLES];
        int train_size = 0;

        for (int j = 0; j < total_samples; j++) {
            if (j != i) {
                train_set[train_size++] = dataset[j];
            }
        }

        char predicted = classify(test_point, train_set, train_size);
        if (predicted == test_point.label) {
            correct++;
        }
    }

    double accuracy = (double)correct / total_samples * 100.0;
    printf("Acurácia do modelo: %.2f%%\n", accuracy);
}

// Função Principal
int main() {
    loadCSV("DARWIN.csv");

    printf("Total de amostras carregadas: %d\n", total_samples);

    double min[FEATURES], max[FEATURES];
    computeMinMax(dataset, total_samples, min, max);
    normalize(dataset, total_samples, min, max);

    evaluateModel();

    return 0;
}
