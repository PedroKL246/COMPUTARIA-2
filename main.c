#include "main.h"

// Vetor global para armazenar todos os dados do CSV
DataPoint dataset[MAX_SAMPLES];
// Variavel global para contar o numero total de amostras lidas
int total_samples = 0;

// Funcao principal do programa
int main(void)
{
    // Carrega os dados do arquivo CSV para o vetor dataset
    loadCSV("DARWIN.csv");

    // Mostra quantas amostras foram carregadas
    printf("Total de amostras carregadas: %d\n", total_samples);

    // Vetores para armazenar o minimo e maximo de cada feature (para normalizacao)
    double min[FEATURES], max[FEATURES];

    // Calcula o minimo e maximo de cada feature
    computeMinMax(dataset, total_samples, min, max);

    // Normaliza os dados (deixa todas as features no intervalo [0,1])
    normalize(dataset, total_samples, min, max);

    // Avalia o modelo usando Leave-One-Out (cada amostra e testada uma vez)
    evaluateModel();

    return 0;
}

// Funcao para carregar o CSV e preencher o vetor dataset
void loadCSV(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        printf("Erro ao abrir o arquivo.\n");
        exit(1);
    }

    char line[10000];

    // Le a primeira linha (cabecalho)
    fgets(line, sizeof(line), file);

    // Conta o numero de colunas para saber quantas features existem
    int columns = 0;
    for (int i = 0; line[i] != '\0'; i++) {
        if (line[i] == ',') columns++;
    }
    int num_features = columns - 1; // -1 porque 1 e o ID e 1 e a label

    // Le cada linha do arquivo (cada amostra)
    while (fgets(line, sizeof(line), file))
    {
        char *token;

        token = strtok(line, ","); // Ignora o ID

        // Le as features da amostra
        for (int f = 0; f < num_features; f++)
        {
            token = strtok(NULL, ",");
            if (token == NULL)
            {
                printf("Erro ao ler feature na linha %d\n", total_samples + 1);
                exit(1);
            }
            dataset[total_samples].features[f] = atof(token);
        }

        // Le a label (ultima coluna)
        token = strtok(NULL, ",\n\r");
        if (token == NULL)
        {
            printf("Erro ao ler label na linha %d\n", total_samples + 1);
            exit(1);
        }
        dataset[total_samples].label = token[0]; // 'P' ou 'H'

        total_samples++;
    }

    fclose(file);
}

// Funcao para calcular o minimo e maximo de cada feature
void computeMinMax(DataPoint *data, int size, double *min, double *max)
{
    for (int f = 0; f < FEATURES; f++)
    {
        min[f] = data[0].features[f];
        max[f] = data[0].features[f];
        for (int i = 1; i < size; i++)
        {
            if (data[i].features[f] < min[f])
                min[f] = data[i].features[f];
            if (data[i].features[f] > max[f])
                max[f] = data[i].features[f];
        }
        // Protege contra divisao por zero na normalizacao
        if (max[f] == min[f])
            max[f] = min[f] + 1.0;
    }
}

// Funcao para normalizar os dados (deixar todas as features entre 0 e 1)
void normalize(DataPoint *data, int size, double *min, double *max)
{
    for (int i = 0; i < size; i++)
    {
        for (int f = 0; f < FEATURES; f++)
        {
            data[i].features[f] = (data[i].features[f] - min[f]) / (max[f] - min[f]);
        }
    }
}

// Funcao para calcular a distancia Euclidiana entre dois pontos
double euclideanDistance(DataPoint a, DataPoint b)
{
    double sum = 0;
    for (int f = 0; f < FEATURES; f++)
    {
        sum += pow(a.features[f] - b.features[f], 2);
    }
    return sqrt(sum);
}

// Estrutura auxiliar para armazenar a distancia e a label de cada vizinho
typedef struct
{
    double distance;
    char label;
} Neighbor;

// Funcao de comparacao para ordenar os vizinhos pelo qsort
int compare(const void *a, const void *b)
{
    Neighbor *n1 = (Neighbor *)a;
    Neighbor *n2 = (Neighbor *)b;
    if (n1->distance < n2->distance)
        return -1;
    else if (n1->distance > n2->distance)
        return 1;
    else
        return 0;
}

// Funcao KNN: classifica um ponto de teste usando os K vizinhos mais proximos
char classify(DataPoint test_point, DataPoint *data, int size)
{
    double weightP = 0;
    double weightH = 0;
    double epsilon = 1e-6; // Para evitar divisao por zero

    Neighbor neighbors[size];

    // Calcula a distancia de todos os pontos de treino para o ponto de teste
    for (int i = 0; i < size; i++)
    {
        neighbors[i].distance = euclideanDistance(test_point, data[i]);
        neighbors[i].label = data[i].label;
    }

    // Ordena os vizinhos pela distancia
    qsort(neighbors, size, sizeof(Neighbor), compare);

    // Soma os pesos dos K vizinhos mais proximos para cada classe
    for (int i = 0; i < K; i++)
    {
        double weight = 1.0 / (neighbors[i].distance + epsilon);
        if (neighbors[i].label == 'P')
            weightP += weight;
        else if (neighbors[i].label == 'H')
            weightH += weight;
    }

    // Retorna a classe com maior peso
    return (weightP > weightH) ? 'P' : 'H';
}

// Funcao para avaliar o modelo usando Leave-One-Out (cada amostra e testada uma vez)
void evaluateModel()
{
    int correct = 0;

    for (int i = 0; i < total_samples; i++)
    {
        // Separa a amostra i como teste e o resto como treino
        DataPoint test_point = dataset[i];

        DataPoint train_set[MAX_SAMPLES];
        int train_size = 0;

        // Monta o conjunto de treino (todas as amostras menos a de teste)
        for (int j = 0; j < total_samples; j++)
        {
            if (j != i)
            {
                train_set[train_size++] = dataset[j];
            }
        }

        // Classifica a amostra de teste
        char predicted = classify(test_point, train_set, train_size);
        // Conta se acertou
        if (predicted == test_point.label)
        {
            correct++;
        }
    }

    // Calcula e imprime a acuracia
    double accuracy = (double)correct / total_samples * 100.0;
    printf("Acuracia do modelo: %.5f%%\n", accuracy);
}