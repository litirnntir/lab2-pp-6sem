#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <stdexcept>
#include <omp.h>

// Функция для генерации случайной матрицы
std::vector<std::vector<int>> generateRandomMatrix(int rows, int cols) {
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100); // Генерация случайных чисел от 1 до 100
#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}

// Функция для чтения матрицы из файла
std::vector<std::vector<int>> readMatrix(const std::string& filename, int& rows, int& cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл.");
    }

    file >> rows >> cols;
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix[i][j];
        }
    }

    file.close();
    return matrix;
}

// Функция для записи матрицы в файл
void writeMatrix(const std::vector<std::vector<int>>& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для записи.");
    }

    file << matrix.size() << " " << matrix[0].size() << std::endl;
    for (const auto& row : matrix) {
        for (const int& val : row) {
            file << val << " ";
        }
        file << std::endl;
    }

    file.close();
}

// Функция для перемножения двух матриц
std::vector<std::vector<int>> multiplyMatricesFromFile(const std::string& file1, const std::string& file2) {
    int rowsA, colsA, rowsB, colsB;

    std::vector<std::vector<int>> matrix1 = readMatrix(file1, rowsA, colsA);

    std::vector<std::vector<int>> matrix2 = readMatrix(file2, rowsB, colsB);

    if (colsA != rowsB) {
        throw std::runtime_error("Некорректные размеры матриц для умножения.");
    }

    std::vector<std::vector<int>> result(rowsA, std::vector<int>(colsB, 0));
#pragma omp parallel for num_threads(8)
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return result;
}



void calculateConfidenceInterval(const double& mean, const double& stdev, const int& numSamples) {
    double z = 1.96;
    double marginOfError = z * stdev / sqrt(numSamples);

    std::cout << "Confidence Interval for the Execution Time: [" << mean - marginOfError << ", " << mean + marginOfError << "]" << std::endl;
}

int main() {
    std::cout << omp_get_max_threads() << std::endl;
    const int rows1 = 2000;
    const int cols1 = 2000;
    const int rows2 = 2000;
    const int cols2 = 2000;
    std::vector<std::vector<int>> matrix1 = generateRandomMatrix(rows1, cols1);
    std::vector<std::vector<int>> matrix2 = generateRandomMatrix(rows2, cols2);

    writeMatrix(matrix1, "matrix1.txt");
    writeMatrix(matrix2, "matrix2.txt");

    std::vector<std::vector<int>> result;
    auto start = std::chrono::high_resolution_clock::now();
    result = multiplyMatricesFromFile("matrix1.txt", "matrix2.txt");
    auto end = std::chrono::high_resolution_clock::now();
    writeMatrix(result, "result_matrix.txt");

    std::chrono::duration<double> duration = end - start;
    double meanTime = duration.count();

    double stdevTime = 0;

    calculateConfidenceInterval(meanTime, stdevTime, 1);

    std::cout << "The scope of the task: " << rows1 * cols1 + rows2 * cols2 << " elements." << std::endl;
    std::cout << "Execution time: " << meanTime << " seconds." << std::endl;
    return 0;
}