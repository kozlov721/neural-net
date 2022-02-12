#include "linear.h"
#include "network.h"
#include <fstream>
#include <iomanip>
#include <iostream>

#define TRAIN_SAMPLES 60000
#define TEST_SAMPLES 10000
#define SAMPLE_SIZE 784
#define CLASSES 10

void test_and_export(Network &model, const Array &test_x,
        const std::string &file_name) {
    std::vector<int> predicted(TEST_SAMPLES);
    for (int i = 0; i < TEST_SAMPLES; ++i) {
        predicted[i] = model.forward_pass(test_x.get_row(i)).argmax();
    }
    std::ofstream file;
    file.open(file_name, std::ios::out | std::ios::trunc);
    for (const auto &c : predicted) {
        file << int(c) << std::endl;
    }
    file.close();
}

void start_training() {
    Network model({28 * 28, 100, 10});

    Array X("data/fashion_mnist_train_vectors.csv", TRAIN_SAMPLES,
            SAMPLE_SIZE);
    Array y = one_hot_encoding(
        Vector("data/fashion_mnist_train_labels.csv", TRAIN_SAMPLES, 1),
        CLASSES);
    X = X / 255.f - 0.5;

    std::cout << "Starting training." << std::endl;

    model.train(X, y, 10, 0.01, true);

    Array test_x("data/fashion_mnist_test_vectors.csv", TEST_SAMPLES,
                 SAMPLE_SIZE);
    Array test_y = one_hot_encoding(
        Vector("data/fashion_mnist_test_labels.csv", TEST_SAMPLES, 1),
        CLASSES);
    test_x = test_x / 255.f - 0.5;

    float accuracy = model.evaluate(test_x, test_y);
    std::cout << "Test accuracy: " << std::fixed << std::setw(4)
              << std::setprecision(2) << accuracy * 100.f << " %" << std::endl;

    test_and_export(model, X, "trainPredictions.csv");
    test_and_export(model, test_x, "testPredictions.csv");

    std::cout << "Predictions exported." << std::endl;
}

// TODO: rewrite into more general form, so one can specify parameters
// of the network and also run it with data other than fashion mnist.
int main() {
    start_training();
    return 0;
}
