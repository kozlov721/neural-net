#include "network.h"
#include <iomanip>
#include <iostream>
#include <numeric>

static Vector sigmoid(const Vector &x, bool derivative) {
    Vector sig = 1 / (1 + exponential(-x));
    if (derivative)
        sig = sig * (1 - sig);
    return sig;
}

static Vector softmax(const Vector &x, bool derivative) {
    Vector exps = exponential(x - x.max());
    Vector softm = exps / exps.sum();
    if (derivative)
        softm = softm * (1 - softm);
    return softm;
}

void Network::init() {
    for (unsigned int i = 1; i < sizes.size(); ++i) {
        weights.push_back(random_array(sizes[i], sizes[i - 1] + 1));
        momentum.push_back(Matrix(sizes[i], sizes[i - 1] + 1, 0));
    }
}

Vector Network::forward_pass(const Vector &X) {
    activated_layers[0] = X;
    unsigned int i;
    for (i = 1; i < sizes.size(); ++i) {
        auto act = i == sizes.size() - 1 ? softmax : sigmoid;

        // This is not very pretty. Dedicated vector for biases would be nicer.
        add_bias(activated_layers[i - 1]);
        inner_potentials[i - 1] =
            matrix_vector_dot(weights[i - 1], activated_layers[i - 1]);
        remove_bias(activated_layers[i - 1]);

        activated_layers[i] = (*act)(inner_potentials[i - 1], false);
    }
    return activated_layers[i - 1];
}

vector_of_weights Network::backward_pass(const Vector &y, const Vector &y_hat) {
    vector_of_weights changes(sizes.size() - 1);
    Vector error =
        (y_hat - y) * softmax(inner_potentials[sizes.size() - 2], true);

    add_bias(activated_layers[sizes.size() - 2]);
    changes[changes.size() - 1] =
        outer(error, activated_layers[sizes.size() - 2]);
    remove_bias(activated_layers[sizes.size() - 2]);

    add_bias(error);
    for (int i = int(sizes.size() - 1); i > 1; --i) {
        remove_bias(error);
        auto der = sigmoid(inner_potentials[i - 2], true);

        add_bias(der);
        error = matrix_vector_dot(transpose(weights[i - 1]), error) * der;

        remove_bias(error);
        add_bias(activated_layers[i - 2]);
        changes[i - 2] = outer(error, activated_layers[i - 2]);
        remove_bias(activated_layers[i - 2]);
        add_bias(error);
    }
    return changes;
}

float Network::evaluate(const Matrix &X, const Matrix &y) {
    std::vector<int> predictions;
    predictions.reserve(y.get_num_rows());
    for (int j = 0; j < y.get_num_rows(); ++j) {
        predictions.emplace_back(forward_pass(X.get_row(j)).argmax() ==
                                 y.get_row(j).argmax());
    }
    float sum = std::accumulate(predictions.begin(), predictions.end(), 0.0);
    return sum / predictions.size();
}

void Network::train(const Matrix &x_train, const Matrix &y_train,
                    const int epochs, const float rate, const bool verbose) {

    for (int t = 1; t <= epochs; ++t) {

        // Batches would be better
        for (int j = 0; j < y_train.get_num_rows(); ++j) {
            auto changes = backward_pass(y_train.get_row(j),
                                         forward_pass(x_train.get_row(j)));
            update_params(changes, rate);
        }
        if (verbose) {
            float accuracy = evaluate(x_train, y_train);
            std::cout << "Epoch: " << t << ", Accuracy: " << std::fixed
                      << std::setw(4) << std::setprecision(2)
                      << accuracy * 100.f << " %" << std::endl;
        }
    }
}

void Network::update_params(const vector_of_weights &grad, const float rate) {
    for (long unsigned int i = 0; i < weights.size(); ++i) {
        momentum[i] = 0.9 * momentum[i] - rate * grad[i];
        weights[i] = weights[i] + momentum[i];
    }
}

Matrix one_hot_encoding(const Vector &vector, const int classes) {
    Matrix out(vector.size(), classes, 0);
    for (int i = 0; i < vector.size(); ++i)
        out.get_index(i, int(vector[i])) = 1;
    return out;
}

void add_bias(Vector &v) { v.push_back(1); }

void inline remove_bias(Vector &v) { v.pop(); }
