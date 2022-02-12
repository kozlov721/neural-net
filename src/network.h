#pragma once

#include "linear.h"
#include <cmath>
#include <map>
#include <utility>

using vector_of_layers = std::vector<Vector>;
using vector_of_weights = std::vector<Matrix>;

class Network {
    std::vector<int> sizes;
    vector_of_layers inner_potentials;
    vector_of_layers activated_layers;
    vector_of_weights weights;
    vector_of_weights momentum;

    /**
     * Initializes weights and other params of the network.
     */
    void init();

  public:
    /**
     * Constructs a network from a vector of sizes of layers
     * @param s Vector of sizes of layers.
     */
    explicit Network(std::vector<int> s) : sizes(std::move(s)) {
        init();
        inner_potentials = vector_of_layers(sizes.size() - 1);
        activated_layers = vector_of_layers(sizes.size());
    }

    Vector forward_pass(const Vector &);

    /**
     * Computes gradients of the weights.
     * @param y Actual labels.
     * @param y_hat Predicted labels.
     * @return Vector of gradients.
     */
    vector_of_weights backward_pass(const Vector &y, const Vector &y_hat);

    /**
     * Updates weights in the network. Uses SGD with momentum.
     * @param changes Vector of gradients.
     * @param rate Learning rate.
     */
    void update_params(const vector_of_weights &gradients, float rate);

    /**
     * Evaluates an accuracy of the network.
     * @param X Validation vectors.
     * @param y Validation labels.
     * @return Accuracy.
     */
    float evaluate(const Matrix &X, const Matrix &y);

    /**
     * Trains the network for the given number of epochs
     * with the given training rate. If verbose is true, then
     * the method will print current accuracy after each epoch.
     * @param x_train Train vectors.
     * @param y_train Train labels.
     * @param epochs Number of epochs.
     * @param rate Learning rate.
     * @param verbose
     */
    void train(const Matrix &x_train, const Matrix &y_train, int epochs,
               float rate, bool verbose = true);
};

/**
 * Applies OneHotEncoding to an input vector of classes.
 * @param labels Vector of labels.
 * @param classes Number of classes in the labels.
 * @return Vector of encoded classes.
 */
Matrix one_hot_encoding(const Vector &labels, int classes);

/**
 * Adds at the end of the vector.
 */
void add_bias(Vector &);

/**
 * Removes bias from the vector.
 */
void remove_bias(Vector &);
