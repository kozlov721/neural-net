#pragma once

#include <sstream>
#include <string>
#include <vector>

/**
 * Class for representing multidimensional arrays.
 * With a little improvement, it could represent
 * any n-dimensional tensor, but I wanted to be the most
 * effective, so I am taking only 2D matrices and vectors (Nx1 matrix)
 * into a consideration.
 */
class Array {
    std::vector<float> _vector;
    int _rows = 0;
    int _columns = 0;

  public:
    explicit Array() = default;

    /**
     * Constructs an Array from a vector of floats and a shape.
     * @param arr Vector of floats to be transformed to an Array.
     * @param rows Number of rows of the desired Array.
     * @param columns Number of columns.
     */
    explicit Array(std::vector<float> &&arr, int rows, int columns)
        : _vector(arr), _rows(rows), _columns(columns) {}

    /**
     * Constructs an Array from a content of a file and a shape.
     * @param file_name Name of the file with stored content of the Array.
     * @param rows Number of rows.
     * @param columns Number of columns.
     */
    explicit Array(const std::string &file_name, int rows, int columns);

    /**
     * Takes a vector of floats and constructs a Vector,
     * ie matrix with a dimensionality Nx1.
     * @param arr Vector of floats to be transformed into an Array.
     */
    explicit Array(std::vector<float> &&arr)
        : _vector(arr), _rows(arr.size()), _columns(1) {}

    /**
     * Constructs an uninitialized Vector from its length.
     * @param size The length of the Vector.
     */
    explicit Array(int size) : _vector(size), _rows(size), _columns(1) {}

    /**
     * Constructs an uninitialized Matrix from its shape.
     * @param rows Number of rows.
     * @param columns Number of columns.
     */
    explicit Array(int rows, int columns)
        : _vector(rows * columns), _rows(rows), _columns(columns) {}
    /**
     * Constructs a Matrix from its shape and fills it with
     * a default value.
     * @param rows Number of rows.
     * @param columns Number of columns.
     * @param def Default value.
     */
    explicit Array(int rows, int columns, float def)
        : _vector(rows * columns, def), _rows(rows), _columns(columns) {}

    int size() const { return _rows * _columns; }
    auto begin() const { return _vector.begin(); }
    auto end() const { return _vector.end(); }
    auto begin() { return _vector.begin(); }
    auto end() { return _vector.end(); }

    void push_back(float x);
    void pop();

    int argmax() const;
    float max() const;
    float sum() const;
    float mean() const;
    float standard_deviation() const;

    std::string string_shape() const;

    int get_num_rows() const { return _rows; }
    int get_num_columns() const { return _columns; }
    Array get_row(int i) const;

    friend Array operator+(const Array &a, const Array &b);
    friend Array operator-(const Array &a, const Array &b);
    friend Array operator*(const Array &a, const Array &b);
    friend Array operator/(const Array &a, const Array &b);

    friend Array operator*(const float &a, const Array &b);
    friend Array operator*(const Array &a, const float &b);
    friend Array operator/(const float &a, const Array &b);
    friend Array operator/(const Array &a, const float &b);
    friend Array operator-(const float &a, const Array &b);
    friend Array operator-(const Array &a, const float &b);
    friend Array operator+(const float &a, const Array &b);
    friend Array operator+(const Array &a, const float &b);
    Array operator-() const { return -1 * (*this); };
    friend std::ostream &operator<<(std::ostream &, const Array &);

    friend bool operator==(const Array &a, const Array &b);
    friend bool operator!=(const Array &a, const Array &b);

    float &operator[](int i) { return _vector[i]; }
    float operator[](int i) const { return _vector[i]; }

    // Save versions of access operators.
    float &get_index(int i, int j);
    float get_index(int i, int j) const;
    float &get_index(int i);
    float get_index(int i) const;
};

using Matrix = Array;
using Vector = Array;

/**
 * Computes a dot product of a matrix and a column vector.
 * @param m Matrix of a shape MxN.
 * @param v Vector of a shape Nx1.
 * @return Vector of a shape Mx1.
 */
Vector matrix_vector_dot(const Matrix &m, const Vector &v);

/**
 * Computes a dot product of two vectors.
 * @param a First vector.
 * @param b Second vector.
 * @return Result of the dot product.
 */
float vector_vector_dot(const Vector &a, const Vector &b);

/**
 * Computes an outer product of two vectors.
 * @param a Vector of a shape Mx1.
 * @param b Vector of a shape Nx1.
 * @return Matrix of a shape MxN.
 */
Matrix outer(const Vector &a, const Vector &b);

/**
 * Performs an exp(x) function on every element
 * of an array.
 * @param v Input vector.
 * @return Exponentiated array.
 */
Vector exponential(const Vector &v);

/**
 * Transpose a given matrix.
 * @param m Input matrix.
 * @return Transposed matrix.
 */
Matrix transpose(const Matrix &m);

/**
 * Creates a random initialized matrix of a given shape.
 * For initialization it uses He normal initialization.
 * @param rows Number of rows.
 * @param columns Number of columns.
 * @return Random initialized matrix.
 */
Matrix random_array(int rows, int columns);
