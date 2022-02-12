#include "linear.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

void throw_shape_error(const std::string &s1, const std::string &s2,
                       const std::string &operation) {
    std::stringstream error_message;
    error_message << "Shapes " << s1 << " and " << s2
                  << " are not compatible with operation " << operation << ".";
    throw std::invalid_argument(error_message.str());
}

void check_same_shape(const Array &a, const Array &b,
                      const std::string &operation) {
    if (a.get_num_rows() != b.get_num_rows() ||
        a.get_num_columns() != b.get_num_columns())
        throw_shape_error(a.string_shape(), b.string_shape(), operation);
}

int Array::argmax() const {
    int index = 0;
    float max = _vector[0];
    for (int i = 1; i < size(); ++i) {
        if (_vector[i] > max) {
            max = _vector[i];
            index = int(i);
        }
    }
    return index;
}

float Array::max() const {
    float max = _vector[0];
    for (int i = 1; i < size(); ++i) {
        if (_vector[i] > max)
            max = _vector[i];
    }
    return max;
}

float Array::sum() const {
    float out = 0;
    for (const auto &v : *this)
        out += v;
    return out;
}

float Array::mean() const {
    float sum = 0;
    for (const auto &v : *this) {
        sum += v;
    }
    return sum / float(size());
}

std::string Array::string_shape() const {
    std::stringstream out;
    out << "(" << _rows << ", " << _columns << ")";
    return out.str();
}

Array operator+(const Array &a, const Array &b) {
    check_same_shape(a, b, "addition");
    Array out(a._rows, a._columns);
#pragma omp parallel for
    for (int i = 0; i < a.size(); ++i)
        out._vector[i] = a._vector[i] + b._vector[i];
    return out;
}

Array operator-(const Array &a, const Array &b) {
    check_same_shape(a, b, "subtraction");
    Array out(a._rows, a._columns);
#pragma omp parallel for
    for (int i = 0; i < a.size(); ++i)
        out._vector[i] = a._vector[i] - b._vector[i];
    return out;
}

Array operator*(const float &a, const Array &b) {
    Array out(b._rows, b._columns);
#pragma omp parallel for
    for (int i = 0; i < b.size(); ++i)
        out._vector[i] = a * b._vector[i];
    return out;
}

Array operator*(const Array &a, const float &b) { return b * a; }

Array operator/(const float &a, const Array &b) {
    Array out(b._rows, b._columns);
#pragma omp parallel for
    for (int i = 0; i < b.size(); ++i)
        out._vector[i] = a / b._vector[i];
    return out;
}

Array operator/(const Array &a, const float &b) { return a * (1.f / b); }

Array operator-(const float &a, const Array &b) {
    Array out(b._rows, b._columns);
#pragma omp parallel for
    for (int i = 0; i < b.size(); ++i)
        out._vector[i] = a - b._vector[i];
    return out;
}

Array operator-(const Array &a, const float &b) {
    Array out(a._rows, a._columns);
#pragma omp parallel for
    for (int i = 0; i < a.size(); ++i)
        out._vector[i] = a._vector[i] - b;
    return out;
}

Array operator+(const float &a, const Array &b) {
    Array out(b._rows, b._columns);
#pragma omp parallel for
    for (int i = 0; i < b.size(); ++i)
        out._vector[i] = a + b._vector[i];
    return out;
}

Array operator+(const Array &a, const float &b) {
    Array out(a._rows, a._columns);
#pragma omp parallel for
    for (int i = 0; i < a.size(); ++i)
        out._vector[i] = a._vector[i] + b;
    return out;
}

Array operator*(const Array &a, const Array &b) {
    check_same_shape(a, b, "elementwise multiplication");
    Array out(a._rows, a._columns);
#pragma omp parallel for
    for (int i = 0; i < b.size(); ++i)
        out._vector[i] = a._vector[i] * b._vector[i];
    return out;
}

Array operator/(const Array &a, const Array &b) {
    check_same_shape(a, b, "elementwise division");
    Array out(a._rows, a._columns);
#pragma omp parallel for
    for (int i = 0; i < b.size(); ++i)
        out._vector[i] = a._vector[i] / b._vector[i];
    return out;
}

bool operator==(const Array &a, const Array &b) {
    for (int i = 0; i < a.size(); ++i) {
        if (a._vector[i] != b._vector[i])
            return false;
    }
    return true;
}

bool operator!=(const Array &a, const Array &b) { return !(a == b); }

float &Array::get_index(int i, int j) { return _vector[i * _columns + j]; }

float Array::get_index(int i, int j) const { return _vector[i * _columns + j]; }

float &Array::get_index(int i) { return _vector[i]; }

float Array::get_index(int i) const { return _vector[i]; }

Array Array::get_row(int i) const {
    Array row(get_num_columns());
#pragma omp parallel for
    for (int j = 0; j < get_num_columns(); ++j) {
        row[j] = get_index(i, j);
    }
    return row;
}

Array::Array(const std::string &file_name, int rows, int columns) {
    std::ifstream file(file_name);
    std::string line;
    std::string num;
    while (std::getline(file, line)) {
        std::stringstream line_stream(line);
        while (std::getline(line_stream, num, ','))
            _vector.emplace_back(std::stof(num));
    }
    _rows = rows;
    _columns = columns;
}

std::ostream &operator<<(std::ostream &out, const Array &a) {
    out << "(";
    for (const auto &v : a)
        out << std::fixed << std::setw(4) << std::setprecision(2) << v << " ";
    out << ")";
    return out;
}

float Array::standard_deviation() const {
    float out = 0;
    const float m = mean();
    for (auto &v : *this)
        out += std::pow(v - m, 2);
    return std::sqrt(out / float(size()));
}

//?
float matrix_row_vector_dot(const Matrix &a, const Vector &b, int row) {
    float out = 0;
#pragma omp parallel for
    for (int i = 0; i < b.size(); ++i) {
        out += a.get_index(row, i) * b.get_index(i);
    }
    return out;
}

Vector matrix_vector_dot(const Matrix &a, const Vector &b) {
    if (a.get_num_columns() != b.get_num_rows())
        throw_shape_error(a.string_shape(), b.string_shape(),
                          "matrix vector dot");
    Matrix out(a.get_num_rows());
#pragma omp parallel for
    for (int i = 0; i < a.get_num_rows(); ++i)
        out[i] = matrix_row_vector_dot(a, b, i);
    return out;
}

//?
float vector_vector_dot(const Vector &a, const Vector &b) {
    check_same_shape(a, b, "vector dot");
    float out = 0;
#pragma omp parallel for
    for (int i = 0; i < a.size(); ++i)
        out += a[i] * b[i];
    return out;
}

Matrix outer(const Vector &a, const Vector &b) {
    Matrix out(a.size(), b.size());
#pragma omp parallel for
    for (int i = 0; i < a.size(); ++i) {
#pragma omp parallel for
        for (int j = 0; j < b.size(); ++j) {
            out.get_index(i, j) = a[i] * b[j];
        }
    }
    return out;
}

Array exponential(const Array &arr) {
    Array out(arr.get_num_rows(), arr.get_num_columns());
#pragma omp parallel for
    for (int i = 0; i < arr.size(); ++i)
        out[i] = std::exp(arr[i]);
    return out;
}

Matrix transpose(const Matrix &m) {
    Matrix transposed(m.get_num_columns(), m.get_num_rows());
#pragma omp parallel for
    for (int i = 0; i < m.get_num_rows(); ++i) {
#pragma omp parallel for
        for (int j = 0; j < m.get_num_columns(); ++j) {
            transposed.get_index(j, i) = m.get_index(i, j);
        }
    }
    return transposed;
}

Array random_array(int rows, int columns) {
    Array out(rows, columns);
#ifdef RANDOM
    std::random_device rd{};
    std::mt19937 gen{rd()};
#else
    std::mt19937 gen{42};
#endif
    std::normal_distribution<> d{0, 1};
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < columns; ++j)
            out.get_index(i, j) =
                float(d(gen)) * float(std::sqrt(2. / columns));
    return out;
}

void Array::push_back(float x) {
    _vector.push_back(x);
    _rows++;
}

void Array::pop() {
    _vector.pop_back();
    _rows--;
}
