#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include "Vector.h"
#include "Matrix.h"

class LinearRegression {
private:
    Vector poids;
    double biais;

public:
    LinearRegression();
    void entrainer(const Matrix& X, const Vector& y,
                   double lr = 0.01, int epochs = 1000);
    Vector predire(const Matrix& X) const;
    double mse(const Vector& y_pred, const Vector& y_reel) const;
    void afficher() const;
};

#endif