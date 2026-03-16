#include "LinearRegression.h"
#include <iostream>
#include <cmath>
using namespace std;

LinearRegression::LinearRegression() : poids(1, 0.0), biais(0.0) {}

void LinearRegression::entrainer(const Matrix& X, const Vector& y,
                                  double lr, int epochs) {
    int n = X.lignes();
    int m = X.colonnes();
    poids = Vector(m, 0.0);
    biais = 0.0;

    for (int e = 0; e < epochs; e++) {
        Vector y_pred = predire(X);

        // Gradient pour chaque poids
        for (int j = 0; j < m; j++) {
            double grad = 0.0;
            for (int i = 0; i < n; i++)
                grad += (y_pred.getter(i) - y.getter(i)) * X.getter(i, j);
            poids.setter(j, poids.getter(j) - lr * (grad / n));
        }

        // Gradient pour le biais
        double grad_b = 0.0;
        for (int i = 0; i < n; i++)
            grad_b += (y_pred.getter(i) - y.getter(i));
        biais -= lr * (grad_b / n);
    }
}

Vector LinearRegression::predire(const Matrix& X) const {
    int n = X.lignes();
    Vector result(n, biais);
    for (int i = 0; i < n; i++) {
        double somme = biais;
        for (int j = 0; j < X.colonnes(); j++)
            somme += X.getter(i, j) * poids.getter(j);
        result.setter(i, somme);
    }
    return result;
}

double LinearRegression::mse(const Vector& y_pred, const Vector& y_reel) const {
    double somme = 0.0;
    for (int i = 0; i < y_pred.taille(); i++) {
        double diff = y_pred.getter(i) - y_reel.getter(i);
        somme += diff * diff;
    }
    return somme / y_pred.taille();
}

void LinearRegression::afficher() const {
    cout << "Poids : " << poids << endl;
    cout << "Biais : " << biais << endl;
}