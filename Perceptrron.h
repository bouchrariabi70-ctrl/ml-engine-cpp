#ifndef PERCEPTRRON_H
#define PERCEPTRRON_H

#include "Vector.h"
#include "Matrix.h"

class Perceptrron {
private:
    Vector poids;
    double biais;
    double lr;

    int activation(double x) const;

public:
    Perceptrron(double lr = 0.1);
    void entrainer(const Matrix& X, const Vector& y, int epochs = 100);
    Vector predire(const Matrix& X) const;
    double accuracy(const Vector& y_pred, const Vector& y_reel) const;
    void afficher() const;
};

#endifxx