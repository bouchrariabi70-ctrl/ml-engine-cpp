#include "Perceptrron.h"
#include <iostream>
using namespace std;

Perceptrron::Perceptrron(double lr) : poids(1, 0.0), biais(0.0), lr(lr) {}

int Perceptrron::activation(double x) const {
    return x >= 0.0 ? 1 : 0;
}

void Perceptrron::entrainer(const Matrix& X, const Vector& y, int epochs) {
    int n = X.lignes();
    int m = X.colonnes();
    poids = Vector(m, 0.0);
    biais = 0.0;

    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < n; i++) {
            // Calcul de la sortie
            double somme = biais;
            for (int j = 0; j < m; j++)
                somme += X.getter(i, j) * poids.getter(j);

            int y_pred = activation(somme);
            int erreur = (int)y.getter(i) - y_pred;

            // Mise à jour des poids
            for (int j = 0; j < m; j++)
                poids.setter(j, poids.getter(j) + lr * erreur * X.getter(i, j));
            biais += lr * erreur;
        }
    }
}

Vector Perceptrron::predire(const Matrix& X) const {
    int n = X.lignes();
    Vector result(n);
    for (int i = 0; i < n; i++) {
        double somme = biais;
        for (int j = 0; j < X.colonnes(); j++)
            somme += X.getter(i, j) * poids.getter(j);
        result.setter(i, activation(somme));
    }
    return result;
}

double Perceptrron::accuracy(const Vector& y_pred, const Vector& y_reel) const {
    int correct = 0;
    for (int i = 0; i < y_pred.taille(); i++)
        if ((int)y_pred.getter(i) == (int)y_reel.getter(i))
            correct++;
    return (double)correct / y_pred.taille() * 100.0;
}

void Perceptrron::afficher() const {
    cout << "Poids : " << poids << endl;
    cout << "Biais : " << biais << endl;
}