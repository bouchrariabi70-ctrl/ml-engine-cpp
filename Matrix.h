#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include "Vector.h"

class Matrix {
public:
    // Constructeurs
    Matrix();
    Matrix(int l, int c);
    Matrix(int l, int c, double valeur);   // avec valeur initiale
    Matrix(const Matrix& autre);

    // Destructeur
    ~Matrix();

    // Opérateur d'affectation
    Matrix& operator=(const Matrix& m);

    // Méthodes utilitaires
    void afficher() const;
    void setter(int i, int j, double elem);
    double getter(int i, int j) const;
    int lignes() const;
    int colonnes() const;
    Matrix transpose() const;
    double trace() const;
    double determinant() const;

    // Multiplication Matrix * Vector (utile pour ML)
    Vector operator*(const Vector& v) const;

    // Opérateurs arithmétiques
    Matrix operator+(const Matrix& m) const;
    Matrix operator-(const Matrix& m) const;
    Matrix operator*(const Matrix& m) const;
    Matrix operator*(double coef) const;

    // Opérateurs de comparaison
    bool operator==(const Matrix& m) const;
    bool operator!=(const Matrix& m) const;

    // Fonctions amies
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);
    friend Matrix operator*(double c, const Matrix& m);

private:
    int ligne;
    int colone;
    double** tab;
};

#endif
