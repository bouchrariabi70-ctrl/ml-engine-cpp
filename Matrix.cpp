#include "Matrix.h"
#include <iostream>
using namespace std;

// Constructeur par défaut
Matrix::Matrix() {
    ligne = 0;
    colone = 0;
    tab = nullptr;
}

// Constructeur avec dimensions
Matrix::Matrix(int l, int c) {
    ligne = l;
    colone = c;
    tab = new double*[ligne];
    for (int i = 0; i < ligne; i++) {
        tab[i] = new double[colone];
        for (int j = 0; j < colone; j++)
            tab[i][j] = 0.0;
    }
}

// Constructeur avec valeur initiale
Matrix::Matrix(int l, int c, double valeur) {
    ligne = l;
    colone = c;
    tab = new double*[ligne];
    for (int i = 0; i < ligne; i++) {
        tab[i] = new double[colone];
        for (int j = 0; j < colone; j++)
            tab[i][j] = valeur;
    }
}

// Constructeur de copie
Matrix::Matrix(const Matrix& m) {
    ligne = m.ligne;
    colone = m.colone;
    tab = new double*[ligne];
    for (int i = 0; i < ligne; i++) {
        tab[i] = new double[colone];
        for (int j = 0; j < colone; j++)
            tab[i][j] = m.tab[i][j];
    }
}

// Destructeur
Matrix::~Matrix() {
    if (tab != nullptr) {
        for (int i = 0; i < ligne; i++)
            delete[] tab[i];
        delete[] tab;
    }
}

// Opérateur d'affectation
Matrix& Matrix::operator=(const Matrix& m) {
    if (this == &m) return *this;
    // libérer l'ancienne mémoire
    if (tab != nullptr) {
        for (int i = 0; i < ligne; i++)
            delete[] tab[i];
        delete[] tab;
    }
    ligne = m.ligne;
    colone = m.colone;
    tab = new double*[ligne];
    for (int i = 0; i < ligne; i++) {
        tab[i] = new double[colone];
        for (int j = 0; j < colone; j++)
            tab[i][j] = m.tab[i][j];
    }
    return *this;
}

// Méthodes utilitaires
void Matrix::afficher() const {
    for (int i = 0; i < ligne; i++) {
        for (int j = 0; j < colone; j++)
            cout << tab[i][j] << "\t";
        cout << endl;
    }
}

void Matrix::setter(int i, int j, double elem) {
    tab[i][j] = elem;
}

double Matrix::getter(int i, int j) const {
    return tab[i][j];
}

int Matrix::lignes() const {
    return ligne;
}

int Matrix::colonnes() const {
    return colone;
}

Matrix Matrix::transpose() const {
    Matrix v(colone, ligne);
    for (int i = 0; i < ligne; i++)
        for (int j = 0; j < colone; j++)
            v.tab[j][i] = tab[i][j];
    return v;
}

double Matrix::trace() const {
    if (ligne != colone) {
        cerr << "Erreur: trace définie uniquement pour matrices carrées." << endl;
        return 0.0;
    }
    double somme = 0.0;
    for (int i = 0; i < ligne; i++)
        somme += tab[i][i];
    return somme;
}

double Matrix::determinant() const {
    if (ligne != colone) {
        cerr << "Erreur: déterminant défini uniquement pour matrices carrées." << endl;
        return 0.0;
    }
    if (ligne == 1) return tab[0][0];
    if (ligne == 2)
        return tab[0][0]*tab[1][1] - tab[0][1]*tab[1][0];
    if (ligne == 3)
        return tab[0][0]*(tab[1][1]*tab[2][2] - tab[1][2]*tab[2][1])
             - tab[0][1]*(tab[1][0]*tab[2][2] - tab[1][2]*tab[2][0])
             + tab[0][2]*(tab[1][0]*tab[2][1] - tab[1][1]*tab[2][0]);
    cerr << "Déterminant pour n>3 non implémenté." << endl;
    return 0.0;
}

// Matrix * Vector (utile pour ML)
Vector Matrix::operator*(const Vector& v) const {
    if (colone != v.taille()) {
        cerr << "Erreur: dimensions incompatibles Matrix * Vector." << endl;
        return Vector();
    }
    Vector result(ligne);
    for (int i = 0; i < ligne; i++) {
        double somme = 0.0;
        for (int j = 0; j < colone; j++)
            somme += tab[i][j] * v.getter(j);
        result.setter(i, somme);
    }
    return result;
}

// Opérateurs arithmétiques
Matrix Matrix::operator+(const Matrix& m) const {
    if (ligne != m.ligne || colone != m.colone) {
        cerr << "Erreur: dimensions incompatibles pour l'addition." << endl;
        return Matrix();
    }
    Matrix result(ligne, colone);
    for (int i = 0; i < ligne; i++)
        for (int j = 0; j < colone; j++)
            result.tab[i][j] = tab[i][j] + m.tab[i][j];
    return result;
}

Matrix Matrix::operator-(const Matrix& m) const {
    if (ligne != m.ligne || colone != m.colone) {
        cerr << "Erreur: dimensions incompatibles pour la soustraction." << endl;
        return Matrix();
    }
    Matrix result(ligne, colone);
    for (int i = 0; i < ligne; i++)
        for (int j = 0; j < colone; j++)
            result.tab[i][j] = tab[i][j] - m.tab[i][j];
    return result;
}

Matrix Matrix::operator*(const Matrix& m) const {
    if (colone != m.ligne) {
        cerr << "Erreur: dimensions incompatibles pour la multiplication." << endl;
        return Matrix();
    }
    Matrix result(ligne, m.colone);
    for (int i = 0; i < ligne; i++)
        for (int j = 0; j < m.colone; j++)
            for (int k = 0; k < colone; k++)
                result.tab[i][j] += tab[i][k] * m.tab[k][j];
    return result;
}

Matrix Matrix::operator*(double c) const {
    Matrix result(ligne, colone);
    for (int i = 0; i < ligne; i++)
        for (int j = 0; j < colone; j++)
            result.tab[i][j] = c * tab[i][j];
    return result;
}

// Opérateurs de comparaison
bool Matrix::operator==(const Matrix& m) const {
    if (ligne != m.ligne || colone != m.colone) return false;
    for (int i = 0; i < ligne; i++)
        for (int j = 0; j < colone; j++)
            if (tab[i][j] != m.tab[i][j]) return false;
    return true;
}

bool Matrix::operator!=(const Matrix& m) const {
    return !(*this == m);
}

// Fonctions amies
ostream& operator<<(ostream& os, const Matrix& m) {
    for (int i = 0; i < m.ligne; i++) {
        for (int j = 0; j < m.colone; j++)
            os << m.tab[i][j] << "\t";
        os << endl;
    }
    return os;
}

Matrix operator*(double c, const Matrix& m) {
    Matrix result(m.ligne, m.colone);
    for (int i = 0; i < m.ligne; i++)
        for (int j = 0; j < m.colone; j++)
            result.tab[i][j] = c * m.tab[i][j];
    return result;
}