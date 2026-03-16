#include "Vector.h"
#include <iostream>
#include <cmath>
using namespace std;

// Constructeur par défaut
Vector::Vector() {
    ty = 0;
    tab = nullptr;
}

// Constructeur avec taille
Vector::Vector(int t) {
    ty = t;
    tab = new double[ty];
    for (int i = 0; i < ty; i++)
        tab[i] = 0.0;   // initialise à zéro
}

// Constructeur avec taille + valeur initiale
Vector::Vector(int t, double valeur) {
    ty = t;
    tab = new double[ty];
    for (int i = 0; i < ty; i++)
        tab[i] = valeur;
}

// Constructeur de copie
Vector::Vector(const Vector& v) {
    ty = v.ty;
    tab = new double[ty];
    for (int i = 0; i < ty; i++)
        tab[i] = v.tab[i];
}

// Destructeur
Vector::~Vector() {
    delete[] tab;
}

// Opérateur d'affectation
Vector& Vector::operator=(const Vector& v) {
    if (this == &v) return *this;  // auto-affectation
    delete[] tab;
    ty = v.ty;
    tab = new double[ty];
    for (int i = 0; i < ty; i++)
        tab[i] = v.tab[i];
    return *this;
}

// Méthodes
void Vector::afficher() const {
    cout << "Vecteur de taille " << ty << " : [";
    for (int i = 0; i < ty; i++) {
        cout << tab[i];
        if (i < ty - 1) cout << ", ";
    }
    cout << "]" << endl;
}

void Vector::setter(int i, double x) {
    tab[i] = x;
}

double Vector::getter(int i) const {
    return tab[i];
}

int Vector::taille() const {
    return ty;
}

double Vector::somme() const {
    double s = 0;
    for (int i = 0; i < ty; i++)
        s += tab[i];
    return s;
}

double Vector::norme() const {
    double s = 0;
    for (int i = 0; i < ty; i++)
        s += pow(tab[i], 2);
    return sqrt(s);
}

double Vector::produitScalaire(const Vector& v) const {
    if (ty != v.ty) {
        cerr << "Erreur: tailles incompatibles pour le produit scalaire." << endl;
        return 0.0;
    }
    double result = 0.0;
    for (int i = 0; i < ty; i++)
        result += tab[i] * v.tab[i];
    return result;
}

// Opérateurs arithmétiques
Vector Vector::operator+(const Vector& z) const {
    if (ty != z.ty) {
        cerr << "Erreur: tailles incompatibles." << endl;
        return Vector();
    }
    Vector result(ty);
    for (int i = 0; i < ty; i++)
        result.tab[i] = tab[i] + z.tab[i];
    return result;
}

Vector Vector::operator-(const Vector& z) const {
    if (ty != z.ty) {
        cerr << "Erreur: tailles incompatibles." << endl;
        return Vector();
    }
    Vector result(ty);
    for (int i = 0; i < ty; i++)
        result.tab[i] = tab[i] - z.tab[i];
    return result;
}

Vector Vector::operator*(const Vector& z) const {
    if (ty != z.ty) {
        cerr << "Erreur: tailles incompatibles." << endl;
        return Vector();
    }
    Vector result(ty);
    for (int i = 0; i < ty; i++)
        result.tab[i] = tab[i] * z.tab[i];
    return result;
}

Vector Vector::operator/(const Vector& z) const {
    if (ty != z.ty) {
        cerr << "Erreur: tailles incompatibles." << endl;
        return Vector();
    }
    Vector result(ty);
    for (int i = 0; i < ty; i++)
        result.tab[i] = tab[i] / z.tab[i];
    return result;
}

// Opérateurs de comparaison
bool Vector::operator==(const Vector& z) const {
    if (ty != z.ty) return false;
    for (int i = 0; i < ty; i++)
        if (tab[i] != z.tab[i]) return false;
    return true;
}

bool Vector::operator!=(const Vector& z) const {
    return !(*this == z);
}

// Fonctions amies
ostream& operator<<(ostream& os, const Vector& v) {
    os << "[";
    for (int i = 0; i < v.ty; i++) {
        os << v.tab[i];
        if (i < v.ty - 1) os << ", ";
    }
    os << "]";
    return os;
}

Vector operator*(double c, const Vector& v) {
    Vector result(v.ty);
    for (int i = 0; i < v.ty; i++)
        result.tab[i] = c * v.tab[i];
    return result;
}