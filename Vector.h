#ifndef VECTOR_H
#define VECTOR_H
#include <iostream>

class Vector {
public:
    // Constructeurs
    Vector();                          // constructeur par défaut (ajout)
    Vector(int t);
    Vector(int t, double valeur);      // constructeur avec valeur initiale (utile pour ML)
    Vector(const Vector& v);           // constructeur de copie (ajout)

    // Destructeur
    ~Vector();

    // Opérateur d'affectation (ajout)
    Vector& operator=(const Vector& v);

    // Méthodes
    void afficher() const;
    void setter(int i, double x);
    double getter(int i) const;
    int taille() const;
    double norme() const;
    double somme() const;
    double produitScalaire(const Vector& v) const;  // const ref (correction)

    // Opérateurs arithmétiques
    Vector operator+(const Vector& z) const;
    Vector operator-(const Vector& z) const;
    Vector operator*(const Vector& z) const;
    Vector operator/(const Vector& z) const;

    // Opérateurs de comparaison
    bool operator==(const Vector& z) const;
    bool operator!=(const Vector& z) const;

    // Fonctions amies
    friend std::ostream& operator<<(std::ostream& os, const Vector& v);
    friend Vector operator*(double c, const Vector& v);

private:
    int ty;
    double* tab;
};

#endif