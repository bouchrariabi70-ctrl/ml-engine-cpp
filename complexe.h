#ifndef COMPLEXE_H
#define COMPLEXE_H

#include <iostream>
#include <cmath>

class complexe {
private:
    double reel;
    double imgn;

public:
    complexe(double r = 0, double i = 0);
    complexe(const complexe& c);
    ~complexe() {}

    void afficher() const;
    double module() const;
    double argument() const;
    complexe conjugue() const;

    void setter(double r, double i);
    double getReel() const { return reel; }
    double getImgn() const { return imgn; }

    complexe& operator=(const complexe& autre);
    complexe operator+(const complexe& z) const;
    complexe operator-(const complexe& z) const;
    complexe operator*(const complexe& z) const;
    complexe operator/(const complexe& z) const;

    complexe& operator+=(const complexe& autre);
    complexe& operator-=(const complexe& autre);
    complexe& operator*=(const complexe& autre);
    complexe& operator/=(const complexe& autre);

    bool operator==(const complexe& autre) const;
    bool operator!=(const complexe& autre) const;

    friend complexe operator+(double d, const complexe& z);
    friend complexe operator+(const complexe& z, double d);
    friend complexe operator-(double d, const complexe& z);
    friend complexe operator-(const complexe& z, double d);
    friend complexe operator*(const complexe& z, double d);
    friend complexe operator*(double d, const complexe& z);
    friend complexe operator/(const complexe& z, double d);
    friend std::ostream& operator<<(std::ostream& os, const complexe& c);
};

#endif