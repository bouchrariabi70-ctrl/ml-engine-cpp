#include "complexe.h"
#include <iostream>
#include <cmath>
using namespace std;

complexe::complexe(double r, double i) : reel(r), imgn(i) {}

complexe::complexe(const complexe& c) : reel(c.reel), imgn(c.imgn) {}

void complexe::afficher() const {
    if (imgn >= 0)
        cout << reel << " + " << imgn << "i" << endl;
    else
        cout << reel << " - " << (-imgn) << "i" << endl;
}

double complexe::module() const {
    return sqrt(reel * reel + imgn * imgn);
}

double complexe::argument() const {
    return atan2(imgn, reel);
}

complexe complexe::conjugue() const {
    return complexe(reel, -imgn);
}

void complexe::setter(double r, double i) {
    reel = r;
    imgn = i;
}

complexe& complexe::operator=(const complexe& autre) {
    if (this == &autre) return *this;
    reel = autre.reel;
    imgn = autre.imgn;
    return *this;
}

complexe complexe::operator+(const complexe& z) const {
    return complexe(reel + z.reel, imgn + z.imgn);
}

complexe complexe::operator-(const complexe& z) const {
    return complexe(reel - z.reel, imgn - z.imgn);
}

complexe complexe::operator*(const complexe& z) const {
    return complexe(
        reel * z.reel - imgn * z.imgn,
        reel * z.imgn + imgn * z.reel
    );
}

complexe complexe::operator/(const complexe& z) const {
    double denom = z.reel * z.reel + z.imgn * z.imgn;
    if (denom == 0) {
        cerr << "Erreur: division par zero." << endl;
        return complexe(0, 0);
    }
    return complexe(
        (reel * z.reel + imgn * z.imgn) / denom,
        (imgn * z.reel - reel * z.imgn) / denom
    );
}

complexe& complexe::operator+=(const complexe& autre) {
    reel += autre.reel;
    imgn += autre.imgn;
    return *this;
}

complexe& complexe::operator-=(const complexe& autre) {
    reel -= autre.reel;
    imgn -= autre.imgn;
    return *this;
}

complexe& complexe::operator*=(const complexe& autre) {
    double r = reel * autre.reel - imgn * autre.imgn;
    double i = reel * autre.imgn + imgn * autre.reel;
    reel = r;
    imgn = i;
    return *this;
}

complexe& complexe::operator/=(const complexe& autre) {
    *this = *this / autre;
    return *this;
}

bool complexe::operator==(const complexe& autre) const {
    return (reel == autre.reel && imgn == autre.imgn);
}

bool complexe::operator!=(const complexe& autre) const {
    return !(*this == autre);
}

complexe operator+(double d, const complexe& z) {
    return complexe(d + z.getReel(), z.getImgn());
}

complexe operator+(const complexe& z, double d) {
    return complexe(z.getReel() + d, z.getImgn());
}

complexe operator-(double d, const complexe& z) {
    return complexe(d - z.getReel(), -z.getImgn());
}

complexe operator-(const complexe& z, double d) {
    return complexe(z.getReel() - d, z.getImgn());
}

complexe operator*(const complexe& z, double d) {
    return complexe(z.getReel() * d, z.getImgn() * d);
}

complexe operator*(double d, const complexe& z) {
    return complexe(d * z.getReel(), d * z.getImgn());
}

complexe operator/(const complexe& z, double d) {
    if (d == 0) {
        cerr << "Erreur: division par zero." << endl;
        return complexe(0, 0);
    }
    return complexe(z.getReel() / d, z.getImgn() / d);
}

ostream& operator<<(ostream& os, const complexe& c) {
    if (c.getImgn() >= 0)
        os << c.getReel() << " + " << c.getImgn() << "i";
    else
        os << c.getReel() << " - " << (-c.getImgn()) << "i";
    return os;
}