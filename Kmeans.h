#ifndef KMEANS_H
#define KMEANS_H

#include "Vector.h"
#include "Matrix.h"
#include <vector>

class KMeans {
private:
    int k;
    Matrix centroides;
    std::vector<int> labels;

    double distance(const Vector& a, const Vector& b) const;
    int centroideLeProche(const Vector& point) const;

public:
    KMeans(int k);
    void entrainer(const Matrix& X, int maxIter = 100);
    std::vector<int> predire(const Matrix& X) const;
    void afficherCentroides() const;
};

#endif