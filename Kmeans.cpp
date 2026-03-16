#include "Kmeans.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
using namespace std;

KMeans::KMeans(int k) : k(k), centroides(k, 2) {}

double KMeans::distance(const Vector& a, const Vector& b) const {
    double somme = 0.0;
    for (int i = 0; i < a.taille(); i++) {
        double diff = a.getter(i) - b.getter(i);
        somme += diff * diff;
    }
    return sqrt(somme);
}

int KMeans::centroideLeProche(const Vector& point) const {
    int idx = 0;
    double minDist = 1e18;
    for (int i = 0; i < k; i++) {
        // extraire le centroide i comme vecteur
        Vector c(centroides.colonnes());
        for (int j = 0; j < centroides.colonnes(); j++)
            c.setter(j, centroides.getter(i, j));
        double d = distance(point, c);
        if (d < minDist) {
            minDist = d;
            idx = i;
        }
    }
    return idx;
}

void KMeans::entrainer(const Matrix& X, int maxIter) {
    int n = X.lignes();
    int cols = X.colonnes();
    centroides = Matrix(k, cols);
    labels.resize(n, 0);

    // Initialisation : prendre les k premiers points comme centroides
    for (int i = 0; i < k; i++)
        for (int j = 0; j < cols; j++)
            centroides.setter(i, j, X.getter(i, j));

    for (int iter = 0; iter < maxIter; iter++) {
        // Assigner chaque point au centroide le plus proche
        for (int i = 0; i < n; i++) {
            Vector point(cols);
            for (int j = 0; j < cols; j++)
                point.setter(j, X.getter(i, j));
            labels[i] = centroideLeProche(point);
        }

        // Recalculer les centroides
        Matrix nouveauxCentroides(k, cols, 0.0);
        vector<int> compteur(k, 0);

        for (int i = 0; i < n; i++) {
            int cluster = labels[i];
            compteur[cluster]++;
            for (int j = 0; j < cols; j++)
                nouveauxCentroides.setter(cluster, j,
                    nouveauxCentroides.getter(cluster, j) + X.getter(i, j));
        }

        for (int i = 0; i < k; i++)
            if (compteur[i] > 0)
                for (int j = 0; j < cols; j++)
                    nouveauxCentroides.setter(i, j,
                        nouveauxCentroides.getter(i, j) / compteur[i]);

        centroides = nouveauxCentroides;
    }
}

vector<int> KMeans::predire(const Matrix& X) const {
    int n = X.lignes();
    vector<int> result(n);
    for (int i = 0; i < n; i++) {
        Vector point(X.colonnes());
        for (int j = 0; j < X.colonnes(); j++)
            point.setter(j, X.getter(i, j));
        result[i] = centroideLeProche(point);
    }
    return result;
}

void KMeans::afficherCentroides() const {
    cout << "Centroides finaux :" << endl;
    for (int i = 0; i < k; i++) {
        cout << "  Cluster " << i << " : [";
        for (int j = 0; j < centroides.colonnes(); j++) {
            cout << centroides.getter(i, j);
            if (j < centroides.colonnes() - 1) cout << ", ";
        }
        cout << "]" << endl;
    }
}