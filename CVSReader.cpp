// CVSReader.cpp
#include "CVSReader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
using namespace std;

Matrix CVSReader::lireMatrix(const string& chemin, bool ignoreHeader) {
    ifstream fichier(chemin);
    if (!fichier.is_open()) {
        cerr << "Erreur: impossible d'ouvrir " << chemin << endl;
        return Matrix();
    }

    vector<vector<double>> data;
    string ligne;

    if (ignoreHeader) getline(fichier, ligne); // ignorer l'en-tête

    while (getline(fichier, ligne)) {
        vector<double> row;
        stringstream ss(ligne);
        string valeur;
        while (getline(ss, valeur, ',')) {
            try {
                row.push_back(stod(valeur));
            } catch (...) {
                row.push_back(0.0);
            }
        }
        if (!row.empty()) data.push_back(row);
    }

    if (data.empty()) return Matrix();

    int rows = data.size();
    int cols = data[0].size();
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.setter(i, j, data[i][j]);

    return result;
}

Vector CVSReader::extraireColonne(const Matrix& data, int col) {
    int n = data.lignes();
    Vector v(n);
    for (int i = 0; i < n; i++)
        v.setter(i, data.getter(i, col));
    return v;
}

Matrix CVSReader::extraireFeatures(const Matrix& data) {
    int rows = data.lignes();
    int cols = data.colonnes() - 1; // tout sauf la dernière colonne
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.setter(i, j, data.getter(i, j));
    return result;
}