#include <iostream>
#include "Vector.h"
#include "Matrix.h"
#include "complexe.h"
#include "LinearRegression.h"
#include "Kmeans.h"
#include "Perceptrron.h"
#include "CVSReader.h"

using namespace std;

// Normalise un vecteur entre 0 et 1
Vector normaliser(const Vector& v, double& vmin, double& vmax) {
    vmin = v.getter(0);
    vmax = v.getter(0);
    for (int i = 1; i < v.taille(); i++) {
        if (v.getter(i) < vmin) vmin = v.getter(i);
        if (v.getter(i) > vmax) vmax = v.getter(i);
    }
    Vector result(v.taille());
    for (int i = 0; i < v.taille(); i++)
        result.setter(i, (v.getter(i) - vmin) / (vmax - vmin));
    return result;
}

// Normalise une matrice colonne par colonne
Matrix normaliserMatrix(const Matrix& M) {
    Matrix result(M.lignes(), M.colonnes());
    for (int j = 0; j < M.colonnes(); j++) {
        double vmin = M.getter(0, j);
        double vmax = M.getter(0, j);
        for (int i = 1; i < M.lignes(); i++) {
            if (M.getter(i, j) < vmin) vmin = M.getter(i, j);
            if (M.getter(i, j) > vmax) vmax = M.getter(i, j);
        }
        for (int i = 0; i < M.lignes(); i++)
            result.setter(i, j, (M.getter(i, j) - vmin) / (vmax - vmin));
    }
    return result;
}

int main() {

    // ==========================================
    // TEST VECTOR
    // ==========================================
    cout << "=== TEST VECTOR ===" << endl;

    Vector v1(3), v2(3);
    v1.setter(0, 1.0); v1.setter(1, 2.0); v1.setter(2, 3.0);
    v2.setter(0, 4.0); v2.setter(1, 5.0); v2.setter(2, 6.0);

    cout << "v1 = " << v1 << endl;
    cout << "v2 = " << v2 << endl;
    cout << "v1 + v2 = " << (v1 + v2) << endl;
    cout << "v1 - v2 = " << (v1 - v2) << endl;
    cout << "Produit scalaire = " << v1.produitScalaire(v2) << endl;
    cout << "Norme v1 = " << v1.norme() << endl;
    cout << "2.5 * v1 = " << (2.5 * v1) << endl;

    // ==========================================
    // TEST MATRIX
    // ==========================================
    cout << "\n=== TEST MATRIX ===" << endl;

    Matrix A(2, 2), B(2, 2);
    A.setter(0,0, 1); A.setter(0,1, 2);
    A.setter(1,0, 3); A.setter(1,1, 4);

    B.setter(0,0, 5); B.setter(0,1, 6);
    B.setter(1,0, 7); B.setter(1,1, 8);

    cout << "A =" << endl << A;
    cout << "B =" << endl << B;
    cout << "A + B =" << endl << (A + B);
    cout << "A * B =" << endl << (A * B);
    cout << "Transpose A =" << endl << A.transpose();
    cout << "Trace A = " << A.trace() << endl;
    cout << "Det A = " << A.determinant() << endl;

    // ==========================================
    // TEST COMPLEXE
    // ==========================================
    cout << "\n=== TEST COMPLEXE ===" << endl;

    complexe c1(3, 4);
    complexe c2(1, -2);

    cout << "c1 = " << c1 << endl;
    cout << "c2 = " << c2 << endl;
    cout << "c1 + c2 = " << (c1 + c2) << endl;
    cout << "c1 - c2 = " << (c1 - c2) << endl;
    cout << "c1 * c2 = " << (c1 * c2) << endl;
    cout << "c1 / c2 = " << (c1 / c2) << endl;
    cout << "Module c1 = " << c1.module() << endl;
    cout << "Argument c1 = " << c1.argument() << " rad" << endl;
    cout << "Conjugue c1 = " << c1.conjugue() << endl;
    cout << "2.0 * c1 = " << (2.0 * c1) << endl;

    // ==========================================
    // TEST REGRESSION LINEAIRE
    // ==========================================
    cout << "\n=== TEST REGRESSION LINEAIRE ===" << endl;

    Matrix X(4, 1);
    X.setter(0, 0, 50);
    X.setter(1, 0, 80);
    X.setter(2, 0, 100);
    X.setter(3, 0, 120);

    Vector y(4);
    y.setter(0, 150000);
    y.setter(1, 220000);
    y.setter(2, 300000);
    y.setter(3, 380000);

    // Normalisation
    Matrix X_norm = normaliserMatrix(X);
    double ymin, ymax;
    Vector y_norm = normaliser(y, ymin, ymax);

    LinearRegression lr;
    lr.entrainer(X_norm, y_norm, 0.01, 10000);
    Vector pred_norm = lr.predire(X_norm);

    // Denormalisation
    Vector pred(4);
    for (int i = 0; i < 4; i++)
        pred.setter(i, pred_norm.getter(i) * (ymax - ymin) + ymin);

    cout << "Predictions  : " << pred << endl;
    cout << "Reelles      : " << y << endl;
    cout << "MSE (norme)  : " << lr.mse(pred_norm, y_norm) << endl;

    // ==========================================
    // TEST KMEANS
    // ==========================================
    cout << "\n=== TEST KMEANS ===" << endl;

    Matrix points(6, 2);
    points.setter(0, 0, 1);   points.setter(0, 1, 1);
    points.setter(1, 0, 1.5); points.setter(1, 1, 1.5);
    points.setter(2, 0, 1);   points.setter(2, 1, 2);
    points.setter(3, 0, 8);   points.setter(3, 1, 8);
    points.setter(4, 0, 9);   points.setter(4, 1, 8);
    points.setter(5, 0, 8);   points.setter(5, 1, 9);

    KMeans km(2);
    km.entrainer(points);
    km.afficherCentroides();

    // ==========================================
    // TEST CSVREADER
    // ==========================================
    cout << "\n=== TEST CSVREADER ===" << endl;

    Matrix data = CVSReader::lireMatrix("houses.csv");
    Matrix X_csv = CVSReader::extraireFeatures(data);
    Vector y_csv = CVSReader::extraireColonne(data, 1);

    // Normalisation
    Matrix X_csv_norm = normaliserMatrix(X_csv);
    double ymin2, ymax2;
    Vector y_csv_norm = normaliser(y_csv, ymin2, ymax2);

    LinearRegression lr2;
    lr2.entrainer(X_csv_norm, y_csv_norm, 0.01, 10000);
    Vector pred_csv_norm = lr2.predire(X_csv_norm);

    // Denormalisation
    Vector pred_csv(y_csv.taille());
    for (int i = 0; i < y_csv.taille(); i++)
        pred_csv.setter(i, pred_csv_norm.getter(i) * (ymax2 - ymin2) + ymin2);

    cout << "Donnees lues :" << endl << data;
    cout << "Predictions  : " << pred_csv << endl;
    cout << "Reelles      : " << y_csv << endl;
    cout << "MSE (norme)  : " << lr2.mse(pred_csv_norm, y_csv_norm) << endl;

    // ==========================================
    // TEST PERCEPTRON
    // ==========================================
    cout << "\n=== TEST PERCEPTRON ===" << endl;

    Matrix Xp(4, 2);
    Xp.setter(0,0,0); Xp.setter(0,1,0);
    Xp.setter(1,0,0); Xp.setter(1,1,1);
    Xp.setter(2,0,1); Xp.setter(2,1,0);
    Xp.setter(3,0,1); Xp.setter(3,1,1);

    Vector yp(4);
    yp.setter(0,0); yp.setter(1,0);
    yp.setter(2,0); yp.setter(3,1);

    Perceptrron p(0.1);
    p.entrainer(Xp, yp, 100);
    Vector predp = p.predire(Xp);
    cout << "Predictions AND : " << predp << endl;
    cout << "Accuracy        : " << p.accuracy(predp, yp) << "%" << endl;

    return 0;
}