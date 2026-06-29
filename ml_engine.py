"""
ml_engine.py
============
Traduction Python fidèle du projet C++ ml-engine-cpp
(github.com/bouchrariabi70-ctrl/ml-engine-cpp)

Modules :
  - Vector       → class Vector
  - Matrix       → class Matrix
  - Complexe     → class Complexe
  - LinearRegression → class LinearRegression
  - KMeans       → class KMeans
  - Perceptron   → class Perceptron
  - CSVReader    → fonctions read_matrix / extract_features / extract_column
"""

import math
import csv
import random


# ─────────────────────────────────────────────
# 1.  VECTOR
# ─────────────────────────────────────────────
class Vector:
    """Vecteur de taille fixe, équivalent à Vector.h / Vector.cpp"""

    def __init__(self, size: int, default: float = 0.0):
        self._data = [default] * size

    # ── accès ──────────────────────────────────
    def getter(self, i: int) -> float:
        return self._data[i]

    def setter(self, i: int, val: float):
        self._data[i] = val

    def taille(self) -> int:
        return len(self._data)

    # ── opérations vectorielles ─────────────────
    def __add__(self, other: "Vector") -> "Vector":
        assert self.taille() == other.taille()
        result = Vector(self.taille())
        for i in range(self.taille()):
            result.setter(i, self._data[i] + other._data[i])
        return result

    def __sub__(self, other: "Vector") -> "Vector":
        assert self.taille() == other.taille()
        result = Vector(self.taille())
        for i in range(self.taille()):
            result.setter(i, self._data[i] - other._data[i])
        return result

    def __rmul__(self, scalar: float) -> "Vector":
        result = Vector(self.taille())
        for i in range(self.taille()):
            result.setter(i, scalar * self._data[i])
        return result

    def produit_scalaire(self, other: "Vector") -> float:
        assert self.taille() == other.taille()
        return sum(self._data[i] * other._data[i] for i in range(self.taille()))

    def norme(self) -> float:
        return math.sqrt(self.produit_scalaire(self))

    def distance(self, other: "Vector") -> float:
        return (self - other).norme()

    def __repr__(self) -> str:
        vals = " ".join(f"{v:.4f}" for v in self._data)
        return f"[{vals}]"


# ─────────────────────────────────────────────
# 2.  MATRIX
# ─────────────────────────────────────────────
class Matrix:
    """Matrice lignes×colonnes, équivalent à Matrix.h / Matrix.cpp"""

    def __init__(self, rows: int, cols: int, default: float = 0.0):
        self._rows = rows
        self._cols = cols
        self._data = [[default] * cols for _ in range(rows)]

    # ── accès ──────────────────────────────────
    def getter(self, i: int, j: int) -> float:
        return self._data[i][j]

    def setter(self, i: int, j: int, val: float):
        self._data[i][j] = val

    def lignes(self) -> int:
        return self._rows

    def colonnes(self) -> int:
        return self._cols

    # ── opérations ─────────────────────────────
    def __add__(self, other: "Matrix") -> "Matrix":
        assert self._rows == other._rows and self._cols == other._cols
        result = Matrix(self._rows, self._cols)
        for i in range(self._rows):
            for j in range(self._cols):
                result.setter(i, j, self._data[i][j] + other._data[i][j])
        return result

    def __mul__(self, other: "Matrix") -> "Matrix":
        assert self._cols == other._rows
        result = Matrix(self._rows, other._cols)
        for i in range(self._rows):
            for j in range(other._cols):
                s = sum(self._data[i][k] * other._data[k][j]
                        for k in range(self._cols))
                result.setter(i, j, s)
        return result

    def transpose(self) -> "Matrix":
        result = Matrix(self._cols, self._rows)
        for i in range(self._rows):
            for j in range(self._cols):
                result.setter(j, i, self._data[i][j])
        return result

    def trace(self) -> float:
        assert self._rows == self._cols
        return sum(self._data[i][i] for i in range(self._rows))

    def determinant(self) -> float:
        """Développement de Laplace récursif (comme en C++)"""
        n = self._rows
        assert n == self._cols
        if n == 1:
            return self._data[0][0]
        if n == 2:
            return (self._data[0][0] * self._data[1][1]
                    - self._data[0][1] * self._data[1][0])
        det = 0.0
        for j in range(n):
            sub = Matrix(n - 1, n - 1)
            for si in range(1, n):
                sj_dst = 0
                for sj in range(n):
                    if sj == j:
                        continue
                    sub.setter(si - 1, sj_dst, self._data[si][sj])
                    sj_dst += 1
            sign = 1 if j % 2 == 0 else -1
            det += sign * self._data[0][j] * sub.determinant()
        return det

    def get_row_as_vector(self, i: int) -> Vector:
        v = Vector(self._cols)
        for j in range(self._cols):
            v.setter(j, self._data[i][j])
        return v

    def __repr__(self) -> str:
        lines = []
        for row in self._data:
            lines.append("  " + "  ".join(f"{v:8.4f}" for v in row))
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 3.  COMPLEXE
# ─────────────────────────────────────────────
class Complexe:
    """Nombre complexe, équivalent à complexe.h / complexe.cpp"""

    def __init__(self, reel: float = 0.0, img: float = 0.0):
        self.reel = reel
        self.img = img

    def __add__(self, other: "Complexe") -> "Complexe":
        return Complexe(self.reel + other.reel, self.img + other.img)

    def __sub__(self, other: "Complexe") -> "Complexe":
        return Complexe(self.reel - other.reel, self.img - other.img)

    def __mul__(self, other: "Complexe") -> "Complexe":
        return Complexe(
            self.reel * other.reel - self.img * other.img,
            self.reel * other.img + self.img * other.reel,
        )

    def __rmul__(self, scalar: float) -> "Complexe":
        return Complexe(scalar * self.reel, scalar * self.img)

    def __truediv__(self, other: "Complexe") -> "Complexe":
        denom = other.reel ** 2 + other.img ** 2
        return Complexe(
            (self.reel * other.reel + self.img * other.img) / denom,
            (self.img * other.reel - self.reel * other.img) / denom,
        )

    def module(self) -> float:
        return math.sqrt(self.reel ** 2 + self.img ** 2)

    def argument(self) -> float:
        return math.atan2(self.img, self.reel)

    def conjugue(self) -> "Complexe":
        return Complexe(self.reel, -self.img)

    def __repr__(self) -> str:
        sign = "+" if self.img >= 0 else "-"
        return f"({self.reel:.4f} {sign} {abs(self.img):.4f}i)"


# ─────────────────────────────────────────────
# 4.  LINEAR REGRESSION (gradient descent)
# ─────────────────────────────────────────────
class LinearRegression:
    """Régression linéaire par descente de gradient,
    équivalent à LinearRegression.h / LinearRegression.cpp"""

    def __init__(self):
        self.poids = Vector(1)
        self.biais = 0.0

    def entrainer(self, X: Matrix, y: Vector, lr: float = 0.01, epochs: int = 10000):
        n = X.lignes()
        m = X.colonnes()
        self.poids = Vector(m)
        self.biais = 0.0

        for _ in range(epochs):
            y_pred = self.predire(X)

            # gradients poids
            for j in range(m):
                grad = sum(
                    (y_pred.getter(i) - y.getter(i)) * X.getter(i, j)
                    for i in range(n)
                )
                self.poids.setter(j, self.poids.getter(j) - lr * grad / n)

            # gradient biais
            grad_b = sum(y_pred.getter(i) - y.getter(i) for i in range(n))
            self.biais -= lr * grad_b / n

    def predire(self, X: Matrix) -> Vector:
        n = X.lignes()
        result = Vector(n, self.biais)
        for i in range(n):
            s = self.biais
            for j in range(X.colonnes()):
                s += X.getter(i, j) * self.poids.getter(j)
            result.setter(i, s)
        return result

    def mse(self, y_pred: Vector, y_reel: Vector) -> float:
        s = sum(
            (y_pred.getter(i) - y_reel.getter(i)) ** 2
            for i in range(y_pred.taille())
        )
        return s / y_pred.taille()

    def afficher(self):
        print(f"Poids : {self.poids}")
        print(f"Biais : {self.biais:.6f}")


# ─────────────────────────────────────────────
# 5.  K-MEANS
# ─────────────────────────────────────────────
class KMeans:
    """K-Means clustering, équivalent à Kmeans.h / Kmeans.cpp"""

    def __init__(self, k: int, max_iter: int = 100):
        self.k = k
        self.max_iter = max_iter
        self.centroides: list[Vector] = []

    def entrainer(self, X: Matrix):
        n = X.lignes()
        # initialisation aléatoire des centroïdes
        indices = random.sample(range(n), self.k)
        self.centroides = [X.get_row_as_vector(i) for i in indices]

        for _ in range(self.max_iter):
            clusters = [[] for _ in range(self.k)]

            # affectation
            for i in range(n):
                point = X.get_row_as_vector(i)
                distances = [point.distance(c) for c in self.centroides]
                clusters[distances.index(min(distances))].append(i)

            # mise à jour centroïdes
            nouveaux = []
            for ki in range(self.k):
                if not clusters[ki]:
                    nouveaux.append(self.centroides[ki])
                    continue
                dim = X.colonnes()
                nouveau = Vector(dim)
                for i in clusters[ki]:
                    for j in range(dim):
                        nouveau.setter(j, nouveau.getter(j) + X.getter(i, j))
                for j in range(dim):
                    nouveau.setter(j, nouveau.getter(j) / len(clusters[ki]))
                nouveaux.append(nouveau)
            self.centroides = nouveaux

    def afficher_centroides(self):
        for ki, c in enumerate(self.centroides):
            print(f"  Centroïde {ki} : {c}")

    def predire(self, X: Matrix) -> list[int]:
        labels = []
        for i in range(X.lignes()):
            point = X.get_row_as_vector(i)
            distances = [point.distance(c) for c in self.centroides]
            labels.append(distances.index(min(distances)))
        return labels


# ─────────────────────────────────────────────
# 6.  PERCEPTRON
# ─────────────────────────────────────────────
class Perceptron:
    """Perceptron binaire, équivalent à Perceptrron.h / Perceptrron.cpp"""

    def __init__(self, lr: float = 0.1):
        self.lr = lr
        self.poids = Vector(1)
        self.biais = 0.0

    @staticmethod
    def _activation(x: float) -> float:
        return 1.0 if x >= 0.0 else 0.0

    def entrainer(self, X: Matrix, y: Vector, epochs: int = 100):
        m = X.colonnes()
        self.poids = Vector(m)
        self.biais = 0.0

        for _ in range(epochs):
            for i in range(X.lignes()):
                s = self.biais + sum(
                    X.getter(i, j) * self.poids.getter(j) for j in range(m)
                )
                pred = self._activation(s)
                err = y.getter(i) - pred
                for j in range(m):
                    self.poids.setter(
                        j, self.poids.getter(j) + self.lr * err * X.getter(i, j)
                    )
                self.biais += self.lr * err

    def predire(self, X: Matrix) -> Vector:
        n = X.lignes()
        result = Vector(n)
        for i in range(n):
            s = self.biais + sum(
                X.getter(i, j) * self.poids.getter(j) for j in range(X.colonnes())
            )
            result.setter(i, self._activation(s))
        return result

    def accuracy(self, y_pred: Vector, y_reel: Vector) -> float:
        correct = sum(
            1 for i in range(y_pred.taille())
            if y_pred.getter(i) == y_reel.getter(i)
        )
        return 100.0 * correct / y_pred.taille()


# ─────────────────────────────────────────────
# 7.  CSV READER
# ─────────────────────────────────────────────
class CSVReader:
    """Lecture de fichiers CSV, équivalent à CVSReader.h / CVSReader.cpp"""

    @staticmethod
    def lire_matrix(filename: str) -> Matrix:
        rows = []
        with open(filename, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header si présent
            for row in reader:
                try:
                    rows.append([float(v) for v in row])
                except ValueError:
                    continue
        if not rows:
            return Matrix(0, 0)
        m = Matrix(len(rows), len(rows[0]))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                m.setter(i, j, val)
        return m

    @staticmethod
    def extraire_features(data: Matrix, col_cible: int = 1) -> Matrix:
        """Retourne toutes les colonnes sauf col_cible"""
        cols = [j for j in range(data.colonnes()) if j != col_cible]
        result = Matrix(data.lignes(), len(cols))
        for i in range(data.lignes()):
            for dst, src in enumerate(cols):
                result.setter(i, dst, data.getter(i, src))
        return result

    @staticmethod
    def extraire_colonne(data: Matrix, col: int) -> Vector:
        v = Vector(data.lignes())
        for i in range(data.lignes()):
            v.setter(i, data.getter(i, col))
        return v


# ─────────────────────────────────────────────
# 8.  UTILITAIRES (normalisation)
# ─────────────────────────────────────────────
def normaliser_vector(v: Vector) -> tuple[Vector, float, float]:
    vmin = min(v.getter(i) for i in range(v.taille()))
    vmax = max(v.getter(i) for i in range(v.taille()))
    result = Vector(v.taille())
    rang = vmax - vmin if vmax != vmin else 1.0
    for i in range(v.taille()):
        result.setter(i, (v.getter(i) - vmin) / rang)
    return result, vmin, vmax


def normaliser_matrix(M: Matrix) -> Matrix:
    result = Matrix(M.lignes(), M.colonnes())
    for j in range(M.colonnes()):
        vmin = min(M.getter(i, j) for i in range(M.lignes()))
        vmax = max(M.getter(i, j) for i in range(M.lignes()))
        rang = vmax - vmin if vmax != vmin else 1.0
        for i in range(M.lignes()):
            result.setter(i, j, (M.getter(i, j) - vmin) / rang)
    return result


def denormaliser_vector(v_norm: Vector, vmin: float, vmax: float) -> Vector:
    result = Vector(v_norm.taille())
    for i in range(v_norm.taille()):
        result.setter(i, v_norm.getter(i) * (vmax - vmin) + vmin)
    return result


# ─────────────────────────────────────────────
# 9.  MAIN — reproduction exacte du main.cpp
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── TEST VECTOR ────────────────────────────
    print("=== TEST VECTOR ===")
    v1, v2 = Vector(3), Vector(3)
    for i, val in enumerate([1.0, 2.0, 3.0]):
        v1.setter(i, val)
    for i, val in enumerate([4.0, 5.0, 6.0]):
        v2.setter(i, val)
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 - v2 = {v1 - v2}")
    print(f"Produit scalaire = {v1.produit_scalaire(v2):.4f}")
    print(f"Norme v1 = {v1.norme():.4f}")
    print(f"2.5 * v1 = {2.5 * v1}")

    # ── TEST MATRIX ────────────────────────────
    print("\n=== TEST MATRIX ===")
    A, B = Matrix(2, 2), Matrix(2, 2)
    for (i, j, v) in [(0,0,1),(0,1,2),(1,0,3),(1,1,4)]:
        A.setter(i, j, v)
    for (i, j, v) in [(0,0,5),(0,1,6),(1,0,7),(1,1,8)]:
        B.setter(i, j, v)
    print(f"A =\n{A}")
    print(f"B =\n{B}")
    print(f"A + B =\n{A + B}")
    print(f"A * B =\n{A * B}")
    print(f"Transpose A =\n{A.transpose()}")
    print(f"Trace A = {A.trace():.4f}")
    print(f"Det A = {A.determinant():.4f}")

    # ── TEST COMPLEXE ──────────────────────────
    print("\n=== TEST COMPLEXE ===")
    c1, c2 = Complexe(3, 4), Complexe(1, -2)
    print(f"c1 = {c1}")
    print(f"c2 = {c2}")
    print(f"c1 + c2 = {c1 + c2}")
    print(f"c1 - c2 = {c1 - c2}")
    print(f"c1 * c2 = {c1 * c2}")
    print(f"c1 / c2 = {c1 / c2}")
    print(f"Module c1 = {c1.module():.4f}")
    print(f"Argument c1 = {c1.argument():.4f} rad")
    print(f"Conjugué c1 = {c1.conjugue()}")
    print(f"2.0 * c1 = {2.0 * c1}")

    # ── TEST REGRESSION LINEAIRE ───────────────
    print("\n=== TEST REGRESSION LINEAIRE ===")
    X = Matrix(4, 1)
    for i, v in enumerate([50, 80, 100, 120]):
        X.setter(i, 0, v)
    y = Vector(4)
    for i, v in enumerate([150000, 220000, 300000, 380000]):
        y.setter(i, v)

    X_norm = normaliser_matrix(X)
    y_norm, ymin, ymax = normaliser_vector(y)

    lr = LinearRegression()
    lr.entrainer(X_norm, y_norm, 0.01, 10000)
    pred_norm = lr.predire(X_norm)
    pred = denormaliser_vector(pred_norm, ymin, ymax)

    print(f"Prédictions : {pred}")
    print(f"Réelles     : {y}")
    print(f"MSE (normalisé) : {lr.mse(pred_norm, y_norm):.6f}")

    # ── TEST KMEANS ────────────────────────────
    print("\n=== TEST KMEANS ===")
    points = Matrix(6, 2)
    data_pts = [(1,1),(1.5,1.5),(1,2),(8,8),(9,8),(8,9)]
    for i, (x, yv) in enumerate(data_pts):
        points.setter(i, 0, x)
        points.setter(i, 1, yv)
    km = KMeans(2)
    random.seed(42)
    km.entrainer(points)
    print("Centroïdes :")
    km.afficher_centroides()

    # ── TEST PERCEPTRON ────────────────────────
    print("\n=== TEST PERCEPTRON (AND gate) ===")
    Xp = Matrix(4, 2)
    for i, (a, b) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        Xp.setter(i, 0, a)
        Xp.setter(i, 1, b)
    yp = Vector(4)
    for i, v in enumerate([0, 0, 0, 1]):
        yp.setter(i, v)

    p = Perceptron(0.1)
    p.entrainer(Xp, yp, 100)
    predp = p.predire(Xp)
    print(f"Prédictions AND : {predp}")
    print(f"Accuracy : {p.accuracy(predp, yp):.1f}%")
