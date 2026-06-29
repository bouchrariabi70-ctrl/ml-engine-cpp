"""
db_manager.py
=============
Gestion de la base de données MySQL pour stocker les expériences ML.

Dépendances :
  pip install mysql-connector-python

Configuration :
  Modifier les constantes DB_* ci-dessous selon votre environnement MySQL.
"""

import mysql.connector
from mysql.connector import Error
from datetime import datetime

# ─── Configuration MySQL ──────────────────────────────────────────────────────
DB_HOST     = "localhost"
DB_PORT     = 3306
DB_USER     = "root"
DB_PASSWORD = ""
DB_NAME     = "ml_experiences"

# ─── Schéma SQL ───────────────────────────────────────────────────────────────
SQL_CREATE_DB = f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"

SQL_CREATE_REGRESSION = """
CREATE TABLE IF NOT EXISTS regression_lineaire (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    date_exp      DATETIME        NOT NULL,
    nb_points     INT             NOT NULL,
    bruit         FLOAT           NOT NULL,
    epochs        INT             NOT NULL,
    taux_lr       FLOAT           NOT NULL,
    mse           DOUBLE          NOT NULL,
    rmse          DOUBLE          NOT NULL,
    r2            DOUBLE          NOT NULL,
    notes         VARCHAR(255)    DEFAULT NULL
) ENGINE=InnoDB;
"""

SQL_CREATE_KMEANS = """
CREATE TABLE IF NOT EXISTS kmeans (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    date_exp      DATETIME        NOT NULL,
    k_clusters    INT             NOT NULL,
    nb_points     INT             NOT NULL,
    max_iterations INT            NOT NULL,
    iterations_reelles INT        NOT NULL,
    inertie       DOUBLE          NOT NULL,
    silhouette    DOUBLE          NOT NULL,
    notes         VARCHAR(255)    DEFAULT NULL
) ENGINE=InnoDB;
"""

SQL_CREATE_PERCEPTRON = """
CREATE TABLE IF NOT EXISTS perceptron (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    date_exp      DATETIME        NOT NULL,
    nb_points     INT             NOT NULL,
    epochs        INT             NOT NULL,
    taux_lr       FLOAT           NOT NULL,
    accuracy      DOUBLE          NOT NULL,
    precision_val DOUBLE          NOT NULL,
    rappel        DOUBLE          NOT NULL,
    notes         VARCHAR(255)    DEFAULT NULL
) ENGINE=InnoDB;
"""


class DBManager:
    """Gestionnaire de connexion et d'insertion MySQL."""

    def __init__(self):
        self._conn = None
        self._init_db()

    # ── Connexion & initialisation ─────────────────────────────────────────────
    def _connect(self, with_db: bool = True):
        """Ouvre une connexion MySQL (avec ou sans sélection de la DB)."""
        params = dict(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        if with_db:
            params["database"] = DB_NAME
        return mysql.connector.connect(**params)

    def _init_db(self):
        """Crée la base et les tables si elles n'existent pas encore."""
        try:
            # Créer la base si besoin
            conn = self._connect(with_db=False)
            cur = conn.cursor()
            cur.execute(SQL_CREATE_DB)
            conn.commit()
            cur.close()
            conn.close()

            # Créer les tables
            conn = self._connect()
            cur = conn.cursor()
            for sql in [SQL_CREATE_REGRESSION, SQL_CREATE_KMEANS, SQL_CREATE_PERCEPTRON]:
                cur.execute(sql)
            conn.commit()
            cur.close()
            conn.close()
            print("[DB] Base et tables initialisées avec succès.")
        except Error as e:
            print(f"[DB] Erreur d'initialisation : {e}")

    # ── Insertion : Régression Linéaire ───────────────────────────────────────
    def save_regression(self, nb_points: int, bruit: float, epochs: int,
                        taux_lr: float, mse: float, rmse: float, r2: float,
                        notes: str = None):
        """Sauvegarde une expérience de régression linéaire."""
        sql = """
            INSERT INTO regression_lineaire
                (date_exp, nb_points, bruit, epochs, taux_lr, mse, rmse, r2, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (datetime.now(), nb_points, bruit, epochs, taux_lr, mse, rmse, r2, notes)
        return self._execute_insert(sql, values, "regression_lineaire")

    # ── Insertion : K-Means ───────────────────────────────────────────────────
    def save_kmeans(self, k_clusters: int, nb_points: int, max_iterations: int,
                    iterations_reelles: int, inertie: float, silhouette: float,
                    notes: str = None):
        """Sauvegarde une expérience K-Means."""
        sql = """
            INSERT INTO kmeans
                (date_exp, k_clusters, nb_points, max_iterations,
                 iterations_reelles, inertie, silhouette, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (datetime.now(), k_clusters, nb_points, max_iterations,
                  iterations_reelles, inertie, silhouette, notes)
        return self._execute_insert(sql, values, "kmeans")

    # ── Insertion : Perceptron ────────────────────────────────────────────────
    def save_perceptron(self, nb_points: int, epochs: int, taux_lr: float,
                        accuracy: float, precision_val: float, rappel: float,
                        notes: str = None):
        """Sauvegarde une expérience Perceptron."""
        sql = """
            INSERT INTO perceptron
                (date_exp, nb_points, epochs, taux_lr,
                 accuracy, precision_val, rappel, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (datetime.now(), nb_points, epochs, taux_lr,
                  accuracy, precision_val, rappel, notes)
        return self._execute_insert(sql, values, "perceptron")

    # ── Lecture : historique ──────────────────────────────────────────────────
    def get_history(self, table: str, limit: int = 50) -> list[dict]:
        """Retourne les N dernières expériences d'une table sous forme de dict."""
        allowed = {"regression_lineaire", "kmeans", "perceptron"}
        if table not in allowed:
            raise ValueError(f"Table inconnue : {table}")
        try:
            conn = self._connect()
            cur = conn.cursor(dictionary=True)
            cur.execute(f"SELECT * FROM {table} ORDER BY date_exp DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return rows
        except Error as e:
            print(f"[DB] Erreur lecture {table} : {e}")
            return []

    # ── Helpers internes ──────────────────────────────────────────────────────
    def _execute_insert(self, sql: str, values: tuple, table: str) -> int | None:
        """Exécute un INSERT et retourne l'id inséré, ou None si erreur."""
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(sql, values)
            conn.commit()
            last_id = cur.lastrowid
            cur.close()
            conn.close()
            print(f"[DB] Expérience sauvegardée dans '{table}' (id={last_id}).")
            return last_id
        except Error as e:
            print(f"[DB] Erreur insertion dans '{table}' : {e}")
            return None


# ─── Instance globale (singleton léger) ──────────────────────────────────────
_db_instance: DBManager | None = None


def get_db() -> DBManager | None:
    """
    Retourne l'instance unique de DBManager.
    Retourne None si la connexion échoue (le dashboard continue sans DB).
    """
    global _db_instance
    if _db_instance is None:
        try:
            _db_instance = DBManager()
        except Exception as e:
            print(f"[DB] Impossible de se connecter à MySQL : {e}")
            _db_instance = None
    return _db_instance
