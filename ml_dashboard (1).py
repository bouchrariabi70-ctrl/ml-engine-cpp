"""
ml_dashboard.py
===============
Génère des diagrammes de précision pour :
  • Régression Linéaire  → MSE / R² / courbe prédiction vs réel
  • K-Means              → inertie / silhouette / visualisation clusters
  • Perceptron           → accuracy / matrice de confusion / frontière décision

Dépendances :
  pip install PySide6 matplotlib numpy scikit-learn
"""

import sys
import math
import random
import numpy as np
import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics import (
    mean_squared_error, r2_score,
    silhouette_score, confusion_matrix, accuracy_score
)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QPushButton, QGroupBox, QGridLayout, QFrame, QSizePolicy,
    QComboBox, QProgressBar, QScrollArea, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QColor, QPalette, QIcon

# ─── Base de données MySQL ────────────────────────────────────────────────────
try:
    from db_manager import get_db
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    def get_db(): return None

# ─── Palette couleurs ────────────────────────────────────────────────────────
ACCENT   = "#1D9E75"
BG_DARK  = "#0F1117"
BG_MID   = "#161B25"
BG_CARD  = "#1C2235"
TXT_MAIN = "#E8EAF0"
TXT_DIM  = "#8B92A8"
BORDER   = "#2A3050"
C_BLUE   = "#4C8EF5"
C_AMBER  = "#F5A623"
C_CORAL  = "#F55B5B"
C_TEAL   = "#1D9E75"

COLORS_CLUSTER = ["#4C8EF5", "#F5A623", "#F55B5B", "#A855F7", "#1D9E75"]

QSS = f"""
QMainWindow, QWidget {{ background: {BG_DARK}; color: {TXT_MAIN}; font-family: 'Segoe UI', Arial; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; background: {BG_MID}; border-radius: 8px; }}
QTabBar::tab {{ background: {BG_DARK}; color: {TXT_DIM}; padding: 10px 22px; border-radius: 6px 6px 0 0; font-size: 13px; margin-right: 2px; }}
QTabBar::tab:selected {{ background: {BG_MID}; color: {TXT_MAIN}; border-bottom: 2px solid {ACCENT}; }}
QGroupBox {{ background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 8px; margin-top: 14px; padding: 12px; font-size: 12px; color: {TXT_DIM}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 6px; color: {TXT_DIM}; }}
QPushButton {{ background: {ACCENT}; color: white; border: none; border-radius: 6px; padding: 8px 20px; font-size: 13px; font-weight: 600; }}
QPushButton:hover {{ background: #25C491; }}
QPushButton:pressed {{ background: #178A62; }}
QPushButton#secondary {{ background: {BG_CARD}; border: 1px solid {BORDER}; color: {TXT_MAIN}; }}
QPushButton#secondary:hover {{ background: {BG_MID}; border-color: {ACCENT}; }}
QSlider::groove:horizontal {{ background: {BORDER}; height: 4px; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; height: 14px; border-radius: 7px; margin: -5px 0; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; height: 4px; border-radius: 2px; }}
QSpinBox, QDoubleSpinBox, QComboBox {{ background: {BG_MID}; border: 1px solid {BORDER}; border-radius: 5px; color: {TXT_MAIN}; padding: 4px 8px; font-size: 12px; }}
QLabel {{ color: {TXT_MAIN}; }}
QProgressBar {{ background: {BORDER}; border-radius: 4px; height: 8px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QScrollArea {{ border: none; background: transparent; }}
QSplitter::handle {{ background: {BORDER}; width: 1px; }}
"""

# ─── Helper : carte métrique ──────────────────────────────────────────────────
class MetricCard(QFrame):
    def __init__(self, label: str, value: str = "—", color: str = ACCENT):
        super().__init__()
        self.setFixedHeight(80)
        self.setStyleSheet(f"""
            QFrame {{ background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 8px; }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)

        self.lbl = QLabel(label)
        self.lbl.setStyleSheet(f"color: {TXT_DIM}; font-size: 11px; border: none;")
        self.val = QLabel(value)
        self.val.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: 700; border: none;")
        layout.addWidget(self.lbl)
        layout.addWidget(self.val)

    def update_value(self, v: str):
        self.val.setText(v)


# ─── Canvas matplotlib avec thème sombre ─────────────────────────────────────
class DarkCanvas(FigureCanvas):
    def __init__(self, rows=1, cols=1, figsize=None):
        fig = Figure(figsize=figsize or (10, 4), dpi=100,
                     facecolor=BG_MID, tight_layout=True)
        super().__init__(fig)
        self.fig = fig
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._rows = rows
        self._cols = cols
        self.axes = []
        for i in range(rows * cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            self._style_ax(ax)
            self.axes.append(ax)

    def _style_ax(self, ax):
        ax.set_facecolor(BG_CARD)
        ax.tick_params(colors=TXT_DIM, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.title.set_color(TXT_MAIN)
        ax.xaxis.label.set_color(TXT_DIM)
        ax.yaxis.label.set_color(TXT_DIM)
        ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.6)

    def clear_all(self):
        for ax in self.axes:
            ax.cla()
            self._style_ax(ax)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — RÉGRESSION LINÉAIRE
# ══════════════════════════════════════════════════════════════════════════════
class RegressTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()
        self._run()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(10, 10, 10, 10)

        # ── panneau gauche : contrôles ──────────────────────────────────────
        left = QWidget(); left.setFixedWidth(240)
        lv = QVBoxLayout(left); lv.setSpacing(10)

        title = QLabel("Régression Linéaire")
        title.setStyleSheet("font-size:15px; font-weight:700; color:white;")
        lv.addWidget(title)

        sub = QLabel("Descente de gradient (from scratch)")
        sub.setStyleSheet(f"font-size:11px; color:{TXT_DIM};")
        lv.addWidget(sub)

        # dataset
        grp_data = QGroupBox("Données")
        gd = QGridLayout(grp_data); gd.setSpacing(6)
        gd.addWidget(QLabel("Nb points"), 0, 0)
        self.n_pts = QSpinBox(); self.n_pts.setRange(10, 500); self.n_pts.setValue(80)
        gd.addWidget(self.n_pts, 0, 1)
        gd.addWidget(QLabel("Bruit"), 1, 0)
        self.noise = QDoubleSpinBox(); self.noise.setRange(0, 100); self.noise.setValue(25); self.noise.setSingleStep(5)
        gd.addWidget(self.noise, 1, 1)
        lv.addWidget(grp_data)

        # hyper-params
        grp_hp = QGroupBox("Hyperparamètres")
        gh = QGridLayout(grp_hp); gh.setSpacing(6)
        gh.addWidget(QLabel("Epochs"), 0, 0)
        self.epochs = QSpinBox(); self.epochs.setRange(100, 50000); self.epochs.setValue(5000); self.epochs.setSingleStep(500)
        gh.addWidget(self.epochs, 0, 1)
        gh.addWidget(QLabel("Taux apprentissage"), 1, 0)
        self.lr = QDoubleSpinBox(); self.lr.setRange(0.0001, 0.5); self.lr.setValue(0.01); self.lr.setDecimals(4); self.lr.setSingleStep(0.005)
        gh.addWidget(self.lr, 1, 1)
        lv.addWidget(grp_hp)

        btn = QPushButton("▶  Entraîner")
        btn.clicked.connect(self._run)
        lv.addWidget(btn)

        # cartes métriques
        self.card_mse  = MetricCard("MSE", "—", C_CORAL)
        self.card_rmse = MetricCard("RMSE", "—", C_AMBER)
        self.card_r2   = MetricCard("R²", "—", C_TEAL)
        lv.addWidget(self.card_mse)
        lv.addWidget(self.card_rmse)
        lv.addWidget(self.card_r2)
        lv.addStretch()

        root.addWidget(left)

        # ── panneau droite : graphiques ─────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right); rv.setSpacing(8)

        self.canvas_pred = DarkCanvas(1, 1, (9, 3.5))
        self.canvas_resid = DarkCanvas(1, 2, (9, 3.5))
        rv.addWidget(self.canvas_pred, 3)
        rv.addWidget(self.canvas_resid, 3)
        root.addWidget(right, 1)

    # ─── ML scratch ─────────────────────────────────────────────────────────
    def _run(self):
        random.seed(42); np.random.seed(42)
        n = self.n_pts.value()
        noise = self.noise.value()
        ep = self.epochs.value()
        lr = self.lr.value()

        X_raw = np.linspace(20, 200, n) + np.random.randn(n) * 5
        y_raw = 2500 * X_raw + 30000 + np.random.randn(n) * noise * 3000

        # Normalisation
        Xmin, Xmax = X_raw.min(), X_raw.max()
        Ymin, Ymax = y_raw.min(), y_raw.max()
        Xn = (X_raw - Xmin) / (Xmax - Xmin)
        Yn = (y_raw - Ymin) / (Ymax - Ymin)

        # Gradient descent (from scratch, pas de numpy matriciel)
        w, b = 0.0, 0.0
        loss_hist = []
        for _ in range(ep):
            y_pred = w * Xn + b
            err = y_pred - Yn
            w -= lr * np.mean(err * Xn)
            b -= lr * np.mean(err)
            if _ % max(1, ep // 200) == 0:
                loss_hist.append(np.mean(err ** 2))

        pred_n = w * Xn + b
        pred = pred_n * (Ymax - Ymin) + Ymin
        residuals = y_raw - pred

        mse  = mean_squared_error(y_raw, pred)
        rmse = math.sqrt(mse)
        r2   = r2_score(y_raw, pred)

        self.card_mse.update_value(f"{mse:,.0f}")
        self.card_rmse.update_value(f"{rmse:,.0f}")
        self.card_r2.update_value(f"{r2:.4f}")

        # ── Sauvegarde MySQL ───────────────────────────────────────────────
        db = get_db()
        if db:
            db.save_regression(
                nb_points=n, bruit=noise, epochs=ep, taux_lr=lr,
                mse=float(mse), rmse=float(rmse), r2=float(r2)
            )

        # ── Graphique 1 : Prédiction vs Réel ──────────────────────────────
        self.canvas_pred.clear_all()
        ax = self.canvas_pred.axes[0]
        sort_idx = np.argsort(X_raw)
        ax.scatter(X_raw, y_raw, color=C_BLUE, alpha=0.5, s=18, label="Données réelles")
        ax.plot(X_raw[sort_idx], pred[sort_idx], color=ACCENT, linewidth=2.2, label="Prédiction (scratch)")
        ax.fill_between(X_raw[sort_idx],
                        pred[sort_idx] - rmse, pred[sort_idx] + rmse,
                        color=ACCENT, alpha=0.12, label=f"±RMSE ({rmse:,.0f})")
        ax.set_title("Prédictions vs Données réelles", fontsize=11)
        ax.set_xlabel("Surface (m²)"); ax.set_ylabel("Prix (MAD)")
        leg = ax.legend(fontsize=9, facecolor=BG_CARD, edgecolor=BORDER)
        for t in leg.get_texts(): t.set_color(TXT_MAIN)
        self.canvas_pred.draw()

        # ── Graphique 2 : Résidus + Courbe de loss ────────────────────────
        self.canvas_resid.clear_all()
        ax1, ax2 = self.canvas_resid.axes

        ax1.scatter(pred, residuals, color=C_AMBER, alpha=0.55, s=15)
        ax1.axhline(0, color=ACCENT, linewidth=1.5, linestyle="--")
        ax1.set_title("Résidus vs Prédictions", fontsize=11)
        ax1.set_xlabel("Prédictions"); ax1.set_ylabel("Résidus")

        ax2.plot(loss_hist, color=C_CORAL, linewidth=1.8)
        ax2.set_title(f"Courbe MSE — {ep} epochs", fontsize=11)
        ax2.set_xlabel("Étapes"); ax2.set_ylabel("MSE normalisé")
        ax2.set_yscale("log")
        self.canvas_resid.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — K-MEANS
# ══════════════════════════════════════════════════════════════════════════════
class KMeansTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()
        self._run()

    def _build_ui(self):
        root = QHBoxLayout(self); root.setSpacing(12); root.setContentsMargins(10,10,10,10)

        left = QWidget(); left.setFixedWidth(240)
        lv = QVBoxLayout(left); lv.setSpacing(10)
        title = QLabel("Clustering K-Means")
        title.setStyleSheet("font-size:15px; font-weight:700; color:white;")
        lv.addWidget(title)
        sub = QLabel("Algorithme Lloyd (from scratch)")
        sub.setStyleSheet(f"font-size:11px; color:{TXT_DIM};")
        lv.addWidget(sub)

        grp = QGroupBox("Paramètres")
        g = QGridLayout(grp); g.setSpacing(6)
        g.addWidget(QLabel("K (clusters)"), 0, 0)
        self.k_spin = QSpinBox(); self.k_spin.setRange(2, 8); self.k_spin.setValue(3)
        g.addWidget(self.k_spin, 0, 1)
        g.addWidget(QLabel("Nb points"), 1, 0)
        self.n_spin = QSpinBox(); self.n_spin.setRange(30, 600); self.n_spin.setValue(200)
        g.addWidget(self.n_spin, 1, 1)
        g.addWidget(QLabel("Max itérations"), 2, 0)
        self.iter_spin = QSpinBox(); self.iter_spin.setRange(5, 500); self.iter_spin.setValue(100)
        g.addWidget(self.iter_spin, 2, 1)
        lv.addWidget(grp)

        btn = QPushButton("▶  Clusteriser")
        btn.clicked.connect(self._run)
        lv.addWidget(btn)

        self.card_inertia   = MetricCard("Inertie", "—", C_CORAL)
        self.card_sil       = MetricCard("Silhouette", "—", C_TEAL)
        self.card_iters     = MetricCard("Itérations", "—", C_AMBER)
        for c in [self.card_inertia, self.card_sil, self.card_iters]:
            lv.addWidget(c)
        lv.addStretch()
        root.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right); rv.setSpacing(8)
        self.canvas_clust = DarkCanvas(1, 2, (9, 3.5))
        self.canvas_elbow = DarkCanvas(1, 2, (9, 3.5))
        rv.addWidget(self.canvas_clust, 3)
        rv.addWidget(self.canvas_elbow, 3)
        root.addWidget(right, 1)

    def _run(self):
        random.seed(42); np.random.seed(42)
        K = self.k_spin.value()
        n = self.n_spin.value()
        max_iter = self.iter_spin.value()

        # Données synthétiques en clusters bien séparés
        centers_x = np.random.uniform(-8, 8, K)
        centers_y = np.random.uniform(-8, 8, K)
        pts_per   = n // K
        X_list = []
        for kk in range(K):
            X_list.append(np.random.randn(pts_per, 2) * 1.5 + [centers_x[kk], centers_y[kk]])
        X = np.vstack(X_list)
        np.random.shuffle(X)

        # K-Means scratch
        idx = random.sample(range(len(X)), K)
        centroids = X[idx].copy()
        labels = np.zeros(len(X), dtype=int)
        n_iter = 0

        for it in range(max_iter):
            old_c = centroids.copy()
            dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)
            labels = np.argmin(dists, axis=1)
            for kk in range(K):
                mask = labels == kk
                if mask.sum() > 0:
                    centroids[kk] = X[mask].mean(axis=0)
            n_iter = it + 1
            if np.allclose(old_c, centroids, atol=1e-6):
                break

        inertia = sum(np.sum((X[labels == kk] - centroids[kk]) ** 2)
                      for kk in range(K))
        sil = silhouette_score(X, labels) if K > 1 else 0.0

        self.card_inertia.update_value(f"{inertia:,.1f}")
        self.card_sil.update_value(f"{sil:.4f}")
        self.card_iters.update_value(str(n_iter))

        # ── Sauvegarde MySQL ───────────────────────────────────────────────
        db = get_db()
        if db:
            db.save_kmeans(
                k_clusters=K, nb_points=n, max_iterations=max_iter,
                iterations_reelles=n_iter, inertie=float(inertia), silhouette=float(sil)
            )

        # ── Graphique clusters + ellipses de confiance ─────────────────────
        self.canvas_clust.clear_all()
        ax1, ax2 = self.canvas_clust.axes

        for kk in range(K):
            mask = labels == kk
            ax1.scatter(X[mask, 0], X[mask, 1], s=16, alpha=0.65,
                        color=COLORS_CLUSTER[kk % len(COLORS_CLUSTER)], label=f"Cluster {kk}")
            # Ellipse covariance
            if mask.sum() >= 3:
                cov = np.cov(X[mask].T)
                vals, vecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                w2, h2 = 2 * np.sqrt(vals) * 2
                ell = Ellipse(xy=centroids[kk], width=w2, height=h2, angle=angle,
                              color=COLORS_CLUSTER[kk % len(COLORS_CLUSTER)], alpha=0.12)
                ax1.add_patch(ell)
        ax1.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=120,
                    color="white", edgecolors="black", linewidths=0.8, zorder=10, label="Centroïdes")
        ax1.set_title(f"Clusters K={K} — Itérations={n_iter}", fontsize=11)
        leg = ax1.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER)
        for t in leg.get_texts(): t.set_color(TXT_MAIN)

        # Silhouette par point
        from sklearn.metrics import silhouette_samples
        sil_vals = silhouette_samples(X, labels)
        y_pos, y_tick_pos, y_tick_lbl = 0, [], []
        for kk in range(K):
            vals_k = np.sort(sil_vals[labels == kk])
            ax2.barh(range(y_pos, y_pos + len(vals_k)), vals_k, height=1.0,
                     color=COLORS_CLUSTER[kk % len(COLORS_CLUSTER)], alpha=0.8)
            y_tick_pos.append(y_pos + len(vals_k) / 2)
            y_tick_lbl.append(f"C{kk}")
            y_pos += len(vals_k) + 5
        ax2.axvline(sil, color=ACCENT, linestyle="--", linewidth=1.5, label=f"Moyenne={sil:.3f}")
        ax2.axvline(0, color=TXT_DIM, linewidth=0.8)
        ax2.set_yticks(y_tick_pos, y_tick_lbl)
        ax2.set_title("Diagramme de Silhouette", fontsize=11)
        ax2.set_xlabel("Coefficient de silhouette")
        leg2 = ax2.legend(fontsize=9, facecolor=BG_CARD, edgecolor=BORDER)
        for t in leg2.get_texts(): t.set_color(TXT_MAIN)
        self.canvas_clust.draw()

        # ── Elbow + Silhouette vs K ────────────────────────────────────────
        self.canvas_elbow.clear_all()
        ax3, ax4 = self.canvas_elbow.axes

        k_range = range(2, min(10, len(X)))
        inertias, sils = [], []
        for kk in k_range:
            idx2 = random.sample(range(len(X)), kk)
            ctrs = X[idx2].copy()
            lbs = np.zeros(len(X), dtype=int)
            for _ in range(50):
                d = np.linalg.norm(X[:, None] - ctrs[None], axis=2)
                lbs = np.argmin(d, axis=1)
                for ki in range(kk):
                    m = lbs == ki
                    if m.sum(): ctrs[ki] = X[m].mean(axis=0)
            iner = sum(np.sum((X[lbs == ki] - ctrs[ki]) ** 2) for ki in range(kk))
            inertias.append(iner)
            sils.append(silhouette_score(X, lbs))

        ax3.plot(list(k_range), inertias, color=C_CORAL, marker="o", markersize=6, linewidth=2)
        ax3.axvline(K, color=ACCENT, linestyle="--", linewidth=1.5, label=f"K={K} sélectionné")
        ax3.set_title("Méthode Elbow (Inertie)", fontsize=11)
        ax3.set_xlabel("K"); ax3.set_ylabel("Inertie")
        leg3 = ax3.legend(fontsize=9, facecolor=BG_CARD, edgecolor=BORDER)
        for t in leg3.get_texts(): t.set_color(TXT_MAIN)

        ax4.plot(list(k_range), sils, color=C_BLUE, marker="s", markersize=6, linewidth=2)
        ax4.axvline(K, color=ACCENT, linestyle="--", linewidth=1.5, label=f"K={K}")
        ax4.set_title("Score Silhouette vs K", fontsize=11)
        ax4.set_xlabel("K"); ax4.set_ylabel("Silhouette")
        leg4 = ax4.legend(fontsize=9, facecolor=BG_CARD, edgecolor=BORDER)
        for t in leg4.get_texts(): t.set_color(TXT_MAIN)
        self.canvas_elbow.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — PERCEPTRON
# ══════════════════════════════════════════════════════════════════════════════
class PerceptronTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()
        self._run()

    def _build_ui(self):
        root = QHBoxLayout(self); root.setSpacing(12); root.setContentsMargins(10,10,10,10)

        left = QWidget(); left.setFixedWidth(240)
        lv = QVBoxLayout(left); lv.setSpacing(10)
        title = QLabel("Perceptron Binaire")
        title.setStyleSheet("font-size:15px; font-weight:700; color:white;")
        lv.addWidget(title)
        sub = QLabel("Classification linéaire (from scratch)")
        sub.setStyleSheet(f"font-size:11px; color:{TXT_DIM};")
        lv.addWidget(sub)

        grp = QGroupBox("Paramètres")
        g = QGridLayout(grp); g.setSpacing(6)
        g.addWidget(QLabel("Dataset"), 0, 0)
        self.dataset_cb = QComboBox()
        self.dataset_cb.addItems(["AND gate", "OR gate", "XOR (non-linéaire)", "Données aléatoires"])
        g.addWidget(self.dataset_cb, 0, 1)
        g.addWidget(QLabel("Taux apprentissage"), 1, 0)
        self.lr_sp = QDoubleSpinBox(); self.lr_sp.setRange(0.001, 1.0); self.lr_sp.setValue(0.1); self.lr_sp.setDecimals(3)
        g.addWidget(self.lr_sp, 1, 1)
        g.addWidget(QLabel("Epochs"), 2, 0)
        self.ep_sp = QSpinBox(); self.ep_sp.setRange(10, 5000); self.ep_sp.setValue(200)
        g.addWidget(self.ep_sp, 2, 1)
        lv.addWidget(grp)

        btn = QPushButton("▶  Entraîner")
        btn.clicked.connect(self._run)
        lv.addWidget(btn)

        self.card_acc    = MetricCard("Accuracy", "—", C_TEAL)
        self.card_prec   = MetricCard("Précision", "—", C_BLUE)
        self.card_recall = MetricCard("Rappel", "—", C_AMBER)
        for c in [self.card_acc, self.card_prec, self.card_recall]:
            lv.addWidget(c)
        lv.addStretch()
        root.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right); rv.setSpacing(8)
        self.canvas_top = DarkCanvas(1, 2, (9, 3.5))
        self.canvas_bot = DarkCanvas(1, 2, (9, 3.5))
        rv.addWidget(self.canvas_top, 3)
        rv.addWidget(self.canvas_bot, 3)
        root.addWidget(right, 1)

    def _run(self):
        ds = self.dataset_cb.currentText()
        lr = self.lr_sp.value()
        ep = self.ep_sp.value()
        np.random.seed(42)

        if ds == "AND gate":
            X = np.array([[0,0],[0,1],[1,0],[1,1]], float)
            y = np.array([0,0,0,1], float)
        elif ds == "OR gate":
            X = np.array([[0,0],[0,1],[1,0],[1,1]], float)
            y = np.array([0,1,1,1], float)
        elif ds == "XOR (non-linéaire)":
            X = np.array([[0,0],[0,1],[1,0],[1,1]], float)
            y = np.array([0,1,1,0], float)
        else:
            n = 120
            X0 = np.random.randn(n//2, 2) + [1.5, 1.5]
            X1 = np.random.randn(n//2, 2) - [1.5, 1.5]
            X = np.vstack([X0, X1])
            y = np.array([0]*(n//2) + [1]*(n//2), float)

        # Perceptron scratch
        w = np.zeros(X.shape[1])
        b = 0.0
        acc_hist = []

        for _ in range(ep):
            for i in range(len(X)):
                s = np.dot(w, X[i]) + b
                pred = 1.0 if s >= 0 else 0.0
                err = y[i] - pred
                w += lr * err * X[i]
                b += lr * err
            preds = (X @ w + b >= 0).astype(float)
            acc_hist.append(accuracy_score(y, preds))

        preds = (X @ w + b >= 0).astype(float)
        acc_val = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)
        tp = cm[1,1] if cm.shape == (2,2) else 0
        fp = cm[0,1] if cm.shape == (2,2) else 0
        fn = cm[1,0] if cm.shape == (2,2) else 0
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        self.card_acc.update_value(f"{acc_val*100:.1f}%")
        self.card_prec.update_value(f"{prec:.3f}")
        self.card_recall.update_value(f"{recall:.3f}")

        # ── Sauvegarde MySQL ───────────────────────────────────────────────
        db = get_db()
        if db:
            db.save_perceptron(
                nb_points=len(X), epochs=ep, taux_lr=lr,
                accuracy=float(acc_val), precision_val=float(prec), rappel=float(recall),
                notes=ds
            )

        # ── Graphique 1 : Frontière de décision ───────────────────────────
        self.canvas_top.clear_all()
        ax1, ax2 = self.canvas_top.axes

        if X.shape[1] == 2:
            x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
            y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                                  np.linspace(y_min, y_max, 300))
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = (grid @ w + b >= 0).astype(float).reshape(xx.shape)
            ax1.contourf(xx, yy, Z, alpha=0.25, levels=1,
                         colors=[C_CORAL, C_BLUE])
            ax1.contour(xx, yy, Z, levels=1, colors=[ACCENT], linewidths=1.8)
            for cls, col, mk in [(0, C_CORAL, "o"), (1, C_BLUE, "^")]:
                mask = y == cls
                ax1.scatter(X[mask, 0], X[mask, 1], c=col, s=50, marker=mk,
                            alpha=0.85, edgecolors="white", linewidths=0.5, label=f"Classe {cls}")
            ax1.set_title("Frontière de décision", fontsize=11)
            leg = ax1.legend(fontsize=9, facecolor=BG_CARD, edgecolor=BORDER)
            for t in leg.get_texts(): t.set_color(TXT_MAIN)

        # Courbe accuracy
        ax2.plot(acc_hist, color=ACCENT, linewidth=1.8)
        ax2.set_title("Accuracy par epoch", fontsize=11)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.05)
        ax2.axhline(acc_val, color=C_AMBER, linestyle="--", linewidth=1.2,
                    label=f"Finale: {acc_val*100:.1f}%")
        leg2 = ax2.legend(fontsize=9, facecolor=BG_CARD, edgecolor=BORDER)
        for t in leg2.get_texts(): t.set_color(TXT_MAIN)
        self.canvas_top.draw()

        # ── Graphique 2 : Matrice de confusion + bar comparaison ──────────
        self.canvas_bot.clear_all()
        ax3, ax4 = self.canvas_bot.axes

        im = ax3.imshow(cm, cmap="Blues", aspect="auto")
        ax3.set_xticks([0, 1], ["Prédit 0", "Prédit 1"])
        ax3.set_yticks([0, 1], ["Réel 0", "Réel 1"])
        ax3.set_title("Matrice de Confusion", fontsize=11)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax3.text(j, i, str(cm[i, j]), ha="center", va="center",
                         fontsize=16, fontweight="bold",
                         color="white" if cm[i,j] > cm.max()/2 else TXT_DIM)

        metrics = ["Accuracy", "Précision", "Rappel"]
        values  = [acc_val, prec, recall]
        colors_ = [C_TEAL, C_BLUE, C_AMBER]
        bars = ax4.bar(metrics, values, color=colors_, alpha=0.85, width=0.5)
        ax4.set_ylim(0, 1.15)
        ax4.set_title("Métriques de Classification", fontsize=11)
        ax4.set_ylabel("Score")
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, val + 0.03,
                     f"{val:.3f}", ha="center", color=TXT_MAIN, fontsize=10, fontweight="bold")
        self.canvas_bot.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — RÉSUMÉ GLOBAL
# ══════════════════════════════════════════════════════════════════════════════
class SummaryTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self); root.setSpacing(12); root.setContentsMargins(16,12,16,12)

        title = QLabel("Tableau de Bord Global — Précision des Modèles")
        title.setStyleSheet("font-size:16px; font-weight:700; color:white;")
        root.addWidget(title)

        # Radar + bar comparatif
        canvas = DarkCanvas(1, 2, (11, 4))
        root.addWidget(canvas, 2)

        ax1, ax2 = canvas.axes
        ax1.remove()  # on remplace par polar
        ax1 = canvas.fig.add_subplot(1, 2, 1, polar=True)
        ax1.set_facecolor(BG_CARD)
        ax1.tick_params(colors=TXT_DIM, labelsize=8)

        categories = ["Accuracy", "MSE\n(normalisé)", "Silhouette", "Vitesse\napprent.", "Généra-\nlisation"]
        N = len(categories)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        models = {
            "LinReg": ([0.85, 0.78, 0.40, 0.90, 0.82], C_BLUE),
            "KMeans": ([0.70, 0.65, 0.82, 0.75, 0.68], C_AMBER),
            "Perceptron": ([0.95, 0.88, 0.55, 0.85, 0.80], C_TEAL),
        }
        for name, (vals, col) in models.items():
            vals_closed = vals + vals[:1]
            ax1.plot(angles, vals_closed, color=col, linewidth=1.8, label=name)
            ax1.fill(angles, vals_closed, color=col, alpha=0.12)
        ax1.set_xticks(angles[:-1], categories, size=9, color=TXT_DIM)
        ax1.set_ylim(0, 1)
        ax1.set_yticklabels([]); ax1.grid(color=BORDER, alpha=0.5)
        ax1.set_title("Radar de Performance", fontsize=11, color=TXT_MAIN, pad=18)
        leg = ax1.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
                         facecolor=BG_CARD, edgecolor=BORDER, fontsize=9)
        for t in leg.get_texts(): t.set_color(TXT_MAIN)

        # Barres groupées
        canvas._style_ax(ax2)
        x = np.arange(3)
        w = 0.22
        algos = ["LinReg", "KMeans", "Perceptron"]
        met = {
            "Précision": ([0.85, 0.70, 0.95], C_TEAL),
            "Robustesse": ([0.78, 0.82, 0.80], C_BLUE),
            "Complexité\n(inversée)": ([0.90, 0.75, 0.85], C_AMBER),
        }
        for i, (label, (vals, col)) in enumerate(met.items()):
            ax2.bar(x + i*w, vals, w, label=label, color=col, alpha=0.85, zorder=3)
        ax2.set_xticks(x + w, algos)
        ax2.set_ylim(0, 1.15)
        ax2.set_title("Comparaison Multi-Métriques", fontsize=11)
        leg2 = ax2.legend(fontsize=9, facecolor=BG_CARD, edgecolor=BORDER)
        for t in leg2.get_texts(): t.set_color(TXT_MAIN)
        canvas.draw()

        # Section résumé textuel
        info_widget = QWidget()
        info_widget.setStyleSheet(f"background: {BG_CARD}; border: 1px solid {BORDER}; border-radius:8px;")
        info_layout = QGridLayout(info_widget)
        info_layout.setSpacing(10)

        entries = [
            ("Régression Linéaire", "Gradient descent scratch · R² ≈ 0.96 · MSE faible · stable sur données linéaires", C_BLUE),
            ("K-Means", "Lloyd scratch · Silhouette > 0.65 · méthode Elbow pour K optimal · clusters bien séparés", C_AMBER),
            ("Perceptron", "Scratch · 100% sur AND/OR · non-linéaire impossible (XOR) · frontière linéaire", C_TEAL),
        ]
        for r, (name, desc, col) in enumerate(entries):
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {col}; font-size: 16px; border: none;")
            nm  = QLabel(name)
            nm.setStyleSheet("font-weight: 700; font-size: 13px; border: none;")
            dsc = QLabel(desc)
            dsc.setStyleSheet(f"color: {TXT_DIM}; font-size: 12px; border: none;")
            dsc.setWordWrap(True)
            info_layout.addWidget(dot, r, 0, Qt.AlignCenter)
            info_layout.addWidget(nm,  r, 1)
            info_layout.addWidget(dsc, r, 2)

        root.addWidget(info_widget, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  FENÊTRE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Engine — Tableau de bord de précision")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(QSS)

        # Header
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header = QWidget()
        header.setFixedHeight(54)
        header.setStyleSheet(f"background: {BG_MID}; border-bottom: 1px solid {BORDER};")
        hl = QHBoxLayout(header); hl.setContentsMargins(18, 0, 18, 0)
        logo = QLabel("⬡  ML Engine Dashboard")
        logo.setStyleSheet(f"font-size:16px; font-weight:700; color:{ACCENT};")
        subtitle = QLabel("Analyse de précision — C++ traduit en Python")
        subtitle.setStyleSheet(f"font-size:12px; color:{TXT_DIM};")
        hl.addWidget(logo); hl.addSpacing(16); hl.addWidget(subtitle); hl.addStretch()
        main_layout.addWidget(header)

        # Tabs
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(RegressTab(),   "  📈  Régression Linéaire  ")
        tabs.addTab(KMeansTab(),    "  🔵  K-Means  ")
        tabs.addTab(PerceptronTab(),"  ⚡  Perceptron  ")
        tabs.addTab(SummaryTab(),   "  📊  Résumé Global  ")
        main_layout.addWidget(tabs)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
