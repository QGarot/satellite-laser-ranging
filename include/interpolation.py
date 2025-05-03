"""
Imported from: https://github.com/QGarot/interpolation-vs-dl/blob/main/src/neville_aitken.py
"""

import numpy as np
import matplotlib.pyplot as plt
from math import cos, pi, floor


def neville(set: list[tuple], x: float) -> float:
    """
    Paramètres :
    - set : liste contenant les points d'interpolations
    - x : valeur en laquelle on cherche à évaluer le polynôme interpolateur
    Retour : le polynôme interpolateur évalué en x selon le schéma de Neville-Aitken
    """

    n = len(set) - 1  # ordre de l'interpolation
    p = np.zeros((n + 1, n + 1))

    # Initialisation :
    # Quelque soit i, le pol. de degré 0 passant par (x_i, y_i) est le pol. constant y_i
    p[:, 0] = [set[i][1] for i in range(n + 1)]

    # Itérations (en respectant l'ordre de calcul)
    for k in range(1, n + 1):
        for i in range(0, n + 1 - k):
            p[i, k] = ((set[i + k][0] - x) * p[i, k - 1] - (set[i][0] - x) * p[i + 1, k - 1]) / (
                        set[i + k][0] - set[i][0])

    return p[0, n]


def get_interpolation_set(f: callable, xj: list) -> list[tuple]:
    """
    Retourne les points d'interpolation (xj, f(xj))
    Paramètres :
    - f : fonction dont on souhaite déterminer les points d'interpolation associés aux xj
    - xj : tableau contenant les abscisses des points d'interpolation
    Retour : [(x0, f(x0)), (x1, f(x1)), ..., (xn, f(xn))]
    """
    yj = [f(x) for x in xj]
    return [(xj[j], yj[j]) for j in range(len(xj))]


def display_interpolation(f: callable, xj: list, R: float) -> None:
    """
    Affiche un graphe représentant la fonction f et son interpolation polynomiale
    Paramètres :
    - f : la fonction que l'on cherche à interpoler
    - xj : liste contenant les points (abscisse) d'interpolation
    - R : rayon de l'intervalle sur lequel on souhaite reprensenter les résultats
    Retour : None
    """
    n = floor(R * 100)  # nombre de points
    pts = np.linspace(-R, R, n)

    # Initialisation (et affichage) des points d'interpolation
    # yj = [f(x) for x in xj]
    set = get_interpolation_set(f, xj)
    # plt.plot(xj, yj, color="tab:orange", marker="o", linestyle="", label="Nœuds")

    # Affichage de la fonction
    fx = np.array([f(x) for x in pts])
    plt.plot(pts, fx, color="tab:blue", label="$f$")

    # Affichage du polynôme d'interpolation
    p = np.array([neville(set, x) for x in pts])
    plt.plot(pts, p, color="tab:orange", linestyle="--", label="$P_{" + str(len(xj) - 1) + "}f$")

    plt.legend(loc="upper right")
    plt.show()


def get_xj(m: int, R: float = 1) -> list:
    """
    Retourne les points (xj) dans l'intervalle [-R, R] pour avoir la meilleure interpolation
    Paramètre :
    - m : nombre de points souhaités
    - R : rayon de l'intervalle dans lequel les points doivent être choisis
    Retour : une liste de points
    """
    return [cos((2 * i + 1) * pi / (2 * m)) * R for i in range(m)]
