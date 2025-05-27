import utility
import trn
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd


def graficar_Ij_todas(Ij_vals, nombres, filename="figure1.png"):
    plt.figure(figsize=(12, 6))
    plt.bar(nombres, Ij_vals, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Índice Colineal Ponderado (Ij)")
    plt.title("Ij de todas las variables")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico guardado como '{filename}'")

def graficar_Ij_seleccionadas(X_sel, y_sel, lambda_opt, filename="figure2.png"):
    Ij_sel = utility.calcular_Ij_ridge(X_sel, y_sel, lambda_opt)
    nombres_sel = X_sel.columns
    plt.figure(figsize=(12, 6))
    plt.bar(nombres_sel, Ij_sel, color="lightgreen")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Índice Colineal Ponderado (Ij)")
    plt.title("Ij de variables seleccionadas")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico guardado como '{filename}'")