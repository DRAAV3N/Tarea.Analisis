#----------------------------------------------------------
# Auxiliary Functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import svd, inv

#---------------------------------------------------------------------
# Calculo de Indice Colineal con Ridge
#---------------------------------------------------------------------

def calcular_vif(X):
    vif = [] #Lista para guardar VIF de cada variable
    for i in range(X.shape[1]):
        y_i = X.iloc[:, i]
        X_otros = X.drop(X.columns[i], axis=1)
        X_otros = np.column_stack([np.ones(X_otros.shape[0]), X_otros])
        beta, _, _, _ = np.linalg.lstsq(X_otros, y_i, rcond=None)
        y_pred = X_otros @ beta
        ss_res = np.sum((y_i - y_pred)**2)
        ss_tot = np.sum((y_i - np.mean(y_i))**2)
        r2 = 1 - (ss_res / ss_tot)
        vif_val = 1 / (1 - r2) if r2 < 1 else np.inf
        vif.append(vif_val)
    return np.array(vif)

def calcular_Ij_ridge(X, y, lambda_opt):
    n = len(y)
    X_values = X.values
    X_with_1 = np.hstack((np.ones((n, 1)), X_values))  # Agrega columna de unos (intercepto)

    # Ridge: (XᵀX + λI)⁻¹ Xᵀ y
    XtX = X_with_1.T @ X_with_1
    ridge = XtX + lambda_opt * np.eye(XtX.shape[0])
    beta = np.linalg.inv(ridge) @ X_with_1.T @ y

    # Predicción y residuos
    y_hat = X_with_1 @ beta
    residuals = y - y_hat
    mse = np.mean(residuals**2)

    # Varianza de los coeficientes (excluye intercepto)
    XtX_inv_ridge = np.linalg.inv(ridge)
    var_betas = mse * np.diag(XtX_inv_ridge)[1:]

    # Calcular VIF como siempre
    vif_vals = calcular_vif(X)

    # Normalización
    P = vif_vals / np.mean(vif_vals)
    Q = var_betas / np.mean(var_betas)

    # Índice colineal ponderado
    Ij = np.sqrt(P * Q)
    return Ij

def coef_ridge(X, y, lambda_opt):
    n = X.shape[0]
    X_with_1 = np.hstack([np.ones((n, 1)), X])  # agregar intercepto
    XtX = X_with_1.T @ X_with_1
    ridge = XtX + lambda_opt * np.eye(XtX.shape[0])
    beta = inv(ridge) @ X_with_1.T @ y
    return beta