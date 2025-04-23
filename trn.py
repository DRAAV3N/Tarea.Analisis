#---------------------------------------------------------------------
# Variable selection and coefficient estimation
#---------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    
#---------------------------------------------------------------------
# Variable selection by use co-linear index


def calcular_vif(X):
    vif = []
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

def calcular_Ij(X, y):
    n = len(y)
    X_with_1 = np.hstack((np.ones((n, 1)), X.values))
    beta = np.linalg.pinv(X_with_1) @ y
    y_hat = X_with_1 @ beta
    residuals = y - y_hat
    mse = np.mean(residuals**2)

    XtX_inv = np.linalg.pinv(X_with_1.T @ X_with_1)
    var_betas = mse * np.diag(XtX_inv)[1:]

    vif_vals = calcular_vif(X)
    P = vif_vals / np.mean(vif_vals)
    Q = var_betas / np.mean(var_betas)

    Ij = np.sqrt(P * Q)
    return Ij, vif_vals, var_betas, beta[1:]

def seleccion_iterativa_colinealidad(df, tau=2.0):
    X = df.drop(columns="Y")
    y = df["Y"].values
    eliminadas = []

    while True:
        Ij, vif_vals, var_betas, coef = calcular_Ij(X, y)
        max_Ij = np.max(Ij)
        if max_Ij <= tau:
            break
        idx_max = np.argmax(Ij)
        var_elim = X.columns[idx_max]
        eliminadas.append((var_elim, max_Ij))
        X = X.drop(columns=var_elim)

    seleccionadas = X.columns.tolist()
    return seleccionadas, eliminadas
#---------------------------------------------------------------------
def main():            
    # Cargar los datos
    df = pd.read_csv('dataset.csv')
    
    # Dividir en train y test
    p_train = 0.80
    cut = int(len(df) * p_train)
    train = df[:cut]
    test = df[cut:]

    train.to_csv("dtrn.csv", index=False)
    test.to_csv("dtst.csv", index=False)

    print("Ejemplos usados para entrenar: ", len(train))
    print("Ejemplos usados para test: ", len(test))
    print(df.head())

    # Proceso de selección
    seleccionadas, eliminadas = seleccion_iterativa_colinealidad(train, tau=2.0)

    # Guardar seleccionadas
    pd.Series(seleccionadas).to_csv("selected_vars.csv", index=False, header=False)

    # Guardar eliminadas con Ij
    df_eliminadas = pd.DataFrame(eliminadas, columns=["Variable", "Ij"])
    df_eliminadas.to_csv("delete_vars.csv", index=False)

    # Calcular coeficientes finales
    X_final = train[seleccionadas]
    y_final = train["Y"].values
    X_with_1 = np.hstack((np.ones((len(X_final), 1)), X_final.values))
    beta_final = np.linalg.pinv(X_with_1) @ y_final

    df_coefs = pd.DataFrame({
        "Variable": seleccionadas,
        "Coef": beta_final[1:]  # sin el intercepto
    })
    df_coefs.to_csv("coefts.csv", index=False)

    print("✅ Archivos guardados:")
    print("- selected_vars.csv")
    print("- delete_vars.csv")
    print("- coefts.csv")

    # Gráfico eliminadas
    if not df_eliminadas.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(df_eliminadas['Variable'], df_eliminadas['Ij'], color='salmon')
        plt.xlabel("Variables eliminadas")
        plt.ylabel("Índice Ij")
        plt.title("Variables eliminadas vs Índice de colinealidad")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("figure1.png")
        plt.close()

    # Gráfico seleccionadas
    # Para obtener Ij de las seleccionadas
    Ij_final, _, _, _ = calcular_Ij(X_final, y_final)
    df_seleccionadas = pd.DataFrame({
        "Variable": seleccionadas,
        "Ij": Ij_final
    })

    plt.figure(figsize=(10, 6))
    plt.bar(df_seleccionadas['Variable'], df_seleccionadas['Ij'], color='seagreen')
    plt.xlabel("Variables seleccionadas")
    plt.ylabel("Índice Ij")
    plt.title("Variables seleccionadas vs Índice de colinealidad")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figure2.png")
    plt.close()

    print("Gráficos generados:")
    print("- figure1.png (eliminadas)")
    print("- figure2.png (seleccionadas)")

if __name__ == '__main__':   
	 main()
# -----------------------------