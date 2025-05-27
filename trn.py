#---------------------------------------------------------------------
# Variable selection and coefficient estimation
#---------------------------------------------------------------------
import utility
import plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import svd, inv
  
#---------------------------------------------------------------------
# Variable selection by use co-linear index
#---------------------------------------------------------------------
def GCV(X, y, lambda_min, lambda_max, n_lambda):
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambda)
    lambda_opt = lambdas[0]
    min_gcv = np.inf

    U, s, Vt = svd(X, full_matrices=False)
    y_t = U.T @ y
    N = X.shape[0]

    for l in lambdas:
        d = s**2 / (s**2 + l)
        df = np.sum(d)
        y_hat = d * y_t
        gcv = np.sum((y_t - y_hat)**2) / (N - df)**2

        if gcv < min_gcv:
            min_gcv = gcv
            lambda_opt = l

    return  lambda_opt

def Selected_Vars(df, lambda_opt, topK):
    x = df.drop(columns="Y").copy()
    y = df["Y"].values
    

    while x.shape[1] > topK:
        Ij = utility.calcular_Ij_ridge(x, y, lambda_opt)
        idx_max = np.argmax(Ij)
        col_elim = x.columns[idx_max]
        print(f"Eliminando: {col_elim} con Ij = {Ij[idx_max]:.4f}")
        x = x.drop(columns=[col_elim])

    # Guardar variables seleccionadas
    selected_vars = pd.DataFrame({'variable': x.columns})
    selected_vars.to_csv("selected_vars.csv", index=False)
    print("Variables seleccionadas guardadas en 'selected_vars.csv'")
    return x, y

def guardar_coefts(X, y, lambda_opt, filename="coefts.csv"):
    n = len(y)
    X_with_1 = np.hstack([np.ones((n, 1)), X])
    I = np.eye(X_with_1.shape[1])
    I[0, 0] = 0  # no regularizar intercepto
    XtX = X_with_1.T @ X_with_1
    beta = np.linalg.inv(XtX + lambda_opt * I) @ X_with_1.T @ y

    df_coef = pd.DataFrame([beta])
    df_lambda = pd.DataFrame([[lambda_opt] + [None]*(len(beta)-1)])
    df_salida = pd.concat([df_coef, df_lambda], ignore_index=True)
    df_salida.to_csv(filename, index=False, header=False)
    print(f"Coeficientes y lambda óptimo guardados en '{filename}'")


def main():            
    # Cargar los datos
    df = pd.read_csv('dataset.csv')

    # Cargar configuración
    config = pd.read_csv('cfg_lambda.csv').iloc[0]
    lambda_min = config['Lambda_min']
    lambda_max = config['Lambda_max']
    n_lambda = int(config['Cantidad_Lambda'])
    topK = int(config['TopK'])
    
    # Dividir en train y test -----------------------
    p_train = 0.80
    cut = int(len(df) * p_train)
    train = df[:cut]
    test = df[cut:]

    train.to_csv("dtrn.csv", index=False)
    test.to_csv("dtst.csv", index=False)

    print("Ejemplos usados para entrenar: ", len(train))
    print("Ejemplos usados para test: ", len(test))
    print(df.head())
    #--------------------------------------------------

    print(df.columns)

    #Elimina variable correspondiente al valor
    X = df.drop(columns="Y").copy()
    y = df["Y"].values

    lambda_opt = GCV(X, y, lambda_min, lambda_max, n_lambda)

    X_sel, y_sel = Selected_Vars(df, lambda_opt, topK)
    guardar_coefts(X_sel.values, y_sel, lambda_opt)
    
    Ij_todas = utility.calcular_Ij_ridge(X, df["Y"].values, lambda_opt)
    # Guardar selected_vars.csv

    plot.graficar_Ij_todas(Ij_todas, X.columns, "figure1.png")
    plot.graficar_Ij_seleccionadas(X_sel, y_sel, lambda_opt, "figure2.png")


if __name__ == '__main__':   
	 main()
# -----------------------------