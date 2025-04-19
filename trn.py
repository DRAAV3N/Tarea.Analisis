#---------------------------------------------------------------------
# Variable selection and coefficient estimation
#---------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    
#---------------------------------------------------------------------
# Variable selection by use co-linear index


def compute_vif(X):
    vif = []
    for j in range(X.shape[1]):
        X_j = X[:, j]
        X_rest = np.delete(X, j, axis=1)

        beta_j = np.linalg.pinv(X_rest) @ X_j
        X_j_hat = X_rest @ beta_j

        corr = np.corrcoef(X_j, X_j_hat)[0, 1]
        R2_j = corr**2 if not np.isnan(corr) else 0.0
        vif_j = 1 / (1 - R2_j) if R2_j < 1 else np.inf
        vif.append(vif_j)
    return np.array(vif)

def selection_vars():
    df = pd.read_csv("dtrn.csv")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    variable_names = df.columns[:-1]
    n, m = X.shape

    vif = compute_vif(X)

    X_with_1 = np.hstack((np.ones((n, 1)), X))
    beta = np.linalg.pinv(X_with_1) @ y

    y_hat = X_with_1 @ beta
    residuals = y - y_hat
    mse = np.mean(residuals**2)

    XtX_inv = np.linalg.pinv(X_with_1.T @ X_with_1)
    var_betas = mse * np.diag(XtX_inv)[1:]  # sin intercepto
    mean_var_beta = np.mean(var_betas)
    Q = var_betas / mean_var_beta

    Ij = np.sqrt(vif * Q)

    df_result = pd.DataFrame({
        'Variable': variable_names,
        'Coef': beta[1:],      # sin intercepto
        'VIF': vif,
        'Var(beta)': var_betas,
        'Qj': Q,
        'Ij': Ij
    })

    df_result.to_csv("indice_colinealidad.csv", index=False)

    return df_result
       
#---------------------------------------------------------------------
def main():            
    #Cargar los datos usando pandas
    df = pd.read_csv('dataset.csv')
    
    #division de los porcentajes de datos
    p_train = 0.80
    cut = int(len(df) * p_train)
    train = df[:cut]
    test = df[cut:]

    #Guardar en archivos CSV
    train.to_csv("dtrn.csv", index=False)
    test.to_csv("dtst.csv", index=False)

    print("Ejemplos usados para entrenar: ", len(train))
    print("Ejemplos usados para test: ", len(test))
    print(df.head())

    df_result = selection_vars()

    #Define el threshold aquí
    threshold = 10.0

    # Selección según umbral de Ij
    seleccionadas = df_result[df_result['Ij'] <= threshold]
    eliminadas = df_result[df_result['Ij'] > threshold]

    # Guardar archivos solicitados
    seleccionadas['Variable'].to_csv("selected_vars.csv", index=False, header=False)
    eliminadas['Variable'].to_csv("delete_vars.csv", index=False, header=False)
    seleccionadas[['Variable', 'Coef']].to_csv("coefts.csv", index=False)

    print("✅ Archivos guardados:")
    print("- selected_vars.csv")
    print("- delete_vars.csv")
    print("- coefts.csv")

    # Gráfico de variables eliminadas
    plt.figure(figsize=(10, 6))
    plt.bar(eliminadas['Variable'], eliminadas['Ij'], color='salmon')
    plt.xlabel("Variables eliminadas")
    plt.ylabel("Índice Ij")
    plt.title("Variables eliminadas vs Índice de colinealidad")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figure1.png")
    plt.close()

    # Gráfico de variables seleccionadas
    plt.figure(figsize=(10, 6))
    plt.bar(seleccionadas['Variable'], seleccionadas['Ij'], color='seagreen')
    plt.xlabel("Variables seleccionadas")
    plt.ylabel("Índice Ij")
    plt.title("Variables seleccionadas vs Índice de colinealidad")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figure2.png")
    plt.close()

    print("📊 Gráficos generados:")
    print("- figure1.png (eliminadas)")
    print("- figure2.png (seleccionadas)")


        

if __name__ == '__main__':   
	 main()
# -----------------------------