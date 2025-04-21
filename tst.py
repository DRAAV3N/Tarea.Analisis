import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#---------------------- FUNCIONES AUXILIARES ------------------------
def load_data_csv(nfile):
    return pd.read_csv(nfile)

def predict(X, beta):
    return X @ beta

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat)**2))

def r2_score(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot

def durbin_watson(y, yhat):
    e = y - yhat
    num = np.sum((e[1:] - e[:-1])**2)
    den = np.sum(e**2)
    return num / den

def plot_real_vs_pred(y, yhat):
    plt.figure()
    plt.plot(y, label='Real Values', color='blue')
    plt.plot(yhat, label='Estimated Values', color='orange')
    plt.title('Real versus Estimados')
    plt.xlabel('Nro. Muestras')
    plt.ylabel('Valores')
    plt.legend()
    plt.grid(True)
    plt.savefig('figure3.png')
    plt.close()

def plot_residuals(y, yhat, dw):
    residuals = y - yhat
    plt.figure()
    plt.scatter(yhat, residuals, color='blue', s=10)
    plt.axhline(0, color='red', linewidth=2)
    plt.title('Linealidad')
    plt.xlabel('Estimated-Y values')
    plt.ylabel('Residual')
    plt.grid(True)
    plt.text(0.05, -3.5, f'Durbin-Watson: {dw:.3f}', fontsize=12, color='blue')
    plt.savefig('figure4.png')
    plt.close()

#---------------------------- MAIN ----------------------------------
def main():
    # Cargar dataset completo como en trn.py
    df = load_data_csv('dataset.csv')

    # División train/test
    p_train = 0.80
    cut = int(len(df) * p_train)
    df_test = df[cut:]

    # Cargar coeficientes y nombres de columnas seleccionadas
    beta = load_data_csv('coefts.csv').iloc[:, 1].values  # columna 'Coef'
    selected_vars = pd.read_csv("selected_vars.csv", header=None).values.flatten()

    # Separar variable dependiente
    y_real = df_test.iloc[:, 0].values

    # Filtrar solo las columnas seleccionadas
    X_test = df_test[selected_vars].values

    # Predicción
    y_pred = predict(X_test, beta)

    # Métricas
    val_rmse = rmse(y_real, y_pred)
    val_r2 = r2_score(y_real, y_pred)
    val_dw = durbin_watson(y_real, y_pred)

    # Guardar métricas
    df_metrics = pd.DataFrame({
        "RMSE": [val_rmse],
        "R2": [val_r2],
        "Durbin-Watson": [val_dw]
    })
    df_metrics.to_csv("metrica.csv", index=False)

    # Guardar real vs predicho
    df_real_pred = pd.DataFrame({
        "Real": y_real,
        "Predicho": y_pred
    })
    df_real_pred.to_csv("real_pred.csv", index=False)

    # Graficar
    plot_real_vs_pred(y_real, y_pred)
    plot_residuals(y_real, y_pred, val_dw)

#---------------------------- RUN -----------------------------------
if __name__ == '__main__':
    main()
