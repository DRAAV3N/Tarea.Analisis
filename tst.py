
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------
# Métricas implementadas a mano
# -------------------
def calcular_rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))

def calcular_r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot

def calcular_durbin_watson(residuals):
    num = np.sum(np.diff(residuals) ** 2)
    den = np.sum(residuals ** 2)
    return num / den

# -------------------
# Cargar datos
# -------------------
df_train = pd.read_csv("dtrn.csv")
df_test = pd.read_csv("dtst.csv")
selected_vars = pd.read_csv("selected_vars.csv", header=None)[0].tolist()

X_train = df_train[selected_vars].values
y_train = df_train["Y"].values
X_test = df_test[selected_vars].values
y_test = df_test["Y"].values

# -------------------
# Estimar coeficientes
# -------------------
X_train_with_1 = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
beta = np.linalg.pinv(X_train_with_1) @ y_train

# -------------------
# Predicción
# -------------------
X_test_with_1 = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
y_pred = X_test_with_1 @ beta
residuals = y_test - y_pred

# -------------------
# Métricas
# -------------------
rmse = calcular_rmse(y_test, y_pred)
r2 = calcular_r2(y_test, y_pred)
dw = calcular_durbin_watson(residuals)

print("=" * 40)
print("VALIDACION EN ETAPA TESTING")
print(f"RMSE           : {rmse:.4f}")
print(f"R^2            : {r2 * 100:.2f}%")
print(f"Durbin-Watson  : {dw:.4f}")
print("=" * 40)

# -------------------
# Gráficos
# -------------------
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Real Values")
plt.plot(y_pred, label="Estimated Values")
plt.xlabel("#Samples")
plt.ylabel("Values")
plt.title("Real versus Estimated")
plt.legend()
plt.tight_layout()
plt.savefig("figure3.png")
plt.close()

# -------------------
# Gráfico de linealidad: Y_hat vs residuos
# -------------------
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color="blue", alpha=0.7)
plt.axhline(y=0, color="red", linewidth=2)
plt.title("Residuals vs Predicted Values")
plt.xlabel(r"$\hat{Y}$ (Predicted Values)")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("figure4.png")
plt.close()