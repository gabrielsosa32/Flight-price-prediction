# ✈️ Predicción de precios de vuelos

> *Regression model that predicts domestic flight ticket prices in India (Random Forest, R² 0.9745 on the test set).*

Modelo de machine learning que estima el precio de un pasaje aéreo a partir de las características del vuelo (clase, duración, escalas, aerolínea, origen y destino). El objetivo planteado es ayudar a una agencia de viajes a estimar precios de forma consistente.

Todo el desarrollo está en [`PruebaFinal.ipynb`](PruebaFinal.ipynb).

---

## 📌 Dataset

Dataset público de Kaggle (*Flight Price Prediction*) con vuelos domésticos de India entre el 11/02/2022 y el 31/03/2022. Viene en dos archivos con las mismas columnas, que se concatenan:

- `business.xlsx` — pasajes en clase Business
- `economy.xlsx` — pasajes en clase Economy

En total: **300.261 filas**, sin valores faltantes. La variable objetivo es `price`, en **rupias indias (INR)**.

---

## ⚙️ Pipeline

1. **Limpieza**: el precio se convierte a número; la duración (`02h 15m`) se pasa a minutos; la columna de escalas se limpia de saltos de línea y espacios.
2. **EDA**: distribuciones de precio, aerolíneas, clases y escalas; boxplots de precio por clase y por cantidad de escalas.
3. **Outliers (IQR)**: se detectaron 2.004 en duración y 123 en precio. **Se conservaron** porque son valores reales y con sentido para el negocio.
4. **Encoding**: Label Encoding para `stop` y `class`; One-Hot para `from`, `to`, `airline` y `ch_code`.
5. **Selección de variables (ANOVA, `f_regression`)**: `class` es por lejos la variable más relevante (F ≈ 2,19 millones), seguida de `num_code`, `time_taken` y `stop`.
6. **Escalado** con `StandardScaler` y split 70/30 (`random_state=42`).
7. **Modelado**: 6 modelos con `GridSearchCV` (cv=5, métrica R²).

---

## 📊 Resultados (conjunto de test, 30%)

| Modelo | Mejores hiperparámetros | R² | MAE (INR) | RMSE (INR) |
|---|---|---|---|---|
| Linear Regression | — | 0,8913 | 5.086 | 7.480 |
| Ridge | `alpha=1.0` | 0,8913 | 5.086 | 7.480 |
| Lasso | `alpha=0.001` | 0,8913 | 5.086 | 7.480 |
| ElasticNet | `alpha=0.01, l1_ratio=0.8` | 0,8913 | 5.090 | 7.480 |
| Árbol de Decisión | `max_depth=None, min_samples_split=5` | 0,9741 | 2.229 | 3.654 |
| **Random Forest** | `n_estimators=500, max_depth=None` | **0,9745** | **2.230** | **3.623** |

**Random Forest** es el modelo elegido: explica más del 97% de la varianza del precio, con un error absoluto medio de unas 2.230 rupias. El árbol de decisión queda prácticamente empatado; los modelos lineales se estancan en R² ≈ 0,89.

### Limitaciones

- El modelo usa las variables numéricas del dataset (`class`, `num_code`, `time_taken`, `stop`); las columnas one-hot (origen, destino, aerolínea) quedan fuera porque pandas las genera como booleanas y no entran en `select_dtypes('number')`. Sumarlas es el siguiente paso natural.
- El modelo entrenado (`modelo_precio_vuelos.pkl`, ~1 GB) **no está en el repo** por su tamaño: se regenera al correr el notebook.

---

## 🚀 Cómo correrlo

```bash
git clone https://github.com/gabrielsosa32/Flight-price-prediction.git
cd Flight-price-prediction
pip install -r requirements.txt
jupyter notebook PruebaFinal.ipynb
```

> El GridSearch del Random Forest es la parte más lenta (varios minutos con todos los núcleos).

---

## 🔧 Tecnologías

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter
