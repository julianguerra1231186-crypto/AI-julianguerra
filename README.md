# 🧠 Modelo de Predicción de Salud Mental  
Sistema de Machine Learning para predecir si una persona podría requerir tratamiento de salud mental, usando múltiples algoritmos de clasificación y un flujo completo de análisis de datos (EDA → Preprocesamiento → Modelos → Métricas → Visualizaciones).

---

## 🚀 Abrir el Proyecto en Google Colab  
Haz clic aquí para ver y ejecutar el notebook en línea:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/julianguerra1231186-crypto/AI-julianguerra/blob/main/ModeloDePrediccionSaludMental.ipynb)

---

## 📌 ¿Qué hace este proyecto?

🔹 Limpia, codifica y balancea un dataset de salud mental  
🔹 Realiza un EDA completo con visualizaciones  
🔹 Entrena 3 modelos de clasificación (Naive Bayes, Decision Tree, SVM)  
🔹 Optimiza modelos con GridSearchCV  
🔹 Compara rendimiento con múltiples métricas  
🔹 Genera gráficas obligatorias para el taller  

---

# 🧩 Cómo solucionamos el taller — Explicación Paso a Paso

## 1️⃣ Preparación del entorno  
- Importamos librerías principales: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.  
- Cargamos el dataset `Mentaldataset1.csv`.  
- Revisamos estructura con `head()`, `info()`, `describe()`.

---

## 2️⃣ Limpieza del dataset  
- Identificamos valores faltantes mediante `isnull().sum()`.  
- Eliminamos filas incompletas con `dropna()` para garantizar consistencia.  

---

## 3️⃣ Codificación de variables categóricas  
- Usamos `LabelEncoder` para todas las columnas categóricas (género, país, ocupación, historial familiar, etc.).  
- Esto permite que los modelos trabajen con datos numéricos.

---

## 4️⃣ EDA — Análisis Exploratorio de Datos  
Realizamos un análisis visual y estadístico:

📊 Histogramas → distribución general  
🔥 Heatmaps → correlaciones Pearson, Spearman y Kendall  
📉 Detección del problema principal: **desbalance de la variable objetivo** (`Treatment`)

---

## 5️⃣ Balanceo del dataset  
- Aplicamos **oversampling** con `resample` para equilibrar ambas clases.  
- Esto previene que los modelos se sesguen hacia la clase mayoritaria.

---

## 6️⃣ División del dataset  
- Definimos `X` (features) e `y` (target: `Treatment`).  
- Dividimos en 80% entrenamiento y 20% prueba con `train_test_split`.

---

## 7️⃣ Entrenamiento de modelos  
Entrenamos y evaluamos:

### 🧪 **1. Naive Bayes (GaussianNB)**
- Entrenamiento base  
- Métricas: precisión, recall, F1-score  
- Optimización con `GridSearchCV`

### 🌳 **2. Decision Tree**
- Entrenamiento base  
- Importancia de variables (`feature_importances_`)  
- Optimización con grid search  

### ⚙️ **3. SVM (Support Vector Machine)**  
- Datos escalados con `StandardScaler`  
- Métricas completas  

---

## 8️⃣ Visualizaciones obligatorias  
Incluimos:

📌 **Heatmap de correlaciones**  
📌 **Gráfico antes vs después del balanceo**  
📌 (Opcionales) Importancia de variables y matrices de confusión  

---

## 9️⃣ Reproducibilidad  
- Se estableció `random_state` en todos los procesos aleatorios.  
- Se creó `requirements.txt` y `.gitignore` para mantener ordenado el proyecto.  

---

# ▶️ Cómo ejecutar el proyecto localmente

```bash
pip install -r requirements.txt
python modelodeprediccionsaludmental.py
