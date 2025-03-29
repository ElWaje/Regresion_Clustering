# Proyecto IA y Big Data: Predicción y Segmentación de Clientes


### 📚 **Máster FP Especialización IA y Big Data - Curso 2024/25**

**Grupo 1:** Ana Ortiz, Alfredo Martínez, Jorge Rodríguez y Enrique Solís  
**Tutor:** Nil Redón Orriols

---

## 📖 **Introducción**

Este proyecto utiliza técnicas avanzadas de Inteligencia Artificial (IA) y Big Data aplicadas sobre un dataset real de transacciones comerciales, proporcionado por Kaggle (E-Commerce Data). El dataset contiene transacciones realizadas entre diciembre de 2010 y diciembre de 2011, con más de 500,000 registros.

---

## 🎯 **Objetivo del Proyecto**

El objetivo principal del proyecto es realizar un análisis exhaustivo mediante dos enfoques principales:

- **Modelos de Regresión**: Para predecir ventas futuras.
- **Modelos de Clustering**: Para segmentar clientes según patrones de comportamiento y optimizar estrategias de marketing.

---

## 🗃️ **Dataset**

- **InvoiceNo**: Número de factura
- **StockCode**: Código de producto
- **Description**: Descripción del producto
- **Quantity**: Cantidad vendida
- **InvoiceDate**: Fecha y hora
- **UnitPrice**: Precio unitario
- **CustomerID**: ID del cliente
- **Country**: País del cliente

---

## 🛠️ **Metodología**

El proyecto se desarrolló siguiendo estos pasos:

### **1. Análisis Exploratorio de Datos (EDA)**

- Visualización inicial y estadística descriptiva
- Identificación y tratamiento de outliers
- Evaluación de correlaciones y tendencias temporales

### **2. Limpieza y Transformación de Datos**

- Manejo de valores nulos y negativos
- Creación de variables adicionales (TotalVentas, Recency, etc.)
- Escalado y normalización

### **3. Modelado Predictivo (Regresión)**

Modelos utilizados:
- ARIMA
- Regresión Lineal
- Random Forest
- Prophet
- LSTM

### **4. Modelado de Segmentación (Clustering)**

Modelos utilizados:
- K-Means
- DBSCAN
- Clustering Aglomerativo
- Gaussian Mixture Model (GMM)

---

## 📊 **Resultados Clave**

### **Regresión**
- Mejor modelo (datos escalados): **LSTM** con RMSE: 38.60
- Mejor modelo (datos sin escalar): **ARIMA** con MAE: 24,632.72

### **Clustering**
- Número óptimo de clusters: **4**
- Silhouette Score: **0.513**
- Índice Davies-Bouldin: **0.792**
- Mejor modelo seleccionado: **K-Means**

---

## 📈 **Dashboard Interactivo**

Se desarrolló un dashboard interactivo en Dash y Plotly que permite:
- Filtrado interactivo por país y cliente
- Visualización dinámica de predicciones y clusters
- Exploración gráfica avanzada

---

## 🚀 **Conclusiones**

- La combinación de regresión y clustering proporciona insights valiosos para estrategias comerciales.
- El escalado de datos mejora significativamente el rendimiento predictivo.
- K-Means resultó ser eficiente y práctico para segmentar clientes.

---

## 🔮 **Perspectivas Futuras**

- Automatización avanzada del pipeline de modelado.
- Integración de los modelos en entornos operacionales en tiempo real.
- Aplicación de técnicas avanzadas como redes neuronales profundas.

---

## ⚙️ **Estructura del Repositorio**

```
.
├── menu.py
├── codigo.py
├── cluster.py
├── Informe de Análisis de Datos EDA  
├── Informe de Pandas Profiling  
├── Informe de predicción de ventas (No Escalado)  
├── Informe de predicción de ventas (Escalado)  
├── Informe de Comparación de Modelos (Escalado vs No Escalado)  
├── Informe Clustering  
├── Informe comparación Clustering
├── Informe Clustering Kmeans  
├── Informe Clustering Dbscan  
├── Informe Clustering Agglomerative  
├── Informe Clustering Gmm  
├── sample_data/data,csv
├── assets/styles.css
├── df_limpio,csv
├── df_clientes.csv
├── clusters_asignados.csv
├── README.md
└── requirements.txt
```

---

## 📍 **Requisitos Técnicos**

Python 3.x con librerías principales:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Dash
- Prophet
- TensorFlow/Keras (para LSTM)

Instalación rápida:
```
pip install -r requirements.txt
```

---

## 📄 **Documentación y Reportes**


## 🔗 **Hipervínculos**

### **Código fuente del proyecto:**

- [menu.py](https://github.com/ElWaje/Regresion_Clustering/tree/main/menu.py)
- [codigo.py](https://github.com/ElWaje/Regresion_Clustering/tree/main/codigo.py)
- [cluster.py](https://github.com/ElWaje/Regresion_Clustering/tree/main/cluster.py)
- [assets/styles.css](https://github.com/ElWaje/Regresion_Clustering/tree/main/assets/styles.css)
- [sample_data/data.csv](https://github.com/ElWaje/Regresion_Clustering/tree/main/sample_data/data.csv)

### **Enlaces a los Informes de Regresión:**

- [Informe de Análisis de Datos EDA](https://github.com/ElWaje/Regresion_Clustering/tree/main/reports/informe_eda.html)
- [Informe de Pandas Profiling](https://github.com/ElWaje/Regresion_Clustering/tree/main/reports/informe_profiling.html)
- [Informe de predicción de ventas (No Escalado)](https://github.com/ElWaje/Regresion_Clustering/tree/main/reports/informe_prediccion_no_escalado.html)
- [Informe de predicción de ventas (Escalado)](https://github.com/ElWaje/Regresion_Clustering/tree/main/reports/informe_prediccion_escalado.html)
- [Informe de Comparación de Modelos (Escalado vs No Escalado)](https://github.com/ElWaje/Regresion_Clustering/tree/main/reports/comparacion_modelos.html)

### **Enlaces a los Informes de Clustering:**

- [Informe Clustering](https://github.com/ElWaje/Regresion_Clustering/tree/main/reports/informe_clustering.html)
- [Informe comparación Clustering](https://github.com/ElWaje/Regresion_Clustering/tree/main/reports/comparacion_clustering.html)
- [Informe Clustering Kmeans](https://github.com/ElWaje/Regresion_Clustering/tree/main/reports/informe_kmeans.html)
- [Informe Clustering Dbscan](https://github.com/ElWaje/Regresion_Clustering/tree/main/reports/informe_dbscan.html)
- [Informe Clustering Agglomerative](https://github.com/ElWaje/Regresion_Clustering/tree/main/reports/informe_agglomerative.html)
- [Informe Clustering Gmm](https://github.com/ElWaje/Regresion_Clustering/tree/main/reports/informe_gmm.html)


---

## 🙌 **Grupo 1**

- **Ana Ortiz**
- **Alfredo Martínez**
- **Jorge Rodríguez**
- **Enrique Solís**

---

## 📝 **Licencia**

Este proyecto está bajo licencia MIT.

---


✨ ¡Gracias por visitar nuestro proyecto!

