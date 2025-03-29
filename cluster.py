import subprocess  
import sys
import os
import time
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Generar gráficos sin ventana interactiva
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO, StringIO
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from sklearn.decomposition import PCA
from dash import Dash, dcc, html, Input, Output
import threading
from ydata_profiling import ProfileReport
import warnings
warnings.simplefilter("ignore", category=UserWarning)

# Importaciones para gráficos adicionales
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage
from pandas.plotting import parallel_coordinates

# ----------------------------------------------
# Instalación de librerías faltantes
# ----------------------------------------------
required_libs = [
    "pandas", "numpy", "matplotlib", "seaborn", "sklearn", 
    "dash", "kaleido", "ydata_profiling", "wordcloud", "scipy"
]

def install_missing_libs():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
    except subprocess.CalledProcessError:
        logging.error("pip no está instalado. Instálalo antes de continuar.")
        return
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            logging.info(f"Instalando {lib}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            except subprocess.CalledProcessError as e:
                logging.error(f"Error al instalar {lib}: {e}")

install_missing_libs()

# Declaramos la variable global para el DataFrame de clientes
df_clientes_global = None

# ----------------------------------------------
# Clase para cargar y explorar el dataset
# ----------------------------------------------
class CargadorDatos:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def cargar_datos(self):
        if not os.path.exists(self.file_path):
            logging.error(f"El archivo '{self.file_path}' no existe.")
            return None
        if not self.file_path.lower().endswith(".csv"):
            logging.warning(f"'{self.file_path}' no es un archivo CSV.")
            return None
        try:
            self.df = pd.read_csv(self.file_path, encoding="ISO-8859-1", low_memory=False)
            logging.info(f"Datos cargados correctamente desde '{self.file_path}' ({len(self.df)} registros).")
            return self.df
        except Exception as e:
            logging.error(f"Error al cargar el archivo CSV: {e}")
            return None

    def explorar_datos(self):
        if self.df is None:
            logging.warning("No hay datos cargados para explorar.")
            return None, None, None
        info_buf = StringIO()
        self.df.info(buf=info_buf)
        info = info_buf.getvalue()
        return info, self.df.head(), self.df.describe()

# ----------------------------------------------
# Clase para limpiar y transformar datos
# ----------------------------------------------
class ProcesadorDatos:
    def __init__(self, df, df_original):
        self.df = df.copy()
        self.df_original = df_original.copy()
        self.stats_limpieza = {}

    def limpiar_datos(self, mantener_devoluciones=True, imputar_nulos=True):
        start_time = time.perf_counter()
        self.stats_limpieza["Registros originales"] = len(self.df)
        self.stats_limpieza["Valores nulos antes"] = self.df.isna().sum().sum()
        self.df.dropna(subset=["CustomerID"], inplace=True)
        if mantener_devoluciones:
            logging.info("Se mantienen transacciones con valores negativos en Quantity (devoluciones).")
        else:
            self.df = self.df[self.df["Quantity"] > 0]
        if imputar_nulos:
            logging.info("Se imputan valores faltantes en lugar de eliminarlos.")
            for col in ["CustomerID", "Quantity", "UnitPrice"]:
                if col in self.df.columns and self.df[col].isna().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
        else:
            self.df.dropna(subset=["CustomerID"], inplace=True)
        if "Quantity" in self.df.columns and "UnitPrice" in self.df.columns:
            self.df["TotalGastado"] = self.df["Quantity"] * self.df["UnitPrice"]
            self.df["TotalGastado"] = self.df["TotalGastado"].fillna(0)
        else:
            logging.error("No se encontraron las columnas necesarias para calcular TotalGastado.")
            return None
        if "InvoiceDate" in self.df.columns:
            try:
                self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"], errors="coerce")
                logging.info("'InvoiceDate' convertido correctamente a datetime.")
            except Exception as e:
                logging.error(f"Error al convertir 'InvoiceDate': {e}")
        else:
            logging.error("No se encontró la columna 'InvoiceDate'.")
        self.df["Year"] = self.df["InvoiceDate"].dt.year
        self.df["Month"] = self.df["InvoiceDate"].dt.month
        self.df["Quarter"] = self.df["InvoiceDate"].dt.quarter
        self.df["DayOfWeek"] = self.df["InvoiceDate"].dt.dayofweek
        self.df["Hour"] = self.df["InvoiceDate"].dt.hour
        self.stats_limpieza["Registros después de limpieza"] = len(self.df)
        self.stats_limpieza["Valores nulos después"] = self.df.isna().sum().sum()
        elapsed = time.perf_counter() - start_time
        logging.info(f"Limpieza completada: {len(self.df)} registros en {elapsed:.2f} seg.")
        return self.df

    def escalar_datos(self, metodo="StandardScaler", features=None):
        start_time = time.perf_counter()
        if features is None:
            features = ["Quantity", "UnitPrice", "TotalGastado"]
        escaladores = {"StandardScaler": StandardScaler(), "RobustScaler": RobustScaler(), "MinMaxScaler": MinMaxScaler()}
        if metodo not in escaladores:
            logging.warning(f"Método {metodo} no válido. Usando StandardScaler.")
            metodo = "StandardScaler"
        self.scaler = escaladores[metodo]
        self.df[features] = self.scaler.fit_transform(self.df[features])
        elapsed = time.perf_counter() - start_time
        logging.info(f"Datos escalados con {metodo} en {elapsed:.2f} seg.")
        return self.df

    def generar_tabla_limpieza(self):
        return pd.DataFrame.from_dict(self.stats_limpieza, orient='index', columns=['Valor'])

# ----------------------------------------------
# Transformación a datos de clientes
# ----------------------------------------------
def transformar_a_dataset_clientes(df):
    if "StockCode" not in df.columns:
        logging.warning("La columna 'StockCode' no se encontró; se usará 'Description' para contar productos únicos.")
        producto_col = "Description"
    else:
        producto_col = "StockCode"
    reference_date = df["InvoiceDate"].max()
    df_clientes = df.groupby("CustomerID").agg(
        TotalCompras=("InvoiceDate", "count"),
        TotalGastado=("TotalGastado", "sum"),
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        NumeroProductosDistintos=(producto_col, "nunique")
    ).reset_index()
    df_clientes["TicketPromedio"] = df_clientes["TotalGastado"] / df_clientes["TotalCompras"]
    return df_clientes

# ----------------------------------------------
# Entrenadores para Clustering
# ----------------------------------------------
class EntrenadorClustering_KMeans:
    def __init__(self, df):
        self.df = df.copy()
        self.model = None
        self.labels = None
        self.silhouette = None
        self.db_index = None
        self.ch_score = None

    def entrenar(self, features, n_clusters=3, sample_size=10000):
        X = self.df[features]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels = kmeans.fit_predict(X)
        self.model = kmeans
        inertia = self.model.inertia_
        logging.info(f"[K-Means] Inercia: {inertia:.2f}")
        indices = []
        for cl in np.unique(self.labels):
            idx = np.where(self.labels == cl)[0]
            n_sample = max(1, int(len(idx) / len(X) * sample_size))
            n_sample = min(n_sample, len(idx))
            indices.extend(np.random.choice(idx, n_sample, replace=False))
        X_sample = X.iloc[indices]
        labels_sample = self.labels[indices]
        if len(np.unique(labels_sample)) < 2:
            logging.warning("[K-Means] La muestra estratificada tiene un solo cluster. Saltando métricas.")
            self.silhouette = None
            self.ch_score = None
            self.db_index = None
        else:
            self.silhouette = silhouette_score(X_sample, labels_sample)
            self.ch_score = calinski_harabasz_score(X_sample, labels_sample)
            self.db_index = davies_bouldin_score(X_sample, labels_sample)
        logging.info(f"[K-Means] Usando muestra estratificada de {len(X_sample)} registros para métricas.")
        return self.labels

    def graficar(self, features):
        X = self.df[features]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=self.labels, palette="viridis", s=80, legend="full")
        centroids = self.model.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroides')
        plt.title("Clusters obtenidos con K-Means")
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.legend()
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()
        return f'<img src="data:image/png;base64,{img_b64}" style="width:100%;">'

class EntrenadorClustering_DBSCAN:
    def __init__(self, df):
        self.df = df.copy()
        self.model = None
        self.labels = None
        self.silhouette = None
        self.db_index = None
        self.ch_score = None

    def entrenar(self, features, eps=0.7, min_samples=10, sample_size=10000):
        X = self.df[features]
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        if X.shape[0] > 10000:
            X = X.sample(n=10000, random_state=42)
            logging.warning("DBSCAN: Se usa una muestra de 10,000 registros para evitar problemas de memoria.")
        self.labels = dbscan.fit_predict(X)
        self.model = dbscan
        if len(np.unique(self.labels)) > 1:
            self.silhouette = silhouette_score(X, self.labels)
            self.ch_score = calinski_harabasz_score(X, self.labels)
            self.db_index = davies_bouldin_score(X, self.labels)
        else:
            self.silhouette = None
            self.ch_score = None
            self.db_index = None
        self.df = X.copy()
        return self.labels

    def graficar(self, features):
        X = self.df[features]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=self.labels, palette="viridis", s=80, legend="full")
        plt.title("Clusters obtenidos con DBSCAN")
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()
        return f'<img src="data:image/png;base64,{img_b64}" style="width:100%;">'

class EntrenadorClustering_Agglomerative:
    def __init__(self, df):
        self.df = df.copy()
        self.model = None
        self.labels = None
        self.silhouette = None
        self.db_index = None
        self.ch_score = None

    def entrenar(self, features, n_clusters=3, sample_size=10000):
        X = self.df[features]
        if X.shape[0] > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
            logging.warning("Agglomerative: Se usa una muestra de {} registros.".format(sample_size))
        else:
            X_sample = X.copy()
        agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        self.labels = agglo.fit_predict(X_sample)
        self.model = agglo
        self.silhouette = silhouette_score(X_sample, self.labels)
        self.ch_score = calinski_harabasz_score(X_sample, self.labels)
        self.db_index = davies_bouldin_score(X_sample, self.labels)
        self.df = X_sample.copy()
        return self.labels

    def graficar(self, features):
        X = self.df[features]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=self.labels, palette="viridis", s=80, legend="full")
        plt.title("Clusters obtenidos con Agglomerative Clustering")
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()
        return f'<img src="data:image/png;base64,{img_b64}" style="width:100%;">'

class EntrenadorClustering_GMM:
    def __init__(self, df):
        self.df = df.copy()
        self.model = None
        self.labels = None
        self.silhouette = None
        self.db_index = None
        self.ch_score = None

    def entrenar(self, features, n_components=3, sample_size=10000):
        X = self.df[features]
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        self.labels = gmm.predict(X)
        self.model = gmm
        if X.shape[0] > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
            labels_sample = gmm.predict(X_sample)
            if len(np.unique(labels_sample)) < 2:
                logging.warning("[GMM] La muestra tiene un solo cluster. Saltando métricas.")
                self.silhouette = None
                self.ch_score = None
                self.db_index = None
            else:
                self.silhouette = silhouette_score(X_sample, labels_sample)
                self.ch_score = calinski_harabasz_score(X_sample, labels_sample)
                self.db_index = davies_bouldin_score(X_sample, labels_sample)
            logging.info(f"[GMM] Usando muestra de {sample_size} registros para métricas.")
        else:
            self.silhouette = silhouette_score(X, self.labels)
            self.ch_score = calinski_harabasz_score(X, self.labels)
            self.db_index = davies_bouldin_score(X, self.labels)
        self.df = X_sample.copy() if X.shape[0] > sample_size else X.copy()
        return self.labels

    def graficar(self, features):
        X = self.df[features]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=self.labels, palette="viridis", s=80, legend="full")
        plt.title("Clusters obtenidos con GMM")
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()
        return f'<img src="data:image/png;base64,{img_b64}" style="width:100%;">'

# ----------------------------------------------
# Funciones para gráficos adicionales de clusters
# ----------------------------------------------
def graficar_silhouette(X, labels):
    n_clusters = len(np.unique(labels))
    silhouette_vals = silhouette_samples(X, labels)
    y_lower = 10
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in np.unique(labels):
        ith_silhouette_vals = silhouette_vals[labels == i]
        ith_silhouette_vals.sort()
        size_cluster_i = ith_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_silhouette_vals,
                        facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    avg_silhouette = silhouette_score(X, labels)
    ax.axvline(x=avg_silhouette, color="red", linestyle="--")
    ax.set_title("Gráfico de Silhouette")
    ax.set_xlabel("Valor de Silhouette")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    buffer = BytesIO() 
    try:
        plt.savefig(buffer, format="png", bbox_inches="tight")
    except Exception as e:
        if "Done" in str(e):
            # La excepción "Done" indica que la renderización finalizó; se ignora.
            pass
        else:
            raise
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return f'<img src="data:image/png;base64,{img_b64}" style="width:100%;">'

def graficar_pairplot(df, hue="Cluster"):
    # Si el dataframe es muy grande, tomar una muestra (por ejemplo, 1000 registros)
    if df.shape[0] > 1000:
        df = df.sample(n=1000, random_state=42)
    pair = sns.pairplot(df, hue=hue, diag_kind="kde")
    pair.fig.suptitle("Pairplot de Variables", y=1.02)
    buffer = BytesIO()
    pair.fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close('all')
    return f'<img src="data:image/png;base64,{img_b64}" style="width:100%;">'

def graficar_dendrograma(X):
    # Si X es muy grande, muestrear (por ejemplo, 500 registros)
    if X.shape[0] > 500:
        X = X.sample(n=500, random_state=42)
    Z = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, ax=ax)
    ax.set_title("Dendrograma (Clustering Jerárquico)")
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return f'<img src="data:image/png;base64,{img_b64}" style="width:100%;">'

def graficar_parallel_coordinates(df, class_column="Cluster"):
    fig, ax = plt.subplots(figsize=(10, 6))
    parallel_coordinates(df, class_column, colormap=plt.get_cmap("Set2"))
    ax.set_title("Parallel Coordinates Plot")
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return f'<img src="data:image/png;base64,{img_b64}" style="width:100%;">'

# ----------------------------------------------
# Funciones para gráficos de análisis de variables
# ----------------------------------------------
def graficar_histogramas_variables(df, variables):
    imgs = ""
    for var in variables:
        plt.figure(figsize=(6,4))
        sns.histplot(df[var], kde=True, bins=30)
        plt.title(f"Histograma de {var}")
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        imgs += f'<img src="data:image/png;base64,{img_b64}" style="width:100%;"><br>'
        plt.close()
    return imgs

def graficar_boxplots_variables(df, variables):
    imgs = ""
    for var in variables:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[var])
        plt.title(f"Boxplot de {var}")
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        imgs += f'<img src="data:image/png;base64,{img_b64}" style="width:100%;"><br>'
        plt.close()
    return imgs

def graficar_heatmap_variables(df, variables):
    corr = df[variables].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap de Correlación")
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return f'<img src="data:image/png;base64,{img_b64}" style="width:100%;">'

def graficar_density_plots_variables(df, variables):
    imgs = ""
    for var in variables:
        plt.figure(figsize=(6,4))
        sns.kdeplot(df[var], fill=True)
        plt.title(f"Density Plot de {var}")
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        imgs += f'<img src="data:image/png;base64,{img_b64}" style="width:100%;"><br>'
        plt.close()
    return imgs

def graficar_violin_plots_variables(df, variables):
    """Genera un violin plot para cada variable en la lista 'variables'."""
    imgs = ""
    for var in variables:
        plt.figure(figsize=(6,4))
        sns.violinplot(y=df[var], color="skyblue")
        plt.title(f"Violin Plot de {var}")
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        imgs += f'<img src="data:image/png;base64,{img_b64}" style="width:100%;"><br>'
        plt.close()
    return imgs

def graficar_swarm_plots_variables(df, variables):
    """Genera un swarm plot para cada variable en la lista 'variables' con muestreo y manejo de la excepción 'Done'."""
    imgs = ""
    for var in variables:
        plt.figure(figsize=(6, 4))
        # Extraer la serie de datos y eliminar nulos
        data = df[var].dropna()
        # Si hay demasiados datos, tomar una muestra aleatoria (por ejemplo, 1000 puntos)
        if len(data) > 1000:
            data_sample = data.sample(1000, random_state=42)
        else:
            data_sample = data
        try:
            sns.swarmplot(x=data_sample, color="coral")
        except Exception as e:
            # Si la excepción es la interna "Done", la ignoramos
            if e.__class__.__name__ == "Done":
                pass
            else:
                raise e
        plt.title(f"Swarm Plot de {var}")
        buffer = BytesIO()
        try:
            plt.savefig(buffer, format="png", bbox_inches="tight")
        except Exception as e:
            if e.__class__.__name__ == "Done":
                pass
            else:
                raise e
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        imgs += f'<img src="data:image/png;base64,{img_b64}" style="width:100%;"><br>'
        plt.close()
    return imgs


def graficar_cdf_plots_variables(df, variables):
    """Genera un gráfico CDF (función de distribución acumulada) para cada variable."""
    imgs = ""
    for var in variables:
        data = np.sort(df[var].dropna().values)
        cdf = np.arange(1, len(data)+1) / float(len(data))
        plt.figure(figsize=(6,4))
        plt.plot(data, cdf, marker=".", linestyle="none")
        plt.xlabel(var)
        plt.ylabel("CDF")
        plt.title(f"CDF de {var}")
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        imgs += f'<img src="data:image/png;base64,{img_b64}" style="width:100%;"><br>'
        plt.close()
    return imgs


class GeneradorInformeClustering:
    def __init__(self, df_original, df, info_text, head, describe, stats_limpieza,
                entrenador, features, n_clusters):
        self.df_original = df_original
        self.df = df
        self.info_text = info_text
        self.head = head
        self.describe = describe
        self.stats_limpieza = stats_limpieza
        self.entrenador = entrenador
        self.features = features
        self.n_clusters = n_clusters

    def generar_informe(self, output_file):
        # Determinar las variables para las estadísticas
        if set(self.features).issubset(set(self.df.columns)):
            features_for_stats = self.features
        else:
            features_for_stats = list(self.df.columns)
            
        # Gráficos básicos del clustering
        grafico_clusters = self.entrenador.graficar(features_for_stats)
        try:
            X = self.df[features_for_stats]
            inertias = []
            ks = range(1, 11)
            if hasattr(self.entrenador.model, "inertia_"):
                for k in ks:
                    km = KMeans(n_clusters=k, random_state=42).fit(X)
                    inertias.append(km.inertia_)
                plt.figure(figsize=(10,6))
                plt.plot(ks, inertias, marker='o')
                plt.xlabel("Número de clusters")
                plt.ylabel("Inercia")
                plt.title("Método del Codo")
                buffer = BytesIO()
                plt.savefig(buffer, format="png", bbox_inches="tight")
                buffer.seek(0)
                elbow_img = base64.b64encode(buffer.read()).decode("utf-8")
                plt.close()
                grafico_elbow = f'<img src="data:image/png;base64,{elbow_img}" style="width:100%;">'
            else:
                grafico_elbow = "<p>No se aplica el método del codo para este algoritmo.</p>"
        except Exception as e:
            grafico_elbow = f"<p>Error al generar el gráfico del codo: {e}</p>"

        # Estadísticas por cluster
        try:
            X = self.df[features_for_stats]
            df_clusters = X.copy()
            df_clusters['Cluster'] = self.entrenador.labels
            stats_cluster = df_clusters.groupby('Cluster').agg(['mean', 'median', 'std'])
            stats_html = stats_cluster.to_html(classes='table table-striped', na_rep='-')
        except Exception as e:
            stats_html = f"<p>Error al generar estadísticas por cluster: {e}</p>"

        # Gráficos adicionales de clusters
        silhouette_img = ""
        pairplot_img = ""
        dendrograma_img = ""
        parallel_img = ""
        if self.entrenador.labels is not None and len(np.unique(self.entrenador.labels)) > 1:
            silhouette_img = graficar_silhouette(self.df, self.entrenador.labels)
        try:
            df_pair = self.df_original.copy()
            df_pair["Cluster"] = self.entrenador.labels
            pairplot_img = graficar_pairplot(df_pair, hue="Cluster")
        except Exception as e:
            pairplot_img = f"<p>Error al generar el pairplot: {e}</p>"
        try:
            dendrograma_img = graficar_dendrograma(self.df)
        except Exception as e:
            dendrograma_img = f"<p>Error al generar el dendrograma: {e}</p>"
        try:
            df_parallel = self.df_original.copy()
            df_parallel["Cluster"] = self.entrenador.labels.astype(str)
            parallel_img = graficar_parallel_coordinates(df_parallel, class_column="Cluster")
        except Exception as e:
            parallel_img = f"<p>Error al generar el Parallel Coordinates Plot: {e}</p>"

        # Gráficos de análisis de variables existentes
        variables = ["TotalCompras", "TotalGastado", "TicketPromedio", "Recency", "NumeroProductosDistintos"]
        hist_vars = graficar_histogramas_variables(self.df_original, variables)
        box_vars = graficar_boxplots_variables(self.df_original, variables)
        pair_vars = graficar_pairplot(self.df_original[variables], hue=None)
        heat_vars = graficar_heatmap_variables(self.df_original, variables)
        density_vars = graficar_density_plots_variables(self.df_original, variables)
        
        # Nuevos gráficos de análisis adicionales:
        violin_vars = graficar_violin_plots_variables(self.df_original, variables)
        swarm_vars = graficar_swarm_plots_variables(self.df_original, variables)
        cdf_vars = graficar_cdf_plots_variables(self.df_original, variables)

        # Construir el HTML del informe (se utiliza el CSS externo)
        html_content = f"""
        <html>
        <head>
            <title>Informe de Clustering</title>
            <meta charset="UTF-8">
            <link rel="stylesheet" type="text/css" href="assets/styles.css">
        </head>
        <body>
            <div class="container">
                <h1>Informe de Clustering</h1>
                <h2>Información del Dataset</h2>
                <div class="info-box">
                    <pre class="code-block">{self.info_text}</pre>
                    {self.head.to_html(classes='table table-striped', na_rep='-')}
                </div>
                <h2>Limpieza y Transformación</h2>
                <div class="info-box">
                    {self.describe.to_html(classes='table table-striped', na_rep='-')}
                    <p>Estadísticas de limpieza:</p>
                    {self.stats_limpieza.to_html(classes='table table-striped', na_rep='-')}
                </div>
                <h2>Clustering</h2>
                <div class="info-box">
                    <p><b>Número de clusters:</b> {self.n_clusters}</p>
                    <p><b>Silhouette Score:</b> {"N/A" if self.entrenador.silhouette is None else f"{self.entrenador.silhouette:.3f}"}</p>
                    <p><b>Calinski-Harabasz Score:</b> {"N/A" if self.entrenador.ch_score is None else f"{self.entrenador.ch_score:.3f}"}</p>
                    <p><b>Davies-Bouldin Index:</b> {"N/A" if self.entrenador.db_index is None else f"{self.entrenador.db_index:.3f}"}</p>
                    {"<p><b>Inercia:</b> " + f"{self.entrenador.model.inertia_:.2f}</p>" if hasattr(self.entrenador.model, "inertia_") else ""}
                </div>
                <h2>Método del Codo</h2>
                <div class="info-box">
                    {grafico_elbow}
                </div>
                <h2>Visualización de Clusters</h2>
                <div class="info-box">
                    {grafico_clusters}
                </div>
                <h2>Análisis Descriptivo por Cluster</h2>
                <div class="info-box">
                    <div class="table-scroll">
                        {stats_html}
                    </div>
                </div>
                <h2>Gráficos de Clusters Adicionales</h2>
                <div class="info-box">
                    <h3>Gráfico de Silhouette</h3>
                    {silhouette_img}
                    <h3>Pairplot de Clusters</h3>
                    {pairplot_img}
                    <h3>Dendrograma</h3>
                    {dendrograma_img}
                    <h3>Parallel Coordinates Plot</h3>
                    {parallel_img}
                </div>
                <h2>Análisis de Variables</h2>
                <div class="info-box">
                    <h3>Histogramas</h3>
                    {hist_vars}
                    <h3>Boxplots</h3>
                    {box_vars}
                    <h3>Pairplot de Variables</h3>
                    {pair_vars}
                    <h3>Heatmap de Correlación</h3>
                    {heat_vars}
                    <h3>Density Plots</h3>
                    {density_vars}
                </div>
                <h2>Análisis Adicional de Variables</h2>
                <div class="info-box">
                    <h3>Violin Plots</h3>
                    {violin_vars}
                    <h3>Swarm Plots</h3>
                    {swarm_vars}
                    <h3>CDF Plots</h3>
                    {cdf_vars}                    
                </div>
            </div>
        </body>
        </html>
        """
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"Informe de Clustering generado: {output_file}")

# ----------------------------------------------
# 4bis. Generación del Informe Comparativo de Clustering
# ----------------------------------------------
class GeneradorInformeComparacionClustering:
    def __init__(self, resultados, df_sample, labels_dict):
        """
        Inicializa la clase con las métricas de cada modelo, el dataframe de muestra y
        el diccionario de etiquetas asignadas por cada modelo.
        """
        self.resultados = resultados
        self.df_sample = df_sample
        self.labels_dict = labels_dict

    def calcular_mejor_modelo(self):
        """
        Calcula el mejor modelo basado en la métrica de Silhouette.
        """
        mejor_modelo = None
        mejor_score = -np.inf
        for modelo, metrics in self.resultados.items():
            score = metrics.get("silhouette")
            if score is not None and score > mejor_score:
                mejor_score = score
                mejor_modelo = modelo
        return mejor_modelo, mejor_score

    def generar_tabla_html(self):
        """
        Genera una tabla HTML con las métricas de evaluación para cada modelo.
        """
        mejor_modelo, mejor_score = self.calcular_mejor_modelo()
        html_table = """
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Modelo</th>
                    <th>Silhouette</th>
                    <th>Calinski-Harabasz</th>
                    <th>Davies-Bouldin</th>
                    <th>Inercia</th>
                </tr>
            </thead>
            <tbody>
        """
        for modelo, metrics in self.resultados.items():
            sil = metrics.get("silhouette")
            ch  = metrics.get("ch_score")
            db  = metrics.get("db_index")
            inertia = metrics.get("inertia", "N/A")
            sil_str = f"{sil:.3f}" if isinstance(sil, (int, float)) else "N/A"
            ch_str = f"{ch:.3f}" if isinstance(ch, (int, float)) else "N/A"
            db_str = f"{db:.3f}" if isinstance(db, (int, float)) else "N/A"
            inertia_str = f"{inertia:.2f}" if isinstance(inertia, (int, float)) else "N/A"
            html_table += f"""
                <tr>
                    <td>{modelo}</td>
                    <td>{sil_str}</td>
                    <td>{ch_str}</td>
                    <td>{db_str}</td>
                    <td>{inertia_str}</td>
                </tr>
            """
        if mejor_modelo is not None:
            html_table += f"""
                <tr class="highlight">
                    <td colspan="5"><b>Mejor Modelo:</b> {mejor_modelo} (Silhouette: {mejor_score:.3f})</td>
                </tr>
            """
        html_table += """
            </tbody>
        </table>
        """
        return html_table

    def generar_radar_grafico(self):
        """
        Genera un gráfico radar comparativo de las métricas (normalizadas) para todos los modelos.
        """
        modelos = list(self.resultados.keys())
        silhouettes = [self.resultados[m].get("silhouette") or 0 for m in modelos]
        ch_scores = [self.resultados[m].get("ch_score") or 0 for m in modelos]
        db_scores = [self.resultados[m].get("db_index") or 0 for m in modelos]
        db_inv = [1/x if x != 0 else 0 for x in db_scores]

        def normalizar(lista):
            mini = min(lista)
            maxi = max(lista)
            if maxi - mini == 0:
                return [0.5 for _ in lista]
            return [(x - mini) / (maxi - mini) for x in lista]

        silhouettes_norm = normalizar(silhouettes)
        ch_norm = normalizar(ch_scores)
        db_norm = normalizar(db_inv)

        angles = np.linspace(0, 2 * np.pi, len(modelos), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        def plot_metric(values, label, color):
            vals = values + values[:1]
            ax.plot(angles, vals, color=color, linewidth=2, label=label)
            ax.fill(angles, vals, color=color, alpha=0.25)
        plot_metric(silhouettes_norm, "Silhouette (norm.)", "blue")
        plot_metric(ch_norm, "Calinski-Harabasz (norm.)", "green")
        plot_metric(db_norm, "1/Davies-Bouldin (norm.)", "red")
        ax.set_thetagrids(np.degrees(angles[:-1]), modelos)
        ax.set_title("Comparación de Métricas (normalizadas)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        radar_img = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()
        return f'<img src="data:image/png;base64,{radar_img}" style="width:100%;">'

    def generar_grafico_barras_metricas(self):
        """
        Genera un conjunto de gráficos de barras (en 2x2) para cada métrica.
        """
        modelos = list(self.resultados.keys())
        silhouettes = [self.resultados[m].get("silhouette") or 0 for m in modelos]
        ch_scores = [self.resultados[m].get("ch_score") or 0 for m in modelos]
        db_scores = [self.resultados[m].get("db_index") or 0 for m in modelos]
        inertia = [self.resultados[m].get("inertia") or 0 for m in modelos]
        db_inv = [1/x if x != 0 else 0 for x in db_scores]

        fig, axs = plt.subplots(2, 2, figsize=(12,10))
        axs = axs.flatten()
        axs[0].bar(modelos, silhouettes, color="blue")
        axs[0].set_title("Silhouette Score")
        axs[1].bar(modelos, ch_scores, color="green")
        axs[1].set_title("Calinski-Harabasz Score")
        axs[2].bar(modelos, db_inv, color="red")
        axs[2].set_title("Inverso Davies-Bouldin")
        axs[3].bar(modelos, inertia, color="purple")
        axs[3].set_title("Inercia (K-Means)")
        for ax in axs:
            ax.set_ylabel("Valor")
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        barras_img = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()
        return f'<img src="data:image/png;base64,{barras_img}" style="width:100%;">'

    def generar_heatmap_metricas(self):
        """
        Genera un heatmap de correlación entre las métricas.
        """
        data = []
        modelos = list(self.resultados.keys())
        for m in modelos:
            row = {
                "Silhouette": self.resultados[m].get("silhouette") or np.nan,
                "Calinski-Harabasz": self.resultados[m].get("ch_score") or np.nan,
                "Davies-Bouldin": self.resultados[m].get("db_index") or np.nan,
                "Inercia": self.resultados[m].get("inertia") or np.nan
            }
            data.append(row)
        df_metricas = pd.DataFrame(data, index=modelos)
        corr = df_metricas.corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlación entre Métricas")
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        heatmap_img = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()
        return f'<img src="data:image/png;base64,{heatmap_img}" style="width:100%;">'

    def generar_clustermap_metricas(self):
        """
        Genera un clustermap de la correlación entre las métricas de evaluación,
        lo que permite agrupar los modelos según la similitud de sus métricas.
        """
        data = []
        modelos = list(self.resultados.keys())
        for m in modelos:
            row = {
                "Silhouette": self.resultados[m].get("silhouette") or np.nan,
                "Calinski-Harabasz": self.resultados[m].get("ch_score") or np.nan,
                "Davies-Bouldin": self.resultados[m].get("db_index") or np.nan,
                "Inercia": self.resultados[m].get("inertia") or np.nan
            }
            data.append(row)
        df_metricas = pd.DataFrame(data, index=modelos)
        # Calcular la correlación y reemplazar los NaN por 0
        corr = df_metricas.corr().fillna(0)
        clustermap = sns.clustermap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        clustermap.fig.suptitle("Clustermap de Correlación entre Métricas", y=1.02)
        buffer = BytesIO()
        clustermap.fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        clustermap_img = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close('all')
        return f'<img src="data:image/png;base64,{clustermap_img}" style="width:100%;">'

    def generar_grafico_elbow_line(self, X):
        """
        Genera el gráfico del codo para K-Means.
        """
        inertias = []
        ks = range(1, 11)
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42).fit(X)
            inertias.append(km.inertia_)
        plt.figure(figsize=(10,6))
        plt.plot(ks, inertias, marker='o')
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Inercia")
        plt.title("Método del Codo para K-Means")
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        elbow_img = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()
        return f'<img src="data:image/png;base64,{elbow_img}" style="width:100%;">'

    def generar_boxplot_clusters(self, df, labels, variable="TotalGastado"):
        """
        Genera un boxplot de la variable indicada por cluster.
        """
        if variable not in df.columns:
            return "<p>No se encontró la variable para el boxplot.</p>"
        df_box = df.copy()
        df_box["Cluster"] = labels
        plt.figure(figsize=(10,6))
        sns.boxplot(x="Cluster", y=variable, data=df_box)
        plt.title(f"Distribución de {variable} por Cluster")
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        box_img = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()
        return f'<img src="data:image/png;base64,{box_img}" style="width:100%;">'

    def generar_informe(self, output_file):
        # Sección de gráficos básicos y estadísticas
        tabla_html = self.generar_tabla_html()
        radar_grafico = self.generar_radar_grafico()
        barras_grafico = self.generar_grafico_barras_metricas()
        heatmap_grafico = self.generar_heatmap_metricas()
        clustermap_img = self.generar_clustermap_metricas()
        X = self.df_sample[self.df_sample.columns.intersection(["Quantity", "UnitPrice", "TotalGastado"])]
        if not X.empty:
            elbow_grafico = self.generar_grafico_elbow_line(X)
        else:
            elbow_grafico = "<p>No se pudo generar el gráfico del codo.</p>"
        mejor_modelo, _ = self.calcular_mejor_modelo()
        if mejor_modelo in self.labels_dict:
            boxplot_grafico = self.generar_boxplot_clusters(self.df_sample, self.labels_dict[mejor_modelo])
        else:
            boxplot_grafico = "<p>No se pudo generar el boxplot por falta de datos.</p>"

        # Construcción final del HTML
        html_content = f"""
        <html>
        <head>
            <title>Informe Comparativo de Clustering</title>
            <meta charset="UTF-8">
            <link rel="stylesheet" type="text/css" href="assets/styles.css">
        </head>
        <body>
            <div class="container">
                <h1>Comparación de Modelos de Clustering</h1>
                <h2>Resumen de Métricas</h2>
                {tabla_html}
                <h2>Gráficos Comparativos</h2>
                <div class="grafico">
                    <h3>Radar de Métricas</h3>
                    {radar_grafico}
                </div>
                <div class="grafico">
                    <h3>Gráfico de Barras</h3>
                    {barras_grafico}
                </div>
                <div class="grafico">
                    <h3>Heatmap de Correlación entre Métricas</h3>
                    {heatmap_grafico}
                </div>
                <div class="grafico">
                    <h3>Clustermap de Métricas</h3>
                    {clustermap_img}
                </div>
                <div class="grafico">
                    <h3>Método del Codo (K-Means)</h3>
                    {elbow_grafico}
                </div>
                <div class="grafico">
                    <h3>Boxplot del Mejor Modelo</h3>
                    {boxplot_grafico}
                </div>
            </div>
        </body>
        </html>
        """
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"Informe comparativo generado: {output_file}")

# ----------------------------------------------
# Dashboard Interactivo para Clustering
# ----------------------------------------------
def iniciar_dashboard_clustering(initial_n_clusters, informe_path, df, features):
    global df_clientes_global  # Aseguramos que se use la variable global

    # Preprocesamiento para generar datos en 3D para el scatterplot interactivo
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    pca3 = PCA(n_components=3)
    pcs = pca3.fit_transform(df_scaled[features])
    df_pca3 = pd.DataFrame(pcs, columns=["PC1", "PC2", "PC3"], index=df.index)
    # Asignamos clusters usando K-Means (ejemplo)
    km_model = KMeans(n_clusters=initial_n_clusters, random_state=42)
    df_pca3["Cluster"] = km_model.fit_predict(df_scaled[features])
    
    # Añadimos las columnas adicionales para permitir colorear por TotalGastado y TicketPromedio
    df_pca3["TotalGastado"] = df["TotalGastado"].values
    df_pca3["TicketPromedio"] = df["TicketPromedio"].values
    
    # Guardamos el DataFrame original de clientes en la variable global
    df_clientes_global = df.copy()

    # Creamos la aplicación Dash (Dash carga automáticamente los archivos CSS en la carpeta assets)
    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Store(id="store-df-clientes"),
        html.H1("Dashboard Interactivo de Clustering", className="container"),
        html.Div([
            html.Label("Algoritmo:", className="label"),
            dcc.Dropdown(
                id="algoritmo-dropdown",
                options=[
                    {"label": "K-Means", "value": "kmeans"},
                    {"label": "DBSCAN", "value": "dbscan"},
                    {"label": "Agglomerative", "value": "agglomerative"},
                    {"label": "Gaussian Mixture", "value": "gmm"}
                ],
                value="kmeans",
                className="dropdown"
            )
        ], style={"width": "40%", "margin": "auto"}),
        html.Br(),
        html.Div([
            html.Label("Número de clusters (para K-Means, Agglomerative y GMM):", className="label"),
            dcc.Slider(
                id="slider-clusters", min=2, max=10, step=1, value=initial_n_clusters,
                marks={i: str(i) for i in range(2, 11)},
                className="slider"
            )
        ], style={"width": "80%", "margin": "auto"}),
        html.Br(),
        html.Button("Actualizar Informe", id="btn-actualizar", n_clicks=0, className="btn"),
        html.Div(id="informe-clustering", className="info-box", style={"margin-top": "20px"}),
        html.H2("Scatterplot 3D Interactivo", className="container"),
        html.Div([
            html.Label("Variable para colorear:", className="label"),
            dcc.Dropdown(
                id="dropdown-color",
                options=[
                    {"label": "Cluster", "value": "Cluster"},
                    {"label": "TotalGastado", "value": "TotalGastado"},
                    {"label": "TicketPromedio", "value": "TicketPromedio"}
                ],
                value="Cluster",
                className="dropdown"
            )
        ], style={"width": "40%", "margin": "auto"}),
        dcc.Graph(id="scatter3d-graph"),
        html.Br(),
        html.H2("Histograma Interactivo", className="container"),
        dcc.Graph(id="histograma-graph")
    ])

    # Callback para almacenar el DataFrame de clientes en el Store
    @app.callback(
        Output("store-df-clientes", "data"),
        [Input("btn-actualizar", "n_clicks")]
    )
    def almacenar_df_clientes(n_clicks):
        global df_clientes_global
        return df_clientes_global.to_json(date_format="iso", orient="split")

    # Callback para actualizar el informe en el IFrame (mantiene la funcionalidad original)
    @app.callback(
        Output("informe-clustering", "children"),
        [Input("btn-actualizar", "n_clicks"),
        Input("slider-clusters", "value"),
        Input("algoritmo-dropdown", "value")]
    )
    def actualizar_informe(n_clicks, n_clusters, algoritmo):
        # Procesamos df_clientes a partir del df original
        if "InvoiceDate" in df.columns:
            procesador = ProcesadorDatos(df.copy(), df.copy())
            df_limpio = procesador.limpiar_datos(mantener_devoluciones=True, imputar_nulos=True)
            df_clientes = transformar_a_dataset_clientes(df_limpio)
        else:
            df_clientes = df.copy()
        features_local = features
        scaler = StandardScaler()
        df_clientes_scaled = df_clientes.copy()
        df_clientes_scaled[features_local] = scaler.fit_transform(df_clientes[features_local])
        
        # Entrenamos el modelo según el algoritmo seleccionado
        if algoritmo in ["dbscan", "agglomerative"]:
            if df_clientes_scaled.shape[0] > 10000:
                logging.warning("Para {} se usa una muestra de 10,000 registros.".format(algoritmo))
                df_sample = df_clientes_scaled.sample(n=10000, random_state=42)
            else:
                df_sample = df_clientes_scaled.copy()
            if algoritmo == "dbscan":
                entrenador = EntrenadorClustering_DBSCAN(df_sample)
                entrenador.entrenar(features_local, eps=0.7, min_samples=10)
            elif algoritmo == "agglomerative":
                entrenador = EntrenadorClustering_Agglomerative(df_sample)
                entrenador.entrenar(features_local, n_clusters=n_clusters)
            pca = PCA(n_components=2)
            X_sample = df_sample[features_local]
            df_pca = pd.DataFrame(pca.fit_transform(X_sample), columns=["PC1", "PC2"], index=df_sample.index)
        elif algoritmo == "gmm":
            if df_clientes_scaled.shape[0] > 10000:
                df_sample = df_clientes_scaled.sample(n=10000, random_state=42)
            else:
                df_sample = df_clientes_scaled.copy()
            entrenador = EntrenadorClustering_GMM(df_sample)
            entrenador.entrenar(features_local, n_components=n_clusters)
            pca = PCA(n_components=2)
            X_sample = df_sample[features_local]
            df_pca = pd.DataFrame(pca.fit_transform(X_sample), columns=["PC1", "PC2"], index=df_sample.index)
        else:
            entrenador = EntrenadorClustering_KMeans(df_clientes_scaled)
            entrenador.entrenar(features_local, n_clusters=n_clusters)
            pca = PCA(n_components=2)
            X_all = df_clientes_scaled[features_local]
            df_pca = pd.DataFrame(pca.fit_transform(X_all), columns=["PC1", "PC2"], index=df_clientes_scaled.index)
        
        entrenador.df = df_pca
        features_vis = list(df_pca.columns)
        generador = GeneradorInformeClustering(
            df.copy(), df_pca,
            "Información del dataset", df.head(), df.describe(),
            procesador.generar_tabla_limpieza() if "InvoiceDate" in df.columns else pd.DataFrame(),
            entrenador, features_vis, n_clusters
        )
        informe_file = f"informe_clustering_{algoritmo}_{n_clusters}.html"
        generador.generar_informe(informe_file)
        with open(informe_file, "r", encoding="utf-8") as f:
            html_informe = f.read()
        return html.Iframe(srcDoc=html_informe, style={"width": "100%", "height": "1200px", "border": "none"})

    # Callback para actualizar el scatterplot 3D interactivo
    @app.callback(
        Output("scatter3d-graph", "figure"),
        [Input("dropdown-color", "value")]
    )
    def actualizar_scatter3d(color_col):
        import plotly.express as px
        fig = px.scatter_3d(
            df_pca3,
            x="PC1", y="PC2", z="PC3",
            color=color_col,
            title="Scatterplot 3D de Clusters",
            hover_data=df_pca3.columns
        )
        return fig

    # Callback para actualizar el histograma interactivo
    @app.callback(
        Output("histograma-graph", "figure"),
        [Input("dropdown-color", "value"),
        Input("store-df-clientes", "data")]
    )
    def actualizar_histograma(var_color, df_json):
        import plotly.express as px
        from io import StringIO
        # Reconstruir el DataFrame a partir del JSON almacenado
        df_clientes = pd.read_json(StringIO(df_json), orient="split")
        if var_color not in df_clientes.columns:
            var_color = "TotalGastado"
        fig = px.histogram(df_clientes, x="TotalGastado", color=var_color, nbins=30,
                        title="Histograma interactivo de TotalGastado")
        return fig

    threading.Thread(target=app.run_server, kwargs={"debug": True, "use_reloader": False, "port":8050}).start()

# ----------------------------------------------
# Función para comparar los 4 algoritmos de Clustering
# ----------------------------------------------
def comparar_modelos_clustering(df, features, n_clusters=4):
    resultados = {}
    labels_dict = {}
    if df.shape[0] > 10000:
        df_sample = df.sample(n=10000, random_state=42)
    else:
        df_sample = df.copy()

    # K-Means
    km_entrenador = EntrenadorClustering_KMeans(df_sample)
    km_labels = km_entrenador.entrenar(features, n_clusters=n_clusters)
    resultados["K-Means"] = {
        "silhouette": km_entrenador.silhouette,
        "ch_score": km_entrenador.ch_score,
        "db_index": km_entrenador.db_index,
        "inertia": km_entrenador.model.inertia_ if hasattr(km_entrenador.model, "inertia_") else None
    }
    labels_dict["K-Means"] = km_labels

    # DBSCAN
    dbscan_entrenador = EntrenadorClustering_DBSCAN(df_sample)
    dbscan_labels = dbscan_entrenador.entrenar(features, eps=0.7, min_samples=10)
    resultados["DBSCAN"] = {
        "silhouette": dbscan_entrenador.silhouette,
        "ch_score": dbscan_entrenador.ch_score,
        "db_index": dbscan_entrenador.db_index,
        "inertia": None
    }
    labels_dict["DBSCAN"] = dbscan_labels

    # Agglomerative
    agglo_entrenador = EntrenadorClustering_Agglomerative(df_sample)
    agglo_labels = agglo_entrenador.entrenar(features, n_clusters=n_clusters)
    resultados["Agglomerative"] = {
        "silhouette": agglo_entrenador.silhouette,
        "ch_score": agglo_entrenador.ch_score,
        "db_index": agglo_entrenador.db_index,
        "inertia": None
    }
    labels_dict["Agglomerative"] = agglo_labels

    # GMM
    gmm_entrenador = EntrenadorClustering_GMM(df_sample)
    gmm_labels = gmm_entrenador.entrenar(features, n_components=n_clusters)
    resultados["GMM"] = {
        "silhouette": gmm_entrenador.silhouette,
        "ch_score": gmm_entrenador.ch_score,
        "db_index": gmm_entrenador.db_index,
        "inertia": None
    }
    labels_dict["GMM"] = gmm_labels

    return resultados, df_sample, labels_dict

# ----------------------------------------------
# 7. Ejecución Principal
# ----------------------------------------------
def main():
    global df_clientes_global  # Declaramos la variable global
    file_path = os.path.join("sample_data", "data.csv")
    logging.info("📥 Cargando datos...")
    sys.stdout.flush()
    cargador = CargadorDatos(file_path)
    df_original = cargador.cargar_datos()
    if df_original is None or df_original.empty:
        logging.error("No se pudo cargar el dataset o está vacío.")
        sys.stdout.flush()
        return

    info_text, head, describe = cargador.explorar_datos()
    logging.info("🛠️ Limpiando y transformando datos...")
    sys.stdout.flush()
    procesador = ProcesadorDatos(df_original.copy(), df_original.copy())
    df_limpio = procesador.limpiar_datos(mantener_devoluciones=True, imputar_nulos=True)
    if df_limpio is None or df_limpio.empty:
        logging.error("La limpieza resultó en un dataset vacío.")
        sys.stdout.flush()
        return
    stats_limpieza = procesador.generar_tabla_limpieza()
    logging.info("Tabla de limpieza generada.")
    sys.stdout.flush()

    df_clientes = transformar_a_dataset_clientes(df_limpio)
    logging.info("Transformación a datos de clientes completada.")
    sys.stdout.flush()

    # GUARDAR el DataFrame de clientes en un CSV para compartir con el menú
    try:
        df_clientes.to_csv("df_clientes.csv", index=False)
        logging.info("df_clientes.csv generado correctamente.")
    except Exception as e:
        logging.error(f"Error al guardar df_clientes.csv: {e}")
    
    # Guardamos el DataFrame de clientes en una variable global para usar en callbacks
    df_clientes_global = df_clientes.copy()

    features = ["TicketPromedio", "TotalCompras", "TotalGastado", "Recency", "NumeroProductosDistintos"]
    scaler = StandardScaler()
    df_clientes_scaled = df_clientes.copy()
    df_clientes_scaled[features] = scaler.fit_transform(df_clientes[features])
    logging.info("📊 Entrenando modelo de Clustering (K-Means) sobre datos de clientes...")
    sys.stdout.flush()
    entrenador_inicial = EntrenadorClustering_KMeans(df_clientes_scaled)
    n_clusters = 4
    entrenador_inicial.entrenar(features, n_clusters=n_clusters)

    logging.info("📑 Generando informe de Clustering individual...")
    sys.stdout.flush()
    generador = GeneradorInformeClustering(
        df_clientes.copy(), df_clientes_scaled[features].copy(),
        info_text, head, describe, stats_limpieza,
        entrenador_inicial, features, n_clusters
    )
    informe_file = "informe_clustering.html"
    generador.generar_informe(informe_file)
    logging.info("Informe individual generado: " + informe_file)
    sys.stdout.flush()

    if hasattr(entrenador_inicial, "labels"):
        df_result = df_clientes_scaled[features].copy()
        df_result['Cluster'] = entrenador_inicial.labels
        df_result.to_csv("clusters_asignados.csv", index=False)
        logging.info("Resultados exportados a clusters_asignados.csv")
        sys.stdout.flush()

    iniciar_dashboard_clustering(n_clusters, informe_file, df_clientes.copy(), features)
    logging.info("Dashboard iniciado en http://127.0.0.1:8050")
    sys.stdout.flush()

    logging.info("Generando informe comparativo de los 4 modelos...")
    sys.stdout.flush()
    resultados_comp, df_comp, labels_dict = comparar_modelos_clustering(df_clientes_scaled, features, n_clusters=n_clusters)
    comparador = GeneradorInformeComparacionClustering(resultados_comp, df_comp, labels_dict)
    informe_comp_file = f"informe_comparacion_clustering_{n_clusters}.html"
    comparador.generar_informe(informe_comp_file)
    logging.info(f"Informe comparativo de clustering generado: {informe_comp_file}")
    sys.stdout.flush()

    logging.info("Para ver el informe individual, abre " + informe_file + " en tu navegador.")
    logging.info("Para ver el informe comparativo, abre " + informe_comp_file + " en tu navegador.")
    logging.info("Para ver el dashboard, abre http://127.0.0.1:8050 en tu navegador.")
    logging.info("Para detener la ejecución, presiona Ctrl+C.")
    sys.stdout.flush()

if __name__ == "__main__":
    main()

