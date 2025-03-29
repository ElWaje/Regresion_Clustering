import subprocess
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Para entornos sin servidor de ventanas
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO, StringIO
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from ydata_profiling import ProfileReport
import threading
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import warnings
warnings.simplefilter("ignore", category=UserWarning)

# ------------------------------
# 📦 Instalación de Librerías Faltantes
# ------------------------------
# Agregamos 'prophet' y 'tensorflow' a la lista de librerías
required_libs = [
    "pandas", 
    "numpy", 
    "matplotlib", 
    "seaborn", 
    "sklearn", 
    "statsmodels",
    "pmdarima", 
    "plotly", 
    "dash", 
    "kaleido", 
    "ydata_profiling", 
    "wordcloud",
    "prophet",          # Para Prophet
    "torch"        # Para LSTM 
]

def install_missing_libs():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
    except subprocess.CalledProcessError:
        print("❌ Error: pip no está instalado. Instálalo antes de continuar.")
        return

    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            print(f"📦 Instalando {lib}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            except subprocess.CalledProcessError as e:
                print(f"❌ Error al instalar {lib}: {e}")

# Llamamos a la función de instalación
install_missing_libs()

# ------------------------------
# 📊 Importar Bibliotecas Adicionales
# ------------------------------
# IMPORTANTE:
# Si aparece un error de "No module named 'prophet'" o "No module named 'tensorflow'"
# la instalación podría requerir reiniciar el entorno. 
# Repitir o ejecutar nuevamente tras la instalación exitosa.

# Prophet para series temporales complejas
from prophet import Prophet

# Pytorch para el modelo LSTM
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        # Usamos 2 capas LSTM con dropout y batch_first=True para que la entrada tenga forma (batch, sequence, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        # Se toma la salida del último paso de tiempo
        out = self.fc(out[:, -1, :])
        return out

# ----------------------------------------------
# 1. Cargar y Explorar el Dataset
# ----------------------------------------------
class CargadorDatos:
    """Clase para cargar y explorar un dataset desde un archivo CSV."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def cargar_datos(self):
        if not os.path.exists(self.file_path):
            print(f"❌ Error: El archivo '{self.file_path}' no existe.")
            return None
        if not self.file_path.lower().endswith(".csv"):
            print(f"⚠️ Advertencia: '{self.file_path}' no es un archivo CSV.")
            return None
        try:
            self.df = pd.read_csv(self.file_path, encoding="ISO-8859-1", low_memory=False)
            print(f"✅ Datos cargados correctamente desde '{self.file_path}' ({len(self.df)} registros).")
            return self.df
        except pd.errors.ParserError as e:
            print(f"❌ Error al analizar el archivo CSV: {e}")
            return None
        except Exception as e:
            print(f"❌ Error al cargar el archivo CSV: {e}")
            return None

    def explorar_datos(self):
        if self.df is None:
            print("⚠️ Advertencia: No hay datos cargados para explorar.")
            return None, None, None
        info_buf = StringIO()
        self.df.info(buf=info_buf)
        info = info_buf.getvalue()
        return info, self.df.head(), self.df.describe()

# ----------------------------------------------
# 2. Limpieza y Transformación de Datos
# ----------------------------------------------
class ProcesadorDatos:
    """Clase para limpiar y transformar el dataset."""
    def __init__(self, df, df_original):
        self.df = df.copy()
        self.df_original = df_original.copy()
        self.stats_limpieza = {}

    def limpiar_datos(self, mantener_devoluciones=True, imputar_nulos=True, usar_log=False):
        """
        Limpia el DataFrame y, de forma opcional, crea la columna `LogTotalVentas`
        (transformación logarítmica) si usar_log=True.
        """
        self.stats_limpieza["Registros originales"] = len(self.df)
        self.stats_limpieza["Valores nulos antes"] = self.df.isna().sum().sum()

        # 1) Devoluciones
        if mantener_devoluciones:
            print("✅ Se mantienen transacciones con valores negativos en Quantity (devoluciones).")
        else:
            # Si NO se quieren devoluciones, filtramos transacciones con Quantity > 0
            self.df = self.df[self.df['Quantity'] > 0]

        # 2) Imputar Nulos vs Eliminar
        if imputar_nulos:
            print("✅ Se imputan valores faltantes en lugar de eliminarlos.")
            for col in ["CustomerID", "Quantity", "UnitPrice"]:
                if col in self.df.columns and self.df[col].isna().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
        else:
            self.df.dropna(subset=['CustomerID'], inplace=True)

        # 3) Calcular TotalVentas
        if "Quantity" in self.df.columns and "UnitPrice" in self.df.columns:
            self.df["TotalVentas"] = self.df["Quantity"] * self.df["UnitPrice"]
            self.df["TotalVentas"] = self.df["TotalVentas"].fillna(0)
        else:
            print("❌ Error: No se encontraron las columnas necesarias para calcular `TotalVentas`.")
            return None

        # 4) Convertir InvoiceDate
        if "InvoiceDate" in self.df.columns:
            try:
                self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"], errors="coerce")
                print("✅ 'InvoiceDate' convertido correctamente a datetime.")
            except Exception as e:
                print(f"❌ Error al convertir 'InvoiceDate': {e}")
        else:
            print("❌ Error: No se encontró la columna `InvoiceDate`.")

        # 5) Variables Derivadas
        self.df["Year"] = self.df["InvoiceDate"].dt.year
        self.df["Month"] = self.df["InvoiceDate"].dt.month
        self.df["Quarter"] = self.df["InvoiceDate"].dt.quarter
        self.df["DayOfWeek"] = self.df["InvoiceDate"].dt.dayofweek
        self.df["Hour"] = self.df["InvoiceDate"].dt.hour

        # 6) Transformación logarítmica opcional
        if usar_log:
            # Para evitar problemas con ventas = 0, usamos log1p -> log(1 + x)
            self.df["LogTotalVentas"] = np.log1p(self.df["TotalVentas"])
            print("✅ Se aplicó transformación logarítmica a 'TotalVentas' (LogTotalVentas).")

        self.stats_limpieza["Registros después de limpieza"] = len(self.df)
        self.stats_limpieza["Valores nulos después"] = self.df.isna().sum().sum()

        print(f"✅ Limpieza completada: {len(self.df)} registros restantes.")
        return self.df

    def escalar_datos(self, metodo="StandardScaler"):
        escaladores = {
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
            "MinMaxScaler": MinMaxScaler()
        }
        if metodo not in escaladores:
            print(f"⚠️ Método {metodo} no válido. Usando StandardScaler por defecto.")
            metodo = "StandardScaler"
        self.scaler = escaladores[metodo]

        # Puedes ajustar si quieres escalar también 'LogTotalVentas'
        columnas_numericas = ["Quantity", "UnitPrice", "TotalVentas"]
        self.df[columnas_numericas] = self.scaler.fit_transform(self.df[columnas_numericas])
        print(f"✅ Datos escalados con {metodo}.")
        return self.df

    def generar_tabla_limpieza(self):
        return pd.DataFrame.from_dict(self.stats_limpieza, orient='index', columns=['Valor'])

# ----------------------------------------------
# 3. División del Dataset en Train-Test
# ----------------------------------------------
class DivisorDatos:
    """Divide el dataset en Entrenamiento, Validación y Test Final."""
    def __init__(self, df, usar_log=False):
        print("Versión actualizada de DivisorDatos cargada")
        """
        usar_log: si deseas que, tras agrupar por día,
                se calcule también la columna 'LogTotalVentas'
                como log(1 + sum_of_TotalVentas).
        """
        self.df = df.copy()
        self.usar_log = usar_log
        self.filas_train = 0
        self.filas_validation = 0
        self.filas_test = 0

    def dividir_datos(self):
        # 1) Agrupamos por fecha y sumamos 'TotalVentas' 
        self.df = (
            self.df
            .groupby(self.df["InvoiceDate"].dt.date)["TotalVentas"]
            .sum()
            .reset_index()
        )

        # 2) Convertimos a datetime
        self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"])
        self.df.set_index("InvoiceDate", inplace=True)

        # 3) Resample diario, llenar NAs con 0
        self.df = self.df.resample('D').sum().fillna(0)

        # 4) Si usar_log=True, creamos la columna 'LogTotalVentas'
        if self.usar_log:
            self.df["LogTotalVentas"] = np.log1p(self.df["TotalVentas"])
            print("✅ Se creó columna 'LogTotalVentas' con la suma diaria.")

        # 5) Definimos fechas de división
        train_start = "2010-12-01"
        train_end   = "2011-10-08"
        val_start   = "2011-10-09"
        val_end     = "2011-11-08"
        test_start  = "2011-11-09"
        test_end    = "2011-12-09"

        # 6) Hacemos la partición
        train      = self.df.loc[train_start:train_end]
        validation = self.df.loc[val_start:val_end]
        test       = self.df.loc[test_start:test_end]

        self.filas_train = len(train)
        self.filas_validation = len(validation)
        self.filas_test = len(test)

        print(f"✅ División completada:")
        print(f"  - Entrenamiento: {self.filas_train} filas ({train_start} a {train_end})")
        print(f"  - Validación: {self.filas_validation} filas ({val_start} a {val_end})")
        print(f"  - Test Final: {self.filas_test} filas ({test_start} a {test_end})")

        return train, validation, test

# ----------------------------------------------
# 4. Entrenamiento de Modelos
# ----------------------------------------------class EntrenadorModelos:
class EntrenadorModelos: 
    """
    Clase para entrenar distintos modelos de predicción, 
    aplicando el escalado más apropiado para cada uno:
    - ARIMA: sin escalado (univariante).
    - Regresión Lineal: StandardScaler.
    - Random Forest: RobustScaler.
    - Prophet: sin escalado (serie univariante).
    - LSTM: MinMaxScaler (usando PyTorch).
    Además, si usar_log=True, se usará la columna 'LogTotalVentas' en lugar de 'TotalVentas',
    y se invertirá la predicción con expm1 para calcular errores en la escala real.
    """
    def __init__(self, train_series, validation_series, test_series, usar_log=False):
        self.train_series = train_series
        self.validation_series = validation_series
        self.test_series = test_series
        self.usar_log = usar_log  # Si es True, usamos LogTotalVentas

        self.results = {}
        self.predicciones = {}

    # ------------------------------------------------------------
    # Auxiliar: Evaluar un modelo (Regresión, RF, etc.)
    # ------------------------------------------------------------
    def evaluar_modelo(self, modelo, nombre, X_train, y_train, X_val, X_test):
        modelo.fit(X_train, y_train.ravel())
        pred_val = modelo.predict(X_val)
        pred_test = modelo.predict(X_test)

        # Si usar_log, invertimos expm1 para pasar a escala real
        if self.usar_log:
            pred_val = np.expm1(pred_val)
            pred_test = np.expm1(pred_test)
            real_val = np.expm1(self.validation_series.values)
            real_test = np.expm1(self.test_series.values)
        else:
            real_val = self.validation_series.values
            real_test = self.test_series.values

        mae_val  = mean_absolute_error(real_val, pred_val)
        rmse_val = np.sqrt(mean_squared_error(real_val, pred_val))
        mae_test  = mean_absolute_error(real_test, pred_test)
        rmse_test = np.sqrt(mean_squared_error(real_test, pred_test))

        self.results[nombre] = {
            "MAE_Validation": mae_val,
            "RMSE_Validation": rmse_val,
            "MAE_Test": mae_test,
            "RMSE_Test": rmse_test
        }
        self.predicciones[nombre] = {
            "Validation": pred_val,
            "Test": pred_test
        }

        print(f"✅ {nombre} Validación: MAE={mae_val:.2f}, RMSE={rmse_val:.2f}")
        print(f"✅ {nombre} Test Final: MAE={mae_test:.2f}, RMSE={rmse_test:.2f}")

    # ------------------------------------------------------------
    # Método que aplica un escalador a X_train, X_val, X_test
    # ------------------------------------------------------------
    def escalar_features(self, X_train, X_val, X_test, metodo="standard"):
        if metodo is None:
            return X_train, X_val, X_test

        escaladores = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler()
        }
        if metodo not in escaladores:
            print(f"⚠️ Escalador '{metodo}' no reconocido. No se escalara.")
            return X_train, X_val, X_test

        scaler = escaladores[metodo]
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
        X_test_scaled  = scaler.transform(X_test)
        print(f"✅ Se aplicó escalado: {metodo}")
        return X_train_scaled, X_val_scaled, X_test_scaled

    # ------------------------------------------------------------
    # ARIMA => sin escalado, univariante
    # ------------------------------------------------------------
    def entrenar_arima(self):
        print("\n📈 Entrenando modelo ARIMA...")
        try:
            target_col = "LogTotalVentas" if self.usar_log else "TotalVentas"
            serie_train = (
                self.train_series[target_col]
                .asfreq("D")
                .ffill()
                .bfill()
                .dropna()
            )
            if len(serie_train) < 30:
                print("❌ Error: La serie ARIMA es muy corta o vacía.")
                return

            modelo_arima = ARIMA(serie_train, order=(3, 1, 1))
            modelo_ajustado = modelo_arima.fit()

            serie_val = self.validation_series[target_col].asfreq('D').ffill().bfill().dropna()
            serie_test = self.test_series[target_col].asfreq('D').ffill().bfill().dropna()

            pred_val  = modelo_ajustado.forecast(steps=len(serie_val))
            pred_test = modelo_ajustado.forecast(steps=len(serie_test))

            self.evaluar_modelo_arima(pred_val, pred_test)
        except Exception as e:
            print(f"❌ Error en ARIMA: {e}")

    def evaluar_modelo_arima(self, pred_val, pred_test):
        if self.usar_log:
            pred_val_inv  = np.expm1(pred_val)
            pred_test_inv = np.expm1(pred_test)
            real_val  = np.expm1(self.validation_series.values)
            real_test = np.expm1(self.test_series.values)
        else:
            pred_val_inv  = pred_val
            pred_test_inv = pred_test
            real_val  = self.validation_series.values
            real_test = self.test_series.values

        mae_val  = mean_absolute_error(real_val, pred_val_inv)
        rmse_val = np.sqrt(mean_squared_error(real_val, pred_val_inv))
        mae_test  = mean_absolute_error(real_test, pred_test_inv)
        rmse_test = np.sqrt(mean_squared_error(real_test, pred_test_inv))

        self.results["ARIMA"] = {
            "MAE_Validation": mae_val,
            "RMSE_Validation": rmse_val,
            "MAE_Test": mae_test,
            "RMSE_Test": rmse_test
        }
        self.predicciones["ARIMA"] = {
            "Validation": pred_val_inv,
            "Test": pred_test_inv
        }

        print(f"✅ ARIMA Validación: MAE={mae_val:.2f}, RMSE={rmse_val:.2f}")
        print(f"✅ ARIMA Test Final: MAE={mae_test:.2f}, RMSE={rmse_test:.2f}")

    # ------------------------------------------------------------
    # REGRESIÓN LINEAL => StandardScaler por defecto
    # ------------------------------------------------------------
    def entrenar_regresion_lineal(self):
        print("\n📉 Entrenando modelo de Regresión Lineal...")
        X_train = np.arange(len(self.train_series)).reshape(-1, 1)
        X_val   = np.arange(len(self.train_series), len(self.train_series) + len(self.validation_series)).reshape(-1, 1)
        X_test  = np.arange(len(self.train_series) + len(self.validation_series),
                            len(self.train_series) + len(self.validation_series) + len(self.test_series)).reshape(-1, 1)
        X_train_s, X_val_s, X_test_s = self.escalar_features(X_train, X_val, X_test, metodo="standard")
        target_col = "LogTotalVentas" if self.usar_log else "TotalVentas"
        y_train = self.train_series[target_col].values
        y_val   = self.validation_series[target_col].values
        y_test  = self.test_series[target_col].values

        modelo_lr = LinearRegression()
        nombre_modelo = "Regresión Lineal"
        self.evaluar_modelo(modelo_lr, nombre_modelo, X_train_s, y_train, X_val_s, X_test_s)

    # ------------------------------------------------------------
    # RANDOM FOREST => RobustScaler por defecto
    # ------------------------------------------------------------
    def entrenar_random_forest(self):
        print("\n🌲 Entrenando modelo Random Forest...")
        X_train = np.arange(len(self.train_series)).reshape(-1, 1)
        X_val   = np.arange(len(self.train_series), len(self.train_series) + len(self.validation_series)).reshape(-1, 1)
        X_test  = np.arange(len(self.train_series) + len(self.validation_series),
                            len(self.train_series) + len(self.validation_series) + len(self.test_series)).reshape(-1, 1)
        X_train_s, X_val_s, X_test_s = self.escalar_features(X_train, X_val, X_test, metodo="robust")
        target_col = "LogTotalVentas" if self.usar_log else "TotalVentas"
        y_train = self.train_series[target_col].values
        y_val   = self.validation_series[target_col].values
        y_test  = self.test_series[target_col].values

        modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        nombre_modelo = "Random Forest"
        self.evaluar_modelo(modelo_rf, nombre_modelo, X_train_s, y_train, X_val_s, X_test_s)

    # ------------------------------------------------------------
    # PROPHET => sin escalado
    # ------------------------------------------------------------
    def entrenar_prophet(self):
        print("\n📅 Entrenando modelo Prophet...")
        try:
            target_col = "LogTotalVentas" if self.usar_log else "TotalVentas"
            df_train = self.train_series[target_col].asfreq('D').fillna(0).reset_index()
            df_train.columns = ["ds", "y"]
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            m.fit(df_train)

            steps_val = len(self.validation_series)
            steps_test = len(self.test_series)
            future = m.make_future_dataframe(periods=steps_val + steps_test, freq='D')
            forecast = m.predict(future)
            forecast_val  = forecast.iloc[-(steps_val + steps_test):-steps_test]
            forecast_test = forecast.iloc[-steps_test:]
            pred_val  = forecast_val["yhat"].values
            pred_test = forecast_test["yhat"].values

            if self.usar_log:
                pred_val_inv  = np.expm1(pred_val)
                pred_test_inv = np.expm1(pred_test)
                real_val  = np.expm1(self.validation_series[target_col].values)
                real_test = np.expm1(self.test_series[target_col].values)
            else:
                pred_val_inv  = pred_val
                pred_test_inv = pred_test
                real_val  = self.validation_series[target_col].values
                real_test = self.test_series[target_col].values

            mae_val  = mean_absolute_error(real_val, pred_val_inv)
            rmse_val = np.sqrt(mean_squared_error(real_val, pred_val_inv))
            mae_test  = mean_absolute_error(real_test, pred_test_inv)
            rmse_test = np.sqrt(mean_squared_error(real_test, pred_test_inv))

            self.results["Prophet"] = {
                "MAE_Validation": mae_val,
                "RMSE_Validation": rmse_val,
                "MAE_Test": mae_test,
                "RMSE_Test": rmse_test
            }
            self.predicciones["Prophet"] = {
                "Validation": pred_val_inv,
                "Test": pred_test_inv
            }

            print(f"✅ Prophet Validación: MAE={mae_val:.2f}, RMSE={rmse_val:.2f}")
            print(f"✅ Prophet Test Final: MAE={mae_test:.2f}, RMSE={rmse_test:.2f}")
        except Exception as e:
            print(f"❌ Error en Prophet: {e}")

    # ------------------------------------------------------------
    # LSTM => Usando PyTorch
    # ------------------------------------------------------------
    def entrenar_lstm(self, sequence_length=7, epochs=50):
        print("\n🔮 Entrenando modelo LSTM con PyTorch (mejorado)...")
        try:
            target_col = "LogTotalVentas" if self.usar_log else "TotalVentas"
            # Extraer series y convertir a arrays
            train_series = self.train_series[target_col].asfreq('D').fillna(0)
            val_series   = self.validation_series[target_col].asfreq('D').fillna(0)
            test_series  = self.test_series[target_col].asfreq('D').fillna(0)
            # Imprimir longitudes para depuración
            print("Longitud train_series:", len(train_series))
            print("Longitud val_series:", len(val_series))
            print("Longitud test_series:", len(test_series))
            train_array = train_series.values
            val_array   = val_series.values
            test_array  = test_series.values
            # Guardar fechas
            train_dates = train_series.index
            val_dates   = val_series.index
            test_dates  = test_series.index
            # Escalar con MinMaxScaler
            minmax = MinMaxScaler()
            train_scaled = minmax.fit_transform(train_array.reshape(-1, 1))
            val_scaled   = minmax.transform(val_array.reshape(-1, 1))
            test_scaled  = minmax.transform(test_array.reshape(-1, 1))
            # Función para crear secuencias
            def create_sequences(data, window=7):
                X, y = [], []
                for i in range(len(data) - window):
                    X.append(data[i: i + window])
                    y.append(data[i + window])
                return np.array(X), np.array(y)
            X_train, y_train = create_sequences(train_scaled, sequence_length)
            X_val,   y_val   = create_sequences(val_scaled, sequence_length)
            X_test,  y_test  = create_sequences(test_scaled, sequence_length)
            # Convertir a tensores (las secuencias ya tienen forma (num_secuencias, window, 1))
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor   = torch.tensor(y_val, dtype=torch.float32)
            X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor  = torch.tensor(y_test, dtype=torch.float32)
            
            # Instanciar el modelo LSTM modificado
            model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Ajuste de tasa de aprendizaje
            
            # Ciclo de entrenamiento (más epochs)
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs.squeeze(), y_train_tensor)
                loss.backward()
                optimizer.step()
                if (epoch+1) % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
            # Evaluación en validación y test
            model.eval()
            with torch.no_grad():
                pred_val = model(X_val_tensor).squeeze().numpy()
                pred_test = model(X_test_tensor).squeeze().numpy()
            # Invertir el escalado
            pred_val_inv = minmax.inverse_transform(pred_val.reshape(-1, 1))
            pred_test_inv = minmax.inverse_transform(pred_test.reshape(-1, 1))
            real_val_inv = minmax.inverse_transform(y_val_tensor.numpy().reshape(-1, 1))
            real_test_inv = minmax.inverse_transform(y_test_tensor.numpy().reshape(-1, 1))
            # Invertir log si corresponde
            if self.usar_log:
                pred_val_inv = np.expm1(pred_val_inv)
                pred_test_inv = np.expm1(pred_test_inv)
                real_val_inv = np.expm1(real_val_inv)
                real_test_inv = np.expm1(real_test_inv)
            # Alinear fechas
            val_dates_aligned = val_dates[sequence_length:]
            test_dates_aligned = test_dates[sequence_length:]
            # Calcular métricas
            mae_val = mean_absolute_error(real_val_inv, pred_val_inv)
            rmse_val = np.sqrt(mean_squared_error(real_val_inv, pred_val_inv))
            mae_test = mean_absolute_error(real_test_inv, pred_test_inv)
            rmse_test = np.sqrt(mean_squared_error(real_test_inv, pred_test_inv))
            self.results["LSTM"] = {
                "MAE_Validation": mae_val,
                "RMSE_Validation": rmse_val,
                "MAE_Test": mae_test,
                "RMSE_Test": rmse_test
            }
            self.predicciones["LSTM"] = {
                "Validation": pred_val_inv.flatten(),
                "Validation_Dates": val_dates_aligned,
                "Test": pred_test_inv.flatten(),
                "Test_Dates": test_dates_aligned
            }
            print(f"✅ LSTM (PyTorch) Validación: MAE={mae_val:.2f}, RMSE={rmse_val:.2f}")
            print(f"✅ LSTM (PyTorch) Test:       MAE={mae_test:.2f}, RMSE={rmse_test:.2f}")
        except Exception as e:
            print(f"❌ Error en LSTM (PyTorch): {e}")
        
    # ------------------------------------------------------------
    # Escoger Mejor Modelo según RMSE_Test
    # ------------------------------------------------------------
    def seleccionar_mejor_modelo(self):
        mejor_modelo = min(self.results, key=lambda m: self.results[m]["RMSE_Test"])
        mejor_resultado = self.results[mejor_modelo]
        print("\n🏆 Mejor Modelo:")
        print(f"   📊 Modelo: {mejor_modelo}")
        print(f"   🔹 MAE Test: {mejor_resultado['MAE_Test']:.2f}")
        print(f"   🔹 RMSE Test: {mejor_resultado['RMSE_Test']:.2f}")
        return mejor_modelo, mejor_resultado

    # ------------------------------------------------------------
    # Entrenar Todos con la configuración "por defecto"
    # ------------------------------------------------------------
    def entrenar_todos(self):
        print("\n🚀 Entrenando todos los modelos...")
        self.entrenar_arima()
        self.entrenar_regresion_lineal()
        self.entrenar_random_forest()
        self.entrenar_prophet()
        # Usar la versión PyTorch del LSTM
        self.entrenar_lstm()
        mejor_modelo, mejor_resultado = self.seleccionar_mejor_modelo()
        print("\n✅ ¡Todos los modelos han sido entrenados y evaluados!")
        return self.predicciones, mejor_modelo, mejor_resultado
   
# ----------------------------------------------
# 5. Generación del Informe EDA (Exploratory Data Analysis)
# ----------------------------------------------

class GeneradorInformeEDA:
    """Genera el informe EDA con gráficos y estadísticas generales."""

    def __init__(
        self,
        df_original,         # Dataset antes de la limpieza
        df,                  # Dataset después de la limpieza
        info_text,
        head,
        describe,
        stats_limpieza,
        filas_train,
        filas_validation,
        filas_test
    ):
        self.df_original = df_original
        self.df = df
        self.info_text = info_text
        self.head = head
        self.describe = describe
        self.stats_limpieza = stats_limpieza
        self.filas_train = filas_train
        self.filas_validation = filas_validation
        self.filas_test = filas_test
        self.graficos = {}

    def guardar_grafico(self, plt_figure):
        """Convierte un gráfico de Matplotlib en una imagen en base64 para incluirlo en el HTML."""
        try:
            buffer = BytesIO()
            plt_figure.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            imagen_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()
            plt.close(plt_figure)
            return ( 
                f'<img src="data:image/png;base64,{imagen_base64}" '
                f'style="width:100%; border-radius:10px; box-shadow:2px 2px 10px gray;">'
            )
        except Exception as e:
            print(f"❌ Error al guardar gráfico: {e}")
            return None

    def generar_distribucion_limpieza(self):
        """
        Genera un histograma comparando la distribución real de 'TotalVentas'
        antes y después de la limpieza de datos, usando df_original vs df.
        """
        try:
            # Verificamos que 'TotalVentas' exista en ambos DataFrames
            if "TotalVentas" not in self.df.columns or "TotalVentas" not in self.df_original.columns:
                print("⚠️ No se encontró la columna 'TotalVentas' en df o df_original. "
                    "No se puede generar 'distribucion_limpieza'.")
                return

            # Extraemos las series
            ventas_antes = self.df_original["TotalVentas"].dropna()
            ventas_despues = self.df["TotalVentas"].dropna()

            # Creamos un DataFrame unificado con la etiqueta "Estado"
            data_antes = pd.DataFrame({"Ventas": ventas_antes, "Estado": "Antes de la Limpieza"})
            data_despues = pd.DataFrame({"Ventas": ventas_despues, "Estado": "Después de la Limpieza"})
            data_final = pd.concat([data_antes, data_despues], ignore_index=True)

            # Generamos el histograma comparativo
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(
                data=data_final,
                x="Ventas",
                hue="Estado",
                kde=True,
                bins=50,
                alpha=0.7,
                palette={"Antes de la Limpieza": "red", "Después de la Limpieza": "blue"}
            )
            ax.set_title("Distribución de Ventas Antes y Después de la Limpieza", fontsize=14)
            ax.set_xlabel("Total de Ventas")
            ax.set_ylabel("Frecuencia")
            ax.legend(title="Estado")
            ax.grid(True)

            # Guardamos la figura en self.graficos
            self.graficos["distribucion_limpieza"] = self.guardar_grafico(fig)

        except Exception as e:
            print(f"❌ Error al generar el gráfico de distribución de limpieza: {e}")

    def generar_graficos(self):
        """
        Genera los gráficos exploratorios (histogramas, boxplots, correlaciones, etc.)
        y los guarda en self.graficos. Posteriormente, generar_html() los incluirá en el informe.
        """
        # 1) Convertir "InvoiceDate" si es necesario y agregar columnas "Hora" y "Día"
        try:
            if "InvoiceDate" in self.df.columns:
                if not pd.api.types.is_datetime64_any_dtype(self.df["InvoiceDate"]):
                    self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"], errors="coerce")
                self.df["Hora"] = pd.to_numeric(self.df["InvoiceDate"].dt.hour, errors="coerce").astype("Int64")
                self.df["Día"] = self.df["InvoiceDate"].dt.day_name()
        except Exception as e:
            print(f"❌ Error al procesar 'InvoiceDate': {e}")

        # 2) Generar histogramas para columnas numéricas (excepto 'Hora')
        for column in self.df.select_dtypes(include=[np.number]).columns:
            if column == "Hora":
                continue
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(
                    self.df[column].dropna(),
                    bins=30,
                    kde=True,
                    ax=ax,
                    color="blue",
                    label=column
                )
                ax.set_title(f"Distribución de {column}", fontsize=14)
                ax.legend()
                self.graficos[f"hist_{column}"] = self.guardar_grafico(fig)
            except Exception as e:
                print(f"❌ Error al generar histograma para {column}: {e}")

        # 3) Generar boxplots para columnas numéricas (excepto 'Hora')
        for column in self.df.select_dtypes(include=[np.number]).columns:
            if column == "Hora":
                continue
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=self.df[column].dropna(), ax=ax, color="green")
                ax.set_title(f"Boxplot de {column}", fontsize=14)
                self.graficos[f"box_{column}"] = self.guardar_grafico(fig)
            except Exception as e:
                print(f"❌ Error al generar boxplot para {column}: {e}")

        # 4) Generar stripplots para detectar outliers
        for column in self.df.select_dtypes(include=[np.number]).columns:
            if column == "Hora":
                continue
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.stripplot(
                    x=self.df[column].dropna(),
                    ax=ax,
                    color="red",
                    size=1,
                    jitter=True,
                    alpha=0.4,
                    label=f"Outliers en {column}"
                )
                ax.set_title(f"Outliers en {column} (Stripplot)", fontsize=14)
                self.graficos[f"outliers_{column}"] = self.guardar_grafico(fig)
            except Exception as e:
                print(f"❌ Error al generar stripplot para {column}: {e}")

        # 5) Generar matriz de correlación (usando solo columnas numéricas)
        try:
            df_numerico = self.df.select_dtypes(include=[np.number])
            if not df_numerico.empty and len(df_numerico.columns) > 1:
                corr_matrix = df_numerico.corr()
                if not corr_matrix.empty:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        corr_matrix,
                        annot=True,
                        cmap="coolwarm",
                        fmt=".2f",
                        linewidths=0.5,
                        ax=ax
                    )
                    ax.set_title("Matriz de Correlación", fontsize=14)
                    self.graficos["correlacion"] = self.guardar_grafico(fig)
                else:
                    print("⚠️ La matriz de correlación está vacía.")
            else:
                print("⚠️ No hay suficientes columnas numéricas para generar la matriz de correlación.")
        except Exception as e:
            print(f"❌ Error al generar matriz de correlación: {e}")

        # 6) Generar heatmap de ventas por hora y día
        try:
            if "InvoiceDate" in self.df.columns and "TotalVentas" in self.df.columns:
                self.df["Hora"] = self.df["InvoiceDate"].dt.hour
                self.df["Día"] = self.df["InvoiceDate"].dt.day_name()
                pivot_table = self.df.pivot_table(
                    index="Día",
                    columns="Hora",
                    values="TotalVentas",
                    aggfunc="sum",
                    fill_value=0,
                )
                if not pivot_table.empty:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.heatmap(
                        pivot_table,
                        cmap="coolwarm",
                        annot=True,
                        fmt=".0f",
                        linewidths=0.5,
                        ax=ax
                    )
                    ax.set_title("Heatmap de Ventas por Hora y Día", fontsize=14)
                    self.graficos["heatmap_ventas"] = self.guardar_grafico(fig)
                else:
                    print("⚠️ El pivot table está vacío para Heatmap.")
        except Exception as e:
            print(f"❌ Error al generar heatmap: {e}")

        # 7) Generar pairplot de variables numéricas
        try:
            df_numerico = self.df.select_dtypes(include=[np.number])
            if len(df_numerico.columns) > 1:
                fig = sns.pairplot(self.df, diag_kind="kde", markers="o", plot_kws={'alpha': 0.5})
                fig.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
                self.graficos["pairplot"] = self.guardar_grafico(fig.fig)
        except Exception as e:
            print(f"❌ Error al generar pairplot: {e}")

        # 8) Gráfico de tendencia de ventas mensuales
        try:
            if "InvoiceDate" in self.df.columns and "TotalVentas" in self.df.columns:
                self.df["YearMonth"] = self.df["InvoiceDate"].dt.to_period("M")
                ventas_mensuales = (
                    self.df.groupby("YearMonth")["TotalVentas"].sum().reset_index()
                )
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.lineplot(
                    x=ventas_mensuales["YearMonth"].astype(str),
                    y=ventas_mensuales["TotalVentas"],
                    ax=ax,
                    color="blue",
                    label="Tendencia",
                )
                ax.set_title("Tendencia de Ventas Mensuales", fontsize=14)
                ax.tick_params(axis='x', rotation=45)
                ax.legend()
                self.graficos["ventas_mensuales"] = self.guardar_grafico(fig)
        except Exception as e:
            print(f"❌ Error al generar gráfico de tendencia de ventas mensuales: {e}")

        # 9) Generar wordcloud (si existe la columna "Description")
        try:
            from wordcloud import WordCloud
            if "Description" in self.df.columns:
                text = " ".join(self.df["Description"].dropna().astype(str))
                if text.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    ax.set_title("Wordcloud de Descripción de Productos", fontsize=14)
                    self.graficos["wordcloud"] = self.guardar_grafico(fig)
        except ImportError:
            print("⚠️ `wordcloud` no está instalado. No se generará el gráfico de palabras.")
        except Exception as e:
            print(f"❌ Error al generar wordcloud: {e}")

        # 10) Distribución Acumulada de Ventas (CDF)
        try:
            if "TotalVentas" in self.df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                sorted_data = np.sort(self.df["TotalVentas"].dropna())
                cdf = np.arange(len(sorted_data)) / float(len(sorted_data))
                ax.plot(sorted_data, cdf, marker='.', linestyle='none', label="CDF")
                ax.set_title("Distribución Acumulada de Ventas (CDF)", fontsize=14)
                ax.set_xlabel("Total de Ventas")
                ax.set_ylabel("CDF")
                self.graficos["cdf_ventas"] = self.guardar_grafico(fig)
        except Exception as e:
            print(f"❌ Error al generar CDF: {e}")

        # 11) Generar distribución real de limpieza (en lugar de datos sintéticos)
        try:
            self.generar_distribucion_limpieza()
        except Exception as e:
            print(f"❌ Error al generar la distribución de limpieza: {e}")

        # 12) Eliminar de self.graficos aquellas entradas cuyo valor sea None
        try:
            keys_to_remove = [key for key, value in self.graficos.items() if value is None]
            for key in keys_to_remove:
                del self.graficos[key]
                print(f"❌ Eliminando gráfico no generado: {key}")
        except Exception as e:
            print(f"❌ Error al eliminar gráficos no generados: {e}")

    def generar_html(self, output_file):
        """
        Genera el informe HTML final con todos los gráficos y tablas.
        """
        # 1) Generar los gráficos
        self.generar_graficos()

        # 2) Secciones de HTML
        info_html = f"""
        <div class="info-box">
            <h3>Información General del Dataset</h3>
            <p><b>Total de registros:</b> {self.df.shape[0]}</p>
            <p><b>Total de columnas:</b> {self.df.shape[1]}</p>
            <p><b>Memoria utilizada:</b> {round(self.df.memory_usage(deep=True).sum()/1024**2,2)} MB</p>
        </div>
        <div class="table-container">
            <table>
                <tr>
                    <th>#</th>
                    <th>Columna</th>
                    <th>No Nulos</th>
                    <th>Tipo de Dato</th>
                </tr>
        """
        for i, col in enumerate(self.df.columns):
            non_nulls = self.df[col].count()
            dtype = str(self.df[col].dtype)
            info_html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{col}</td>
                    <td>{non_nulls} / {self.df.shape[0]}</td>
                    <td>{dtype}</td>
                </tr>
            """
        info_html += "</table></div>"

        division_html = f"""
        <div class="info-box">
            <h3>División del Dataset</h3>
            <p>El dataset ha sido dividido en:</p>
            <ul>
                <li><b>Entrenamiento:</b> {self.filas_train} filas</li>
                <li><b>Validación:</b> {self.filas_validation} filas</li>
                <li><b>Test Final:</b> {self.filas_test} filas</li>
            </ul>
            <p>Se mantiene el orden temporal para evitar fuga de información.</p>
        </div>
        """

        # 3) Construir contenido HTML final
        html_content = f"""
        <html>
        <head>
            <title>Informe de Análisis Exploratorio de Datos (EDA)</title>
            <meta charset="UTF-8">
            <link rel="stylesheet" type="text/css" href="assets/styles.css">
        </head>
        <body>
            <div class="container">
                <h1>Informe de Análisis Exploratorio de Datos (EDA)</h1>
                <h2>1. Información del Dataset</h2>
                {info_html}
                <h2>2. División del Dataset</h2>
                {division_html}
                <h2>3. Estadísticas Generales</h2>
                <div class="table-container">
                    {self.describe.to_html(classes='table table-striped', na_rep='-')}
                </div>
                <h2>4. Gráficos Exploratorios</h2>
                <h3>Matriz de Correlación</h3> 
                    {self.graficos.get("correlacion", "<p>❌ No disponible</p>")}
                <h3>Tendencia de Ventas Mensuales</h3>
                    {self.graficos.get("ventas_mensuales", "<p>❌ No disponible</p>")}
                <h3>Distribución Acumulada de Ventas (CDF)</h3>
                    {self.graficos.get("cdf_ventas", "<p>❌ No disponible</p>")}
                <h3>Distribución de Ventas Antes y Después de la Limpieza</h3>
                    {self.graficos.get("distribucion_limpieza", "<p>❌ No disponible</p>")}
                <h2>Análisis de Outliers</h2>
        """
        # 4) Insertar histogramas, boxplots y outliers para cada columna numérica
        for column in list(self.df.select_dtypes(include=[np.number]).columns):
            if f"hist_{column}" in self.graficos:
                html_content += f"""
                <h3>Distribución de {column}</h3>
                {self.graficos.get(f"hist_{column}")}
                """
            if f"box_{column}" in self.graficos:
                html_content += f"""
                <h3>Boxplot de {column}</h3>
                {self.graficos.get(f"box_{column}")}
                """
            if f"outliers_{column}" in self.graficos:
                html_content += f"""
                <h3>Outliers en {column} (Stripplot)</h3>
                {self.graficos.get(f"outliers_{column}")}
                """

        # 5) Pairplot
        if "pairplot" in self.graficos:
            html_content += f"""
                <h2>Relación Entre Variables</h2>
                <h3>Pairplot de Variables Numéricas</h3>
                {self.graficos.get("pairplot", "<p>No disponible</p>")}
            """

        # 6) Cerrar HTML
        html_content += "</div></body></html>"

        # 7) Guardar en archivo
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print("\n✅ Informe generado correctamente:", output_file)

# -------------------------------
# 6. Generación del Informe HTML
# -------------------------------
class GeneradorInforme:
    """Genera el informe de predicción de ventas con gráficos y estadísticas generales."""
    def __init__(self, df_original, df, info_text, describe, resultados, predicciones,
                filas_train, filas_validation, filas_test, mejor_modelo, mejor_resultado,
                train_data, validation_data, test_data):

        # DataFrames y textos informativos
        self.df_original = df_original
        self.df = df
        self.info_text = info_text
        self.describe = describe

        # Resultados del entrenamiento
        self.resultados = resultados
        self.predicciones = predicciones
        self.mejor_modelo = mejor_modelo
        self.mejor_resultado = mejor_resultado

        # Tamaño datasets divididos
        self.filas_train = filas_train
        self.filas_validation = filas_validation
        self.filas_test = filas_test

        # Datos específicos para gráficos o análisis adicional
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

        # Diccionario para guardar gráficos
        self.graficos = {}

    def guardar_grafico(self, plt_figure):
        """Convierte un gráfico de Matplotlib en una imagen en base64 para incluirlo en el HTML."""
        try:
            buffer = BytesIO()
            plt_figure.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            imagen_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()
            plt.close(plt_figure)
            return f'<img src="data:image/png;base64,{imagen_base64}" style="width:100%; border-radius: 10px; box-shadow: 2px 2px 10px gray;">'
        except Exception as e:
            print(f"❌ Error al guardar gráfico: {e}")
            return None

    def guardar_grafico_plotly(self, fig):
        """Convierte un gráfico de Plotly en HTML interactivo."""
        try:
            return fig.to_html(full_html=False, include_plotlyjs="cdn")
        except Exception as e:
            print(f"❌ Error al guardar gráfico interactivo: {e}")
            return "<p>❌ Error al generar gráfico interactivo</p>"

    def histograma_variable_clave(self):
        try:
            fig = px.histogram(
                self.df,
                x='TotalVentas',
                nbins=30,
                title='Distribución de Ventas Totales'
            )
            fig.update_layout(xaxis_title='Total Ventas', yaxis_title='Frecuencia')
            return fig
        except Exception as e:
            print(f"❌ Error histograma variable clave: {e}")
            return None


    def scatter_modelo_predictivo(self):
        try:
            # Ejemplo de datos sintéticos
            df = pd.DataFrame({
                'real': np.random.rand(100) * 100,
                'prediccion': np.random.rand(100) * 100
            })
            fig = px.scatter(
                df, x='real', y='prediccion', trendline='ols',
                title='Predicción vs Realidad (Modelo Predictivo)'
            )
            fig.update_layout(xaxis_title='Real', yaxis_title='Predicción')
            return fig
        except Exception as e:
            print(f"❌ Error scatter modelo predictivo: {e}")
            return None


    def heatmap_transformacion_logaritmica(self):
        try:
            df_corr = self.df.select_dtypes(include=[np.number]).corr()
            fig = ff.create_annotated_heatmap(
                z=df_corr.values,
                x=df_corr.columns.tolist(),
                y=df_corr.columns.tolist(),
                colorscale='Viridis'
            )
            fig.update_layout(title="Mapa de Calor - Correlaciones Numéricas")
            return fig
        except Exception as e:
            print(f"❌ Error heatmap Transformación: {e}")
            return None

    def scatter_modelo_predictivo(self):
        try:
            # Ejemplo de datos sintéticos
            df = pd.DataFrame({
                'real': np.random.rand(100) * 100,
                'prediccion': np.random.rand(100) * 100
            })
            fig = px.scatter(
                df, x='real', y='prediccion', trendline='ols',
                title='Predicción vs Realidad (Modelo Predictivo)'
            )
            fig.update_layout(xaxis_title='Real', yaxis_title='Predicción')
            return fig
        except Exception as e:
            print(f"❌ Error scatter modelo predictivo: {e}")
            return None


    def scatter_3d_ventas(self):
        try:
            df = self.df.copy()
            required_cols = ['Quantity', 'UnitPrice', 'TotalVentas', 'Country']

            if all(col in df.columns for col in required_cols):
                # Filtra valores extremos y negativos (asumiendo que no son útiles)
                df_filtrado = df[
                    (df["Quantity"] > 0) &
                    (df["UnitPrice"] > 0) &
                    (df["TotalVentas"] > 0) &
                    (df["Quantity"] < df["Quantity"].quantile(0.99)) &
                    (df["UnitPrice"] < df["UnitPrice"].quantile(0.99)) &
                    (df["TotalVentas"] < df["TotalVentas"].quantile(0.99))
                ].dropna(subset=['Country'])

                # Verifica si hay datos suficientes después del filtrado
                if df_filtrado.empty:
                    print("⚠️ Después del filtrado, no quedan datos para graficar.")
                    return None

                # Usa una muestra para evitar saturar Plotly
                df_sample = df_filtrado.sample(min(2000, len(df_filtrado)))

                fig = px.scatter_3d(
                    df_sample,
                    x='Quantity', 
                    y='UnitPrice', 
                    z='TotalVentas',
                    color='Country',
                    title='Ventas 3D (filtrado valores extremos)'
                )

                fig.update_layout(scene=dict(
                    xaxis_title='Cantidad',
                    yaxis_title='Precio Unitario',
                    zaxis_title='Ventas Totales'
                ))

                return fig
            else:
                missing_cols = [col for col in required_cols if col not in df.columns]
                print(f"⚠️ Columnas faltantes para scatter 3D ventas: {missing_cols}")
                return None
        except Exception as e:
            print(f"❌ Error creando scatter 3D ventas: {e}")
            return None


    def mapa_ventas(self):
        """Mapa interactivo de Ventas por País (Plotly)."""
        try:
            # Verificar que la columna "Country" exista
            if "Country" not in self.df.columns:
                raise ValueError("La columna 'Country' no se encuentra en el DataFrame.")
            df_filtrado = self.df.groupby("Country")["TotalVentas"].sum().reset_index()
            df_filtrado = df_filtrado[df_filtrado["TotalVentas"] > 0]
            if df_filtrado.empty:
                print("⚠️ Advertencia: No hay datos positivos de ventas para graficar.")
                return None
            fig = px.scatter_geo(
                df_filtrado,
                locations="Country",
                locationmode="country names",
                size="TotalVentas",
                title="🌍 Mapa de Ventas por País",
                projection="natural earth"
            )
            return fig
        except Exception as e:
            print(f"❌ Error en mapa_ventas: {e}")
            return None

    def animacion_ventas(self):
        """Animación de Evolución de Ventas (Plotly)."""
        try:
            # Comprobar que 'InvoiceDate' y 'TotalVentas' existen y que 'InvoiceDate' es datetime
            if "InvoiceDate" not in self.df.columns or "TotalVentas" not in self.df.columns:
                raise ValueError("Las columnas 'InvoiceDate' o 'TotalVentas' no están en el DataFrame.")
            if not pd.api.types.is_datetime64_any_dtype(self.df["InvoiceDate"]):
                self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"], errors="coerce")
            self.df["YearMonth"] = self.df["InvoiceDate"].dt.to_period("M").astype(str)
            fig = px.bar(
                self.df,
                x="YearMonth",
                y="TotalVentas",
                color="Country" if "Country" in self.df.columns else None,
                animation_frame="YearMonth",
                title="📊 Evolución de Ventas con el Tiempo",
                color_continuous_scale="Blues"
            )
            return fig
        except Exception as e:
            print(f"❌ Error en animacion_ventas: {e}")
            return None

    def generar_graficos(self):
        self.graficos = {}  
        # 1) Gráficos interactivos (Plotly)
        graficos_funciones = {
            "scatter_3d_ventas": self.scatter_3d_ventas,
            "scatter_modelo_predictivo": self.scatter_modelo_predictivo,
            "heatmap_transformacion_logaritmica": self.heatmap_transformacion_logaritmica,
            "histograma_variable_clave": self.histograma_variable_clave,
            "mapa_ventas": self.mapa_ventas,
            "animacion_ventas": self.animacion_ventas
        }

        for nombre, funcion in graficos_funciones.items():
            try:
                fig = funcion()
                if fig is not None:
                    self.graficos[nombre] = self.guardar_grafico_plotly(fig)
                else:
                    self.graficos[nombre] = f"<p>❌ No disponible (error en {nombre})</p>"
            except Exception as e:
                print(f"❌ Excepción inesperada en {nombre}: {e}")
                self.graficos[nombre] = "<p>❌ No disponible (excepción inesperada)</p>"
        # 2) mapa de ventas
        temp_fig = None
        try:
            temp_fig = self.mapa_ventas()
            if temp_fig is not None:
                self.graficos["mapa_ventas"] = self.guardar_grafico_plotly(temp_fig)
            else:
                self.graficos["mapa_ventas"] = ""
        except Exception as e:
            print(f"❌ Excepción inesperada en mapa_ventas: {e}")
            self.graficos["mapa_ventas"] = ""

        # 3) animación de ventas
        try:
            fig_anim = self.animacion_ventas()
            if fig_anim is not None:
                self.graficos["animacion_ventas"] = self.guardar_grafico_plotly(fig_anim)
            else:
                self.graficos["animacion_ventas"] = ""
        except Exception as e:
            print(f"❌ Excepción inesperada en animacion_ventas: {e}")
            self.graficos["animacion_ventas"] = ""
    
        # Gráficos de Matplotlib
        # 1) Histograma de Ventas Totales
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=self.df, x="Quantity", y="TotalVentas", hue="TotalVentas", 
                            size="Quantity", sizes=(10, 200), alpha=0.7, palette="coolwarm", ax=ax)
            ax.set_title("📊 Relación entre Cantidad Vendida y Ventas Totales", fontsize=14)
            self.graficos["scatter_ventas"] = self.guardar_grafico(fig)
        except Exception as e:
            print(f"❌ Error en scatter_ventas: {e}")
            self.graficos["scatter_ventas"] = "<p>Error en scatter_ventas</p>"

        #2) Treemap de ventas por país (Plotly)
        try:
            if "Country" in self.df.columns and "TotalVentas" in self.df.columns:
                ventas_pais = self.df.groupby("Country")["TotalVentas"].sum().reset_index()
                fig = px.treemap(ventas_pais, path=["Country"], values="TotalVentas",
                                title="🌍 Treemap de Ventas por País", color="TotalVentas",
                                color_continuous_scale="Blues")
                fig.update_layout(margin=dict(t=30, l=10, r=10, b=10))
                self.graficos["treemap_ventas"] = self.guardar_grafico_plotly(fig)
            else:
                self.graficos["treemap_ventas"] = "<p>No se puede generar treemap_ventas</p>"
        except Exception as e:
            print(f"❌ Error en treemap_ventas: {e}")
            self.graficos["treemap_ventas"] = "<p>Error en treemap_ventas</p>"

        #3) Sankey de ventas por país y cliente (Plotly)
        try:
            if "Country" in self.df.columns and "CustomerID" in self.df.columns and "TotalVentas" in self.df.columns:
                ventas_pais_cliente = self.df.groupby(["Country", "CustomerID"])["TotalVentas"].sum().reset_index()
                sources = list(ventas_pais_cliente["Country"])
                targets = list(ventas_pais_cliente["CustomerID"].astype(str))
                values = list(ventas_pais_cliente["TotalVentas"])
                fig = go.Figure(go.Sankey(
                    node=dict(label=sources + targets, pad=15, thickness=20, color="blue"),
                    link=dict(source=[sources.index(s) for s in sources],
                            target=[len(sources) + targets.index(t) for t in targets],
                            value=values)
                ))
                fig.update_layout(title_text="🔄 Flujo de Ventas por País y Cliente", font_size=12)
                self.graficos["sankey_ventas"] = self.guardar_grafico_plotly(fig)
            else:
                self.graficos["sankey_ventas"] = "<p>No se puede generar sankey_ventas</p>"
        except Exception as e:
            print(f"❌ Error en sankey_ventas: {e}")
            self.graficos["sankey_ventas"] = "<p>Error en sankey_ventas</p>"

        #4) Gráfico de radar interactivo (Plotly)
        try:
            modelos = ["ARIMA", "Regresión Lineal", "Random Forest", "Prophet", "LSTM"]
            mae = [self.resultados.get(m, {}).get("MAE_Test", None) for m in modelos]
            rmse = [self.resultados.get(m, {}).get("RMSE_Test", None) for m in modelos]
            if any(mae) and any(rmse) and all(val is not None for val in mae + rmse):
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=mae, theta=modelos, fill='toself', name="MAE"))
                fig.add_trace(go.Scatterpolar(r=rmse, theta=modelos, fill='toself', name="RMSE"))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                                title="📊 Comparación de Modelos de Predicción")
                self.graficos["radar_modelos"] = self.guardar_grafico_plotly(fig)
            else:
                self.graficos["radar_modelos"] = "<p>No se puede generar radar_modelos</p>"
        except Exception as e:
            print(f"❌ Error en radar_modelos: {e}")
            self.graficos["radar_modelos"] = "<p>Error en radar_modelos</p>"

        #5) Histograma de compras por cliente (Matplotlib)
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            compras_cliente = self.df.groupby("CustomerID")["TotalVentas"].sum()
            if not compras_cliente.empty:
                sns.histplot(compras_cliente, bins=30, kde=True, ax=ax, color="purple")
                ax.set_title("📊 Histograma de Compras por Cliente")
                self.graficos["hist_compras_cliente"] = self.guardar_grafico(fig)
            else:
                print("⚠️ Advertencia: No se generó el histograma de compras por cliente porque los datos están vacíos.")
                self.graficos["hist_compras_cliente"] = "<p>No hay datos para hist_compras_cliente</p>"
        except Exception as e:
            print(f"❌ Error en hist_compras_cliente: {e}")
            self.graficos["hist_compras_cliente"] = "<p>Error en hist_compras_cliente</p>"

        #6) Tendencia de ventas diarias (Matplotlib)
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(x=self.df.index, y=self.df["TotalVentas"], ax=ax, color="blue", label="Ventas Diarias")
            ax.set_title("📈 Tendencia de Ventas Diarias", fontsize=14)
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Total Ventas")
            # Solo se llama a legend() si hay handles (artistas) con labels
            handles, labels = ax.get_legend_handles_labels()
            if handles and any(label and not label.startswith("_") for label in labels):
                ax.legend()
            self.graficos["tendencia"] = self.guardar_grafico(fig)
        except Exception as e:
            print(f"❌ Error en tendencia: {e}")
            self.graficos["tendencia"] = "<p>Error en tendencia</p>"

        #7) Distribución de ventas (Matplotlib)
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(self.df["TotalVentas"], bins=30, kde=True, color="green")
            ax.set_title("📊 Distribución de Ventas", fontsize=14)
            self.graficos["distribucion"] = self.guardar_grafico(fig)
        except Exception as e:
            print(f"❌ Error en distribucion: {e}")
            self.graficos["distribucion"] = "<p>Error en distribucion</p>"

        #8) Mapa de calor de correlación (Matplotlib)
        try:
            df_numeric = self.df.select_dtypes(include=["number"])
            if not df_numeric.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
                ax.set_title("📌 Mapa de Calor de Correlación")
                self.graficos["correlacion"] = self.guardar_grafico(fig)
            else:
                self.graficos["correlacion"] = "<p>No se puede generar correlacion</p>"
        except Exception as e:
            print(f"❌ Error en correlacion: {e}")
            self.graficos["correlacion"] = "<p>Error en correlacion</p>"

        #9) Boxplot de ventas totales (Matplotlib)
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(y=self.df["TotalVentas"], ax=ax, color="orange")
            ax.set_title("📌 Boxplot de Ventas Totales")
            self.graficos["boxplot"] = self.guardar_grafico(fig)
        except Exception as e:
            print(f"❌ Error en boxplot: {e}")
            self.graficos["boxplot"] = "<p>Error en boxplot</p>"

        #10) Gráfico de barras: Ventas por país (Matplotlib)
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            ventas_pais = self.df.groupby("Country")["TotalVentas"].sum().sort_values(ascending=False).head(10)
            sns.barplot(x=ventas_pais.values, y=ventas_pais.index, ax=ax, hue=ventas_pais.index,
                        palette="Blues_r", legend=False)
            ax.set_title("📌 Ventas Totales por País")
            ax.set_xlabel("Total de Ventas")
            ax.set_ylabel("País")
            self.graficos["ventas_pais"] = self.guardar_grafico(fig)
        except Exception as e:
            print(f"❌ Error en ventas_pais: {e}")
            self.graficos["ventas_pais"] = "<p>Error en ventas_pais</p>"

        #11) Histograma de precios unitarios (Matplotlib)
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(self.df["UnitPrice"], bins=30, kde=True, color="red")
            ax.set_title("📌 Distribución de Precios Unitarios")
            self.graficos["hist_precios"] = self.guardar_grafico(fig)
        except Exception as e:
            print(f"❌ Error en hist_precios: {e}")
            self.graficos["hist_precios"] = "<p>Error en hist_precios</p>"

        #12) Gráfico de violín: Precios unitarios (Matplotlib)
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.violinplot(y=self.df["UnitPrice"], ax=ax, color="purple")
            ax.set_title("📌 Distribución de Precios Unitarios (Violín)")
            self.graficos["violin_precios"] = self.guardar_grafico(fig)
        except Exception as e:
            print(f"❌ Error en violin_precios: {e}")
            self.graficos["violin_precios"] = "<p>Error en violin_precios</p>"

        #13) Gráfico de violín: Cantidades vendidas (Matplotlib)
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.violinplot(y=self.df["Quantity"], ax=ax, color="teal")
            ax.set_title("📌 Distribución de Cantidades Vendidas (Violín)")
            self.graficos["violin_cantidades"] = self.guardar_grafico(fig)
        except Exception as e:
            print(f"❌ Error en violin_cantidades: {e}")
            self.graficos["violin_cantidades"] = "<p>Error en violin_cantidades</p>"

        # 14) Comparación de predicciones vs datos reales (Matplotlib)
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Lista de modelos a considerar
            lista_modelos = ["ARIMA", "Regresión Lineal", "Random Forest", "Prophet", "LSTM"]
            # Recolectamos las predicciones disponibles para cada modelo
            modelos_disponibles = {m: self.predicciones[m]["Test"] for m in lista_modelos if m in self.predicciones and "Test" in self.predicciones[m]}
            
            if not modelos_disponibles:
                print("⚠️ No hay predicciones Test para ninguno de los modelos.")
            else:
                # Calculamos la longitud mínima de predicciones entre todos los modelos
                longitudes = [len(preds) for preds in modelos_disponibles.values()]
                n_common = min(longitudes)
                
                # Usamos las últimas n_common fechas del índice de test
                fechas_test = self.df.index[-n_common:]
                
                # Para la serie real, también usamos las últimas n_common ventas
                if "TotalVentas" in self.df.columns and len(self.df["TotalVentas"]) >= n_common:
                    sns.lineplot(x=fechas_test, y=self.df["TotalVentas"][-n_common:], ax=ax, label="Datos Reales", color="black")
                else:
                    print("⚠️ No se pudo graficar 'Datos Reales' (faltan filas o columna 'TotalVentas').")
                
                # Asignamos colores a cada modelo
                color_map = {
                    "ARIMA": "blue",
                    "Regresión Lineal": "red",
                    "Random Forest": "green",
                    "Prophet": "purple",
                    "LSTM": "orange"
                }
                
                # Recorremos cada modelo y recortamos sus predicciones a n_common
                for modelo, preds in modelos_disponibles.items():
                    preds_common = np.array(preds).flatten()[-n_common:]
                    sns.lineplot(x=fechas_test, y=preds_common, ax=ax, label=modelo, color=color_map.get(modelo, "gray"))
                
                ax.set_title("📊 Comparación de Predicciones en el Test", fontsize=14)
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Ventas")
                ax.legend()
                self.graficos["comparacion_modelos"] = self.guardar_grafico(fig)
        except Exception as e:
            print(f"❌ Error en comparacion_modelos: {e}")
            self.graficos["comparacion_modelos"] = "<p>Error en comparacion_modelos</p>"

        #15) Ventas por día de la semana (Matplotlib)
        try:
            self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"])
            self.df["DayOfWeek"] = self.df["InvoiceDate"].dt.dayofweek
            ventas_dia = self.df.groupby("DayOfWeek")["TotalVentas"].sum()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=ventas_dia.index, y=ventas_dia.values, hue=ventas_dia.index, palette="viridis", legend=False)
            ax.set_xticks(range(7))
            ax.set_xticklabels(["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"])
            ax.set_title("📊 Ventas Totales por Día de la Semana")
            ax.set_xlabel("Día de la Semana")
            ax.set_ylabel("Ventas Totales")
            self.graficos["ventas_dia"] = self.guardar_grafico(fig)
        except Exception as e:
            print(f"❌ Error en ventas_dia: {e}")
            self.graficos["ventas_dia"] = "<p>Error en ventas_dia</p>"

        #16) Tendencia de ventas: Entrenamiento, Validación y Test (Matplotlib)
        try:
            fig, ax = plt.subplots(figsize=(12, 5))

            if not self.train_data.empty:
                ax.plot(self.train_data["InvoiceDate"], self.train_data["TotalVentas"],
                        label="Entrenamiento", color="blue", marker="o")
            else:
                print("⚠️ No hay datos de entrenamiento para graficar.")

            if not self.validation_data.empty:
                ax.plot(self.validation_data["InvoiceDate"], self.validation_data["TotalVentas"],
                        label="Validación", color="orange", marker="s")
            else:
                print("⚠️ No hay datos de validación para graficar.")

            if not self.test_data.empty:
                ax.plot(self.test_data["InvoiceDate"], self.test_data["TotalVentas"],
                        label="Test", color="red", marker="^")
            else:
                print("⚠️ No hay datos de test para graficar.")

            ax.set_title("📈 Tendencia de Ventas: Entrenamiento, Validación y Test", fontsize=14)
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Total Ventas")
            # Solo se llama a legend() si hay handles (artistas) con labels
            handles, labels = ax.get_legend_handles_labels()
            if handles and any(label and not label.startswith("_") for label in labels):
                ax.legend()
            self.graficos["tendencia_train_test"] = self.guardar_grafico(fig)
        except Exception as e:
            print(f"❌ Error en tendencia_train_test: {e}")
            self.graficos["tendencia_train_test"] = "<p>Error en tendencia_train_test</p>"

        # 17) Ejemplo de gráfico de comparación de predicciones (varios modelos)
        try:
            fig, ax = plt.subplots(figsize=(10, 5))

            # Lista de modelos a considerar
            lista_modelos = ["ARIMA", "Regresión Lineal", "Random Forest", "Prophet", "LSTM"]
            # Recolectamos los modelos que tienen predicciones en 'Test'
            modelos_disponibles = [m for m in lista_modelos if m in self.predicciones and "Test" in self.predicciones[m]]

            if len(modelos_disponibles) == 0:
                print("⚠️ No hay predicciones Test para ARIMA / RL / RF / Prophet / LSTM.")
            else:
                # Para LSTM, si se han guardado fechas específicas, las usamos; de lo contrario, usamos las últimas 'n' fechas del DataFrame.
                if "LSTM" in modelos_disponibles and "Test_Dates" in self.predicciones["LSTM"]:
                    fechas_test = pd.to_datetime(self.predicciones["LSTM"]["Test_Dates"])
                else:
                    n = len(self.predicciones[modelos_disponibles[0]]["Test"])
                    fechas_test = self.df.index[-n:]
                
                # Aquí, ajustamos para que la longitud de fechas_test y de las predicciones sea la misma:
                n_dates = len(fechas_test)
                n_preds = len(self.predicciones[modelos_disponibles[0]]["Test"])
                n_common = min(n_dates, n_preds)
                fechas_test = fechas_test[-n_common:]  # Tomamos las últimas n_common fechas

                # Dibujar la serie real si es posible (también recortamos si es necesario)
                if "TotalVentas" in self.df.columns and len(self.df["TotalVentas"]) >= n_common:
                    real_series = self.df["TotalVentas"].iloc[-n_common:]
                    sns.lineplot(x=fechas_test, y=real_series, ax=ax, label="Datos Reales", color="black")
                else:
                    print("⚠️ Falta la columna 'TotalVentas' o no hay suficientes filas para trazar los datos reales.")

                # Mapa de colores para cada modelo
                color_map = {
                    "ARIMA": "blue",
                    "Regresión Lineal": "red",
                    "Random Forest": "green",
                    "Prophet": "purple",
                    "LSTM": "orange"
                }

                # Recorrer cada modelo y graficar sus predicciones recortadas
                for modelo in modelos_disponibles:
                    preds = np.array(self.predicciones[modelo]["Test"]).flatten()
                    preds = preds[:n_common]  # Aseguramos que tenga n_common elementos
                    sns.lineplot(x=fechas_test, y=preds, ax=ax,
                                label=modelo,
                                color=color_map.get(modelo, "gray"))

                ax.set_title("Comparación de Predicciones en el Test", fontsize=14)
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Ventas")
                handles, labels = ax.get_legend_handles_labels()
                if handles and any(label and not label.startswith("_") for label in labels):
                    ax.legend()
                self.graficos["comparacion_modelos"] = self.guardar_grafico(fig)
        except Exception as e:
            print(f"❌ Error en comparacion_modelos: {e}")
            self.graficos["comparacion_modelos"] = "<p>Error en comparacion_modelos</p>"

    def generar_html(self, output_file):
        self.generar_graficos()
        info_html = f"""
        <div class="info-box">
            <h3>Información General del Dataset</h3>
            <p><b>Total de registros:</b> {self.df.shape[0]}</p>
            <p><b>Total de columnas:</b> {self.df.shape[1]}</p>
            <p><b>Memoria utilizada:</b> {round(self.df.memory_usage(deep=True).sum()/1024**2,2)} MB</p>
        </div>
        <div class="table-container">
            <table>
                <tr>
                    <th>#</th>
                    <th>Columna</th>
                    <th>No Nulos</th>
                    <th>Tipo de Dato</th>
                </tr>
        """
        for i, col in enumerate(self.df.columns):
            non_nulls = self.df[col].count()
            dtype = str(self.df[col].dtype)
            info_html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{col}</td>
                    <td>{non_nulls} / {self.df.shape[0]}</td>
                    <td>{dtype}</td>
                </tr>
            """
        info_html += "</table></div>"
        division_html = f"""
        <div class="info-box">
            <h3>División del Dataset</h3>
            <p>El dataset ha sido dividido en:</p>
            <ul>
                <li><b>Entrenamiento:</b> {self.filas_train} filas</li>
                <li><b>Validación:</b> {self.filas_validation} filas</li>
                <li><b>Test Final:</b> {self.filas_test} filas</li>
            </ul>
            <p>Se mantiene el orden temporal para evitar fuga de información.</p>
        </div>
        """
        metricas_html = """
        <div class="info-box">
            <h3>Métricas de Modelos de Predicción</h3>
            <table>
                <tr><th>Modelo</th><th>MAE</th><th>RMSE</th></tr>
        """
        if "ARIMA" in self.resultados and "MAE_Test" in self.resultados["ARIMA"]:
            metricas_html += f"""
                <tr><td>ARIMA</td><td>{self.resultados["ARIMA"]["MAE_Test"]:.2f}</td><td>{self.resultados["ARIMA"]["RMSE_Test"]:.2f}</td></tr>
            """
        else:
            print("⚠️ No hay resultados para ARIMA.")
        if "Regresión Lineal" in self.resultados and "MAE_Test" in self.resultados["Regresión Lineal"]:
            metricas_html += f"""
                <tr><td>Regresión Lineal</td><td>{self.resultados["Regresión Lineal"]["MAE_Test"]:.2f}</td><td>{self.resultados["Regresión Lineal"]["RMSE_Test"]:.2f}</td></tr>
            """
        else:
            print("⚠️ No hay resultados para Regresión Lineal.")
        if "Random Forest" in self.resultados and "MAE_Test" in self.resultados["Random Forest"]:
            metricas_html += f"""
                <tr><td>Random Forest</td><td>{self.resultados["Random Forest"]["MAE_Test"]:.2f}</td><td>{self.resultados["Random Forest"]["RMSE_Test"]:.2f}</td></tr>
            """
        else:
            print("⚠️ No hay resultados para Random Forest.")
        if "Prophet" in self.resultados and "MAE_Test" in self.resultados["Prophet"]:
            metricas_html += f"""
                <tr>
                    <td>Prophet</td>
                    <td>{self.resultados["Prophet"]["MAE_Test"]:.2f}</td>
                    <td>{self.resultados["Prophet"]["RMSE_Test"]:.2f}</td>
                </tr>
            """
        else:
            print("⚠️ No hay resultados para Prophet.")

        if "LSTM" in self.resultados and "MAE_Test" in self.resultados["LSTM"]:
            metricas_html += f"""
                <tr>
                    <td>LSTM</td>
                    <td>{self.resultados["LSTM"]["MAE_Test"]:.2f}</td>
                    <td>{self.resultados["LSTM"]["RMSE_Test"]:.2f}</td>
                </tr>
            """
        else:
            print("⚠️ No hay resultados para LSTM.")

        metricas_html += "</table></div>"
        mejor_modelo_html = f"""
        <div class="info-box">
            <h3>Mejor Modelo Seleccionado</h3>
            <p>El modelo más preciso basado en RMSE Test es:</p>
            <h2>{self.mejor_modelo}</h2>
            <ul>
                <li><b>MAE Test:</b> {self.mejor_resultado["MAE_Test"]:.2f}</li>
                <li><b>RMSE Test:</b> {self.mejor_resultado["RMSE_Test"]:.2f}</li>
            </ul>
            <p>Este modelo ha demostrado ser el más preciso para predecir ventas futuras.</p>
        </div>
        """
        html_content = f"""
        <html>
        <head>
            <title>Informe de Predicción de Ventas</title>
            <meta charset="UTF-8">
            <link rel="stylesheet" type="text/css" href="assets/styles.css">
        </head>
        <body>
            <div class="container">
                <h1>Informe de Predicción de Ventas</h1>
                <h2>1. Información del Dataset</h2>
                {info_html}
                <h2>2. División del Dataset</h2>
                {division_html}
                <h2>3. Métricas de Modelos de Predicción</h2>
                {metricas_html}
                <h2>4. Mejor Modelo</h2>
                {mejor_modelo_html}
                <h2>5. Estadísticas Generales</h2>
                <div class="table-container">{self.describe.to_html(classes='table table-striped', na_rep='-')}</div>
                <h2>6. Gráficos Generados</h2>
                <h3>Tendencia de Ventas</h3> {self.graficos.get("tendencia")}
                <h3>Distribución de Ventas</h3> {self.graficos.get("distribucion")}
                <h3>Mapa de Calor de Correlación</h3> {self.graficos.get("correlacion")}
                <h3>Boxplot de Ventas</h3> {self.graficos.get("boxplot")}
                <h3>Ventas por País</h3> {self.graficos.get("ventas_pais")}
                <h3>Distribución de Precios (Violín)</h3> {self.graficos.get("violin_precios")}
                <h3>Distribución de Cantidades (Violín)</h3> {self.graficos.get("violin_cantidades")}
                <h3>Comparación de Predicciones</h3> {self.graficos.get("comparacion_modelos")}
                <h3>Ventas por Día de la Semana</h3> {self.graficos.get("ventas_dia")}
                <h3>Tendencia de Ventas: Entrenamiento, Validación y Test</h3> {self.graficos.get("tendencia_train_test")}
                <h3>Treemap de Ventas por País</h3> {self.graficos.get("treemap_ventas")}
                <h3>Flujo de Ventas por País y Cliente</h3> {self.graficos.get("sankey_ventas")}
                <h3>Comparación de Modelos</h3> {self.graficos.get("radar_modelos")}
                <h3>Relación entre Cantidad y Ventas</h3> {self.graficos.get("scatter_ventas")}
                <h3>Scatter 3D de Ventas</h3>
                {self.graficos.get("scatter_3d_ventas", "<p>No disponible</p>")}

                <h3>Modelo Predictivo</h3>
                {self.graficos.get("scatter_modelo_predictivo", "<p>No disponible</p>")}

                <h3>Mapa de calor - Transformación logarítmica</h3>
                {self.graficos.get("heatmap_transformacion_logaritmica", "<p>No disponible</p>")}

                <h3>Distribución Variable Clave</h3>
                {self.graficos.get("histograma_variable_clave", "<p>No disponible</p>")}
                <h3>Evolución de Ventas</h3> {self.graficos.get("animacion_ventas")}
                <h3>Mapa de Ventas por País</h3> {self.graficos.get("mapa_ventas")}
                <h3>Histograma de Compras por Cliente</h3> {self.graficos.get("hist_compras_cliente", "<p>❌ No disponible</p>")}
            </div>
        </body>
        </html>
        """
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print("\n✅ Informe generado correctamente:", output_file)

# -----------------------------------------
# 7.  Generador de informe de Comparación
# -----------------------------------------
class GeneradorInformeComparacion:
    """Genera un informe comparativo entre modelos entrenados con datos escalados y no escalados,
    incluyendo una tabla de métricas y gráficos interactivos para comparar la mejora en las métricas."""
    def __init__(self, resultados_escalado, resultados_no_escalado, predicciones_escalado, predicciones_no_escalado):
        self.resultados_escalado = resultados_escalado
        self.resultados_no_escalado = resultados_no_escalado
        self.predicciones_escalado = predicciones_escalado
        self.predicciones_no_escalado = predicciones_no_escalado
        self.graficos = {}

    def guardar_grafico(self, fig):
        """Convierte un gráfico de Matplotlib en una imagen en base64 para incluirlo en el HTML."""
        try:
            from io import BytesIO
            import base64
            import matplotlib.pyplot as plt
            buffer = BytesIO()
            fig.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            imagen_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()
            plt.close(fig)
            return f'<img src="data:image/png;base64,{imagen_base64}" style="width:100%; border-radius: 10px; box-shadow: 2px 2px 10px gray;">'
        except Exception as e:
            print("❌ Error al guardar gráfico:", e)
            return "<p>❌ Error al generar gráfico</p>"

    def guardar_grafico_plotly(self, fig):
        """Convierte un gráfico de Plotly en HTML interactivo."""
        try:
            return fig.to_html(full_html=False, include_plotlyjs="cdn")
        except Exception as e:
            print("❌ Error al guardar gráfico interactivo:", e)
            return "<p>❌ Error al generar gráfico interactivo</p>"
    
    def generar_comparacion_metricas(self):
        """Genera una tabla comparativa de métricas entre modelos escalados y no escalados."""
        try:
            modelos = sorted(set(self.resultados_escalado.keys()).union(set(self.resultados_no_escalado.keys())))
            metricas_html = """
            <div class="info-box">
                <h3>Comparación de Modelos: Escalado vs No Escalado</h3>
                <table>
                    <tr>
                        <th>Modelo</th>
                        <th>MAE (Escalado)</th>
                        <th>RMSE (Escalado)</th>
                        <th>MAE (No Escalado)</th>
                        <th>RMSE (No Escalado)</th>
                    </tr>
            """

            for modelo in modelos:
                try:
                    mae_escalado = self.resultados_escalado.get(modelo, {}).get("MAE_Test", "-")
                    rmse_escalado = self.resultados_escalado.get(modelo, {}).get("RMSE_Test", "-")
                    mae_no_escalado = self.resultados_no_escalado.get(modelo, {}).get("MAE_Test", "-")
                    rmse_no_escalado = self.resultados_no_escalado.get(modelo, {}).get("RMSE_Test", "-")

                    # Depuración: imprime valores y tipos
                    print(f"Depuración {modelo}:")
                    print(f"  mae_escalado={mae_escalado}, type={type(mae_escalado)}")
                    print(f"  rmse_escalado={rmse_escalado}, type={type(rmse_escalado)}")
                    print(f"  mae_no_escalado={mae_no_escalado}, type={type(mae_no_escalado)}")
                    print(f"  rmse_no_escalado={rmse_no_escalado}, type={type(rmse_no_escalado)}")

                    # Función auxiliar para convertir a string con 2 decimales o '-'
                    def convertir_a_str_2dec(valor):
                        if valor == "-":
                            return "-"
                        try:
                            # Conviértelo a float (acepta int, np.float32, np.float64, str con número, etc.)
                            valor_float = float(valor)
                            return f"{valor_float:.2f}"
                        except:
                            return "-"

                    # Aplica la función a cada métrica
                    mae_escalado_str = convertir_a_str_2dec(mae_escalado)
                    rmse_escalado_str = convertir_a_str_2dec(rmse_escalado)
                    mae_no_escalado_str = convertir_a_str_2dec(mae_no_escalado)
                    rmse_no_escalado_str = convertir_a_str_2dec(rmse_no_escalado)

                    metricas_html += f"""
                    <tr>
                        <td>{modelo}</td>
                        <td>{mae_escalado_str}</td>
                        <td>{rmse_escalado_str}</td>
                        <td>{mae_no_escalado_str}</td>
                        <td>{rmse_no_escalado_str}</td>
                    </tr>
                    """
                except Exception as inner_e:
                    print(f"Error procesando el modelo {modelo}: {inner_e}")

            metricas_html += "</table></div>"
            return metricas_html

        except Exception as e:
            print(f"❌ Error en generar_comparacion_metricas: {e}")
            return "<p>Error al generar la tabla comparativa de métricas.</p>"

    def calcular_mejora(self, metrica_no_escalada, metrica_escalada):
        try:
            mae_ns = float(metrica_no_escalada)
            mae_s = float(metrica_escalada)
        except Exception as e:
            print("Advertencia: Métricas inválidas:", metrica_no_escalada, metrica_escalada)
            return None
        if mae_ns == 0:
            return 0
        mejora = ((mae_ns - mae_s) / mae_ns) * 100
        return mejora

    def generar_grafico_comparacion(self):
        """
        Genera un gráfico interactivo que muestra el porcentaje de mejora en MAE y RMSE
        al usar datos escalados frente a no escalados para cada modelo.
        """
        try:
            import plotly.graph_objects as go
            modelos = sorted(set(self.resultados_escalado.keys()).union(set(self.resultados_no_escalado.keys())))
            improvement_mae = []
            improvement_rmse = []
            modelos_grafico = []  # Para los modelos con datos válidos

            for m in modelos:
                try:
                    mae_ns = self.resultados_no_escalado.get(m, {}).get("MAE_Test")
                    mae_s = self.resultados_escalado.get(m, {}).get("MAE_Test")
                    rmse_ns = self.resultados_no_escalado.get(m, {}).get("RMSE_Test")
                    rmse_s = self.resultados_escalado.get(m, {}).get("RMSE_Test")
                    
                    # Imprimir para depuración
                    print(f"Modelo {m}: MAE sin escalado = {mae_ns}, MAE escalado = {mae_s}")
                    print(f"Modelo {m}: RMSE sin escalado = {rmse_ns}, RMSE escalado = {rmse_s}")

                    mae_improve = self.calcular_mejora(mae_ns, mae_s)
                    rmse_improve = self.calcular_mejora(rmse_ns, rmse_s)

                    if mae_improve is not None and rmse_improve is not None:
                        improvement_mae.append(mae_improve)
                        improvement_rmse.append(rmse_improve)
                        modelos_grafico.append(m)
                    else:
                        print(f"Advertencia: No se puede calcular la mejora para el modelo '{m}'. Datos faltantes o inválidos.")
                except Exception as inner_e:
                    print(f"Error procesando el modelo {m}: {inner_e}")
            if modelos_grafico:
                fig = go.Figure(data=[
                    go.Bar(name='Mejora MAE (%)', x=modelos_grafico, y=improvement_mae),
                    go.Bar(name='Mejora RMSE (%)', x=modelos_grafico, y=improvement_rmse)
                ])
                fig.update_layout(
                    barmode='group',
                    title='Mejora en Métricas (No Escalado vs Escalado)',
                    xaxis_title='Modelo',
                    yaxis_title='Mejora (%)'
                )
                return fig
            else:
                print("Advertencia: No hay datos suficientes para generar el gráfico de comparación.")
                return None
        except Exception as e:
            print(f"❌ Error en generar_grafico_comparacion: {e}")
            return None

    def generar_html(self, output_file):
        """Genera el informe HTML de comparación que incluye la tabla de métricas y el gráfico interactivo."""
        comparacion_metricas = self.generar_comparacion_metricas()
        grafico_plotly = self.generar_grafico_comparacion()  # Obtener el objeto de gráfico Plotly

        if grafico_plotly:
            grafico_comparacion = self.guardar_grafico_plotly(grafico_plotly)  # Convertir a HTML
        else:
            grafico_comparacion = "<p>No hay gráfico disponible debido a la falta de datos.</p>"  # Mensaje si no hay gráfico

        html_content = f"""
        <html>
        <head>
            <title>Informe de Comparación de Modelos</title>
            <meta charset="UTF-8">
            <link rel="stylesheet" type="text/css" href="assets/styles.css">           
        </head>
        <body>
            <div class="container">
                <h1>Informe de Comparación de Modelos</h1>
                <p>Este informe compara el rendimiento de modelos de Machine Learning con datos escalados y sin escalar.
                El objetivo es determinar si el escalado de datos mejora la precisión del modelo, medida a través de las métricas MAE (Error Absoluto Medio) y RMSE (Raíz del Error Cuadrático Medio).</p>
                {comparacion_metricas}
                <h2>Gráfico Comparativo de Métricas</h2>
                <p>Este gráfico muestra la mejora porcentual en las métricas MAE y RMSE al utilizar datos escalados en comparación con datos sin escalar.
                Un valor positivo indica una mejora al usar datos escalados, mientras que un valor negativo indica lo contrario.
                La ausencia de una barra para un modelo indica que no hay datos disponibles para esa comparación.</p>
                {grafico_comparacion}
            </div>
        </body>
        </html>
        """
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print("\n✅ Informe de comparación generado correctamente:", output_file)

# ----------------------------------------------------
# 8 Función para generar informe de Pandas Profiling
# ----------------------------------------------------
def generar_informe_pandas_profiling(df, output_file="informe_completo_dataset.html"):
    print("\n📊 Generando informe completo con Pandas Profiling...")
    try:
        for col in df.select_dtypes(include=['int32', 'int64', 'float32']).columns:
            df[col] = df[col].astype('float64')
        profile = ProfileReport(df, title="📊 Análisis Completo del Dataset", explorative=True)
        profile.to_file(output_file)
        print(f"✅ Informe generado correctamente: {output_file}")
    except Exception as e:
        print(f"❌ Error al generar el informe de Pandas Profiling: {e}")

# ---------------------------------------------------
# 9. Función para iniciar el dashboard interactivo
# ---------------------------------------------------
def iniciar_dashboard(df):
    app = Dash(__name__)
    app.layout = html.Div([
        html.H1("📊 Dashboard de Ventas en Tiempo Real"),
        dcc.Dropdown(
            id="dropdown_pais",
            options=[{"label": pais, "value": pais} for pais in df["Country"].unique()],
            value=df["Country"].unique()[0],
            multi=False
        ),
        dcc.Graph(id="grafico_ventas")
    ])
    @app.callback(
        Output("grafico_ventas", "figure"),
        Input("dropdown_pais", "value")
    )
    def actualizar_grafico(pais_seleccionado):
        df_filtrado = df[df["Country"] == pais_seleccionado].copy()
        # Si "TotalVentas" no existe, crearla
        if "TotalVentas" not in df_filtrado.columns:
            df_filtrado["TotalVentas"] = df_filtrado["Quantity"].astype(float) * df_filtrado["UnitPrice"].astype(float)
        fig = px.line(df_filtrado, x="InvoiceDate", y="TotalVentas",
                    title=f"📈 Ventas en {pais_seleccionado}",
                    labels={"TotalVentas": "Total de Ventas", "InvoiceDate": "Fecha"})
        return fig
    threading.Thread(target=app.run_server, kwargs={"debug": True, "use_reloader": False}).start()
    
# ----------------------------------------------
# 10. Ejecución Principal
# ----------------------------------------------
def main():
    file_path = "sample_data/data.csv"  # O "./sample_data/data.csv" si lo prefieres
    print("\n📥 Cargando datos...")
    cargador = CargadorDatos(file_path)
    df_original = cargador.cargar_datos()
    if df_original is None or df_original.empty:
        print("❌ Error: No se pudo cargar el dataset o está vacío. Verifica el archivo CSV.")
        return
    # Agregas la columna TotalVentas en el df_original
    if "Quantity" in df_original.columns and "UnitPrice" in df_original.columns:
        df_original["TotalVentas"] = df_original["Quantity"] * df_original["UnitPrice"]
    else:
        print("⚠️ Advertencia: No se pudo crear 'TotalVentas' en df_original porque faltan Quantity o UnitPrice.")

    info_text, head, describe = cargador.explorar_datos()    
    print("\n🛠️ Limpiando y transformando datos...")
    # Limpieza de datos para ambos dataframes
    procesador_no_escalado = ProcesadorDatos(df_original.copy(), df_original.copy())
    df_limpio_no_escalado = procesador_no_escalado.limpiar_datos(mantener_devoluciones=True, imputar_nulos=True)
    if df_limpio_no_escalado is None or df_limpio_no_escalado.empty:
        print("❌ Error: La limpieza de datos resultó en un dataset vacío (no escalado).")
        return
    stats_limpieza = procesador_no_escalado.stats_limpieza
    
    # Guardar el DataFrame limpio en un CSV para compartir con el menú (regresión)
    try:
        df_limpio_no_escalado.to_csv("df_limpio.csv", index=False)
        print("✅ df_limpio.csv generado correctamente.")
    except Exception as e:
        print(f"❌ Error al guardar df_limpio.csv: {e}")

    procesador_escalado = ProcesadorDatos(df_original.copy(), df_original.copy())
    df_limpio_escalado = procesador_escalado.limpiar_datos(mantener_devoluciones=True, imputar_nulos=True)
    if df_limpio_escalado is None or df_limpio_escalado.empty:
        print("❌ Error: La limpieza de datos resultó en un dataset vacío (escalado).")
        return
    df_escalado = procesador_escalado.escalar_datos(metodo="StandardScaler")
    print("✅ Transformación de datos completada.")

    # División de datos en entrenamiento, validación y prueba
    print("\n📉 Dividiendo datos en entrenamiento, validación y prueba (no escalado)...")
    divisor_no_escalado = DivisorDatos(df_limpio_no_escalado.copy())
    train_no_escalado, validation_no_escalado, test_no_escalado = divisor_no_escalado.dividir_datos()

    print("\n📉 Dividiendo datos en entrenamiento, validación y prueba (escalado)...")
    divisor_escalado = DivisorDatos(df_limpio_escalado.copy())
    train_escalado, validation_escalado, test_escalado = divisor_escalado.dividir_datos()

    # Entrenamiento de modelos
    print("\n📊 Entrenando modelos (no escalado)...")
    entrenador_no_escalado = EntrenadorModelos(train_no_escalado.copy(), validation_no_escalado.copy(), test_no_escalado.copy())
    predicciones_no_escalado, mejor_modelo_no_escalado, mejor_resultado_no_escalado = entrenador_no_escalado.entrenar_todos()
    resultados_no_escalado = entrenador_no_escalado.results

    print("\n📊 Entrenando modelos (escalado)...")
    entrenador_escalado = EntrenadorModelos(train_escalado.copy(), validation_escalado.copy(), test_escalado.copy())
    predicciones_escalado, mejor_modelo_escalado, mejor_resultado_escalado = entrenador_escalado.entrenar_todos()
    resultados_escalado = entrenador_escalado.results

    # Preparar los dataframes para el informe
    train_data_no_escalado = train_no_escalado.reset_index()
    validation_data_no_escalado = validation_no_escalado.reset_index()
    test_data_no_escalado = test_no_escalado.reset_index()

    train_data_escalado = train_escalado.reset_index()
    validation_data_escalado = validation_escalado.reset_index()
    test_data_escalado = test_escalado.reset_index()

    print("\n📑 Generando informes...")
    try:
        generar_informe_pandas_profiling(df_limpio_no_escalado.copy(), "informe_completo_dataset.html")
    except ImportError:
        print("⚠️ Advertencia: `ydata_profiling` no está instalado.")

    # Generar informe EDA
    generador_ventas = GeneradorInformeEDA(
        df_original.copy(),
        df_limpio_no_escalado.copy(),
        info_text, head, describe, stats_limpieza,
        divisor_no_escalado.filas_train,
        divisor_no_escalado.filas_validation,
        divisor_no_escalado.filas_test
    )
    generador_ventas.generar_html("informe_analisis_datos_EDA.html")

    # Generar informes de predicción de ventas (no escalado)
    generador_no_escalado = GeneradorInforme(
        df_original.copy(),
        df_limpio_no_escalado.copy(),
        info_text,
        describe,
        resultados_no_escalado,
        predicciones_no_escalado,
        divisor_no_escalado.filas_train,
        divisor_no_escalado.filas_validation,
        divisor_no_escalado.filas_test,
        mejor_modelo_no_escalado,
        mejor_resultado_no_escalado,
        train_data_no_escalado,
        validation_data_no_escalado,
        test_data_no_escalado
    )
    generador_no_escalado.generar_html("informe_prediccion_ventas_no_escalado.html")

    # Generar informes de predicción de ventas (escalado)
    generador_escalado = GeneradorInforme(
        df_original.copy(),
        df_limpio_escalado.copy(),
        info_text,
        describe,
        resultados_escalado,
        predicciones_escalado,
        divisor_escalado.filas_train,
        divisor_escalado.filas_validation,
        divisor_escalado.filas_test,
        mejor_modelo_escalado,
        mejor_resultado_escalado,
        train_data_escalado,
        validation_data_escalado,
        test_data_escalado
    )
    generador_escalado.generar_html("informe_prediccion_ventas_escalado.html")

    # Generar informe de comparación
    resultados_combinados_escalado = entrenador_escalado.results
    resultados_combinados_no_escalado = entrenador_no_escalado.results
    print("Claves en resultados_no_escalado:", resultados_no_escalado.keys())
    print("Claves en resultados_escalado:", resultados_escalado.keys())
    print("Contenido resultados_no_escalado:", resultados_no_escalado)
    print("Contenido resultados_escalado:", resultados_escalado)
    predicciones_combinadas_escalado = predicciones_escalado
    predicciones_combinadas_no_escalado = predicciones_no_escalado
    
    generador_comparacion = GeneradorInformeComparacion(
        resultados_combinados_escalado,
        resultados_combinados_no_escalado,
        predicciones_combinadas_escalado,
        predicciones_combinadas_no_escalado
    )
    generador_comparacion.generar_html("informe_comparacion_modelos.html")

    return df_original


# --------------------------
# 11. Ejecución Principal
# --------------------------
if __name__ == "__main__":
    df_original = main()
    if df_original is not None:
        print("\n🌍 Iniciando Dashboard Interactivo...")
        print("\n📊 Para ver los informes generados, abre los archivos HTML en tu navegador.")
        print("\n🌍 Para ver el dashboard, abre http://127.0.0.1:8050 en tu navegador.")
        print("\n🛑 Para detener la ejecución, presiona Ctrl+C.")
        iniciar_dashboard(df_original)
