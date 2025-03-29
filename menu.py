import os
import sys
import threading
import webbrowser
import subprocess
import multiprocessing
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import flask
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Intentar importar ttkbootstrap; si no existe, instalarlo automáticamente.
try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import *
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ttkbootstrap"])
    import ttkbootstrap as tb
    from ttkbootstrap.constants import *

# Importamos los módulos reales
import codigo    # Módulo de Regresión y EDA (archivo codigo.py)
import cluster   # Módulo de Clustering (archivo cluster.py)

# Variables globales para saber si se han ejecutado los módulos
executed_regression = False
executed_clustering = False

# Diccionario de informes de Regresión (rutas fijas)
regression_reports = {
    "Informe Completo (Pandas Profiling)": "informe_completo_dataset.html",
    "Informe EDA": "informe_analisis_datos_EDA.html",
    "Informe Predicción No Escalado": "informe_prediccion_ventas_no_escalado.html",
    "Informe Predicción Escalado": "informe_prediccion_ventas_escalado.html",
    "Informe Comparación de Modelos": "informe_comparacion_modelos.html"
}

# Para Clustering, se buscarán todos los archivos HTML que contengan la palabra "clustering" en su nombre.
def get_clustering_reports():
    files = [f for f in os.listdir('.') 
            if f.lower().endswith('.html') and 'clustering' in f.lower()]
    return files

# ----------------------------
# Funciones para ejecutar los módulos reales
# ----------------------------
def run_regression():
    global executed_regression
    try:
        threading.Thread(target=codigo.main).start()
        executed_regression = True
        messagebox.showinfo("Regresión", "El módulo de Regresión se ha ejecutado correctamente.")
    except Exception as e:
        messagebox.showerror("Error en Regresión", f"Ocurrió un error: {e}")

def run_clustering():
    global executed_clustering
    try:
        threading.Thread(target=cluster.main).start()
        executed_clustering = True
        messagebox.showinfo("Clustering", "El módulo de Clustering se ha ejecutado correctamente.")
    except Exception as e:
        messagebox.showerror("Error en Clustering", f"Ocurrió un error: {e}")

def run_regression_and_clustering():
    def run_both():
        global executed_regression, executed_clustering
        try:
            # Ejecuta la regresión y espera a que finalice
            codigo.main()
            executed_regression = True  # Marcar regresión como ejecutada
            # Una vez terminada la regresión, ejecuta clustering
            cluster.main()
            executed_clustering = True  # Marcar clustering como ejecutado
            messagebox.showinfo("Regresión y Clustering", "Se han ejecutado ambos módulos de forma secuencial.")
        except Exception as e:
            messagebox.showerror("Error en Regresión y Clustering", f"Ocurrió un error: {e}")
    threading.Thread(target=run_both).start()

# ----------------------------
# Funciones para visualizar informes (submenú)
# ----------------------------
def view_reports(report_list, module_name):
    submenu = tk.Toplevel()
    submenu.title(f"Ver Informes de {module_name}")
    submenu.geometry("450x400")

    lbl = tk.Label(submenu, text=f"Archivos HTML encontrados para {module_name}:", font=("Arial", 12))
    lbl.pack(pady=10)

    # Contenedor con scrollbar
    container = tk.Frame(submenu)
    container.pack(fill="both", expand=True)

    canvas = tk.Canvas(container)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    if not report_list:
        lbl_none = tk.Label(scrollable_frame, text="No se encontraron informes.", font=("Arial", 10))
        lbl_none.pack(pady=10)
    else:
        for report in report_list:
            def open_report(path=report):
                webbrowser.open(f"file://{os.path.abspath(path)}")
            btn = tk.Button(scrollable_frame, text=report, width=40, command=open_report)
            btn.pack(pady=5)

    btn_close = tk.Button(submenu, text="Volver al menú principal", command=submenu.destroy)
    btn_close.pack(pady=10)

def view_regression_reports():
    if not executed_regression:
        messagebox.showwarning("Advertencia", "El módulo de Regresión no se ha ejecutado.\nEjecute primero el módulo de Regresión.")
        return
    view_reports(list(regression_reports.values()), "Regresión")

def view_clustering_reports():
    if not executed_clustering:
        messagebox.showwarning("Advertencia", "El módulo de Clustering no se ha ejecutado.\nEjecute primero el módulo de Clustering.")
        return
    reports = get_clustering_reports()
    view_reports(reports, "Clustering")

def view_all_reports():
    if not (executed_regression and executed_clustering):
        messagebox.showwarning("Advertencia", "No se han ejecutado ambos módulos.\nEjecute primero los módulos correspondientes.")
        return
    all_reports = list(regression_reports.values()) + get_clustering_reports()
    view_reports(all_reports, "Regresión y Clustering")

# ----------------------------
# Funciones para compartir datos entre el menú y los módulos
# ----------------------------
def get_regression_data():
    path = "df_limpio.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        messagebox.showerror("Error", "No se encontró 'df_limpio.csv'. Ejecute primero el módulo de Regresión.")
        return None

def get_clustering_data():
    path = "df_clientes.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        messagebox.showerror("Error", "No se encontró 'df_clientes.csv'. Ejecute primero el módulo de Clustering.")
        return None

# ----------------------------
# Funciones globales para iniciar los dashboards
# ----------------------------
def run_dash_reg_global():
    df = get_regression_data()
    if df is None:
        return
    app = Dash(__name__)
    app.layout = html.Div([
        html.H1("Dashboard de Ventas", style={"textAlign": "center", "color": "#007BFF"}),
        dcc.Dropdown(
            id="dropdown_pais",
            options=[{"label": pais, "value": pais} for pais in df["Country"].unique()],
            value=df["Country"].unique()[0],
            multi=False,
            style={"width": "50%", "margin": "0 auto"}
        ),
        dcc.Graph(id="grafico_ventas"),
        html.Button("Cerrar Dashboard", id="btn-cerrar", 
                    style={"display": "block", "margin": "20px auto",
                        "backgroundColor": "#dc3545", "color": "white", "padding": "10px 20px",
                        "border": "none", "borderRadius": "5px"}),
        html.Div(id="dummy")
    ])

    @app.callback(
        Output("grafico_ventas", "figure"),
        [Input("dropdown_pais", "value")]
    )
    def actualizar_grafico(pais_seleccionado):
        df_filtrado = df[df["Country"] == pais_seleccionado].copy()
        if "TotalVentas" not in df_filtrado.columns:
            df_filtrado["TotalVentas"] = df_filtrado["Quantity"].astype(float) * df_filtrado["UnitPrice"].astype(float)
        fig = px.line(df_filtrado, x="InvoiceDate", y="TotalVentas",
                    title=f"Ventas en {pais_seleccionado}",
                    labels={"TotalVentas": "Total de Ventas", "InvoiceDate": "Fecha"})
        return fig

    @app.callback(
        Output("dummy", "children"),
        [Input("btn-cerrar", "n_clicks")]
    )
    def shutdown_dashboard(n_clicks):
        if n_clicks:
            import sys
            sys.exit()
        return ""
    app.run_server(debug=False, use_reloader=False, port=8051)

def run_dash_clust_global():
    df = get_clustering_data()
    if df is None:
        return
    scaler = StandardScaler()
    features = ["TicketPromedio", "TotalCompras", "TotalGastado", "Recency", "NumeroProductosDistintos"]
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    pca3 = PCA(n_components=3)
    pcs = pca3.fit_transform(df_scaled[features])
    df_pca3 = pd.DataFrame(pcs, columns=["PC1", "PC2", "PC3"], index=df.index)
    from sklearn.cluster import KMeans
    km_model = KMeans(n_clusters=4, random_state=42)
    df_pca3["Cluster"] = km_model.fit_predict(df_scaled[features])
    df_pca3["TotalGastado"] = df["TotalGastado"].values
    df_pca3["TicketPromedio"] = df["TicketPromedio"].values

    app = Dash(__name__)
    app.layout = html.Div([
        html.H1("Dashboard Interactivo de Clustering", style={"textAlign": "center", "color": "#28a745"}),
        dcc.Graph(id="scatter3d-graph"),
        html.Button("Cerrar Dashboard", id="btn-cerrar", 
                    style={"display": "block", "margin": "20px auto",
                        "backgroundColor": "#dc3545", "color": "white", "padding": "10px 20px",
                        "border": "none", "borderRadius": "5px"}),
        html.Div(id="dummy")
    ])

    @app.callback(
        Output("scatter3d-graph", "figure"),
        [Input("dummy", "children")]
    )
    def actualizar_scatter(dummy):
        fig = px.scatter_3d(
            df_pca3,
            x="PC1", y="PC2", z="PC3",
            color="Cluster",
            title="Scatterplot 3D de Clusters",
            hover_data=df_pca3.columns
        )
        return fig

    @app.callback(
        Output("dummy", "children"),
        [Input("btn-cerrar", "n_clicks")]
    )
    def shutdown_dashboard(n_clicks):
        if n_clicks:
            import sys
            sys.exit()
        return ""
    app.run_server(debug=False, use_reloader=False, port=8050)

def iniciar_dashboard_regresion():
    global dash_reg_process
    dash_reg_process = multiprocessing.Process(target=run_dash_reg_global)
    dash_reg_process.start()
    messagebox.showinfo("Dashboard de Regresión", "Dashboard iniciado en el puerto 8051")

def iniciar_dashboard_clustering():
    global dash_clust_process
    dash_clust_process = multiprocessing.Process(target=run_dash_clust_global)
    dash_clust_process.start()
    messagebox.showinfo("Dashboard de Clustering", "Dashboard iniciado en el puerto 8050")
    
# ----------------------------
# Función para salir del programa
# ----------------------------
def exit_program():
    if messagebox.askyesno("Confirmar", "¿Está seguro de que desea salir?"):
        # Ocultar la ventana principal para que no se vea junto a la de créditos
        root.withdraw()

        # Crear una ventana de créditos con ttkbootstrap
        credit_win = tb.Toplevel(root)
        credit_win.title("Créditos")
        credit_win.geometry("400x300")
        credit_win.config(bg="#f8f9fa")

        lbl_title = tb.Label(credit_win, text="Realizado por el Grupo 1",
                            font="Helvetica 16 bold", bootstyle="info")
        lbl_title.pack(pady=20)

        nombres = ["Ana Ortiz", "Alfredo Martínez", "Jorge Rodríguez", "Enrique Solís"]
        for nombre in nombres:
            lbl = tb.Label(credit_win, text=nombre,
                        font="Helvetica 14", bootstyle="secondary")
            lbl.pack(pady=5)

        def close_all():
            root.destroy()  # Cierra completamente la aplicación

        btn_close = tb.Button(credit_win, text="Cerrar", command=close_all,
                            bootstyle="danger", width=20)
        btn_close.pack(pady=20)

# -------------------------------------------------
# Bloque principal: creación del menú (Tkinter)
# -------------------------------------------------
if __name__ == '__main__':
    root = tb.Window(themename="morph")  # Una sola ventana principal
    root.title("Menú Regresión y Clustering")
    root.geometry("600x700")
    root.config(bg="#f8f9fa")

    # Ajustar el estilo de los botones
    root.style.configure("TButton", font="Helvetica 12 bold")

    lbl_title = tb.Label(root, text="Menú Regresión y Clustering", bootstyle="info", font="Helvetica 20 bold")
    lbl_title.pack(pady=20)

    frame_buttons = tb.Frame(root, padding=20)
    frame_buttons.pack(pady=10, fill="both", expand=True)

    btn_style = {"width": 30, "padding": 10}

    btn_regression = tb.Button(frame_buttons, text="Ejecutar Regresión", command=run_regression, bootstyle="primary", **btn_style)
    btn_clustering = tb.Button(frame_buttons, text="Ejecutar Clustering", command=run_clustering, bootstyle="primary", **btn_style)
    btn_reg_and_clust = tb.Button(frame_buttons, text="Ejecutar Regresión y Clustering", command=run_regression_and_clustering, bootstyle="primary", **btn_style)
    btn_view_reg = tb.Button(frame_buttons, text="Ver Informes de Regresión", command=view_regression_reports, bootstyle="secondary", **btn_style)
    btn_view_clust = tb.Button(frame_buttons, text="Ver Informes de Clustering", command=view_clustering_reports, bootstyle="secondary", **btn_style)
    btn_view_all = tb.Button(frame_buttons, text="Ver Todos los Informes", command=view_all_reports, bootstyle="secondary", **btn_style)
    btn_dash_reg = tb.Button(frame_buttons, text="Iniciar Dashboard Regresión", command=iniciar_dashboard_regresion, bootstyle="success", **btn_style)
    btn_dash_clust = tb.Button(frame_buttons, text="Iniciar Dashboard Clustering", command=iniciar_dashboard_clustering, bootstyle="success", **btn_style)
    btn_exit = tb.Button(frame_buttons, text="Salir", command=exit_program, bootstyle="danger", **btn_style)

    btn_regression.pack(pady=8)
    btn_clustering.pack(pady=8)
    btn_reg_and_clust.pack(pady=8)
    btn_view_reg.pack(pady=8)
    btn_view_clust.pack(pady=8)
    btn_view_all.pack(pady=8)
    btn_dash_reg.pack(pady=8)
    btn_dash_clust.pack(pady=8)
    btn_exit.pack(pady=15)

    # Pie del menú
    frame_footer = tb.Frame(root, padding=10)
    frame_footer.pack(side="bottom", fill="x")
    lbl_master = tb.Label(frame_footer, text="Máster FP Big Data e IA", font="Helvetica 14 bold", bootstyle="info")
    lbl_master.pack()
    lbl_group = tb.Label(frame_footer, text="Grupo 1", font="Helvetica 12", bootstyle="secondary")
    lbl_group.pack()

    root.mainloop()
