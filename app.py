# app.py
import pandas as pd
import joblib
import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Si es un archivo .pkl:
import pickle

with open("modelo_entrenado.pkl", "rb") as f:
    modelo = pickle.load(f)

# Cargar dataset
df = pd.read_csv("student_performance.csv")

# Lista de columnas predictoras
columnas_entrada = [col for col in df.columns if col != "pass"]  # Ajusta según tu variable objetivo
X = df[columnas_entrada]
y = df["pass"]

# Inicializar Dash
app = dash.Dash(__name__)
app.title = "POC: Predicción de Rendimiento Estudiantil"

# Layout del dashboard
app.layout = html.Div([
    html.H1("Prueba de Concepto: Clasificación del Rendimiento Estudiantil", style={'textAlign': 'center'}),
    
    html.H2("Visualización del Dataset"),
    dcc.Graph(figure=px.histogram(df, x="age", color="pass", barmode="group", title="Distribución de edad según resultado")),

    html.H2("Formulario para predicción individual"),
    html.Div([
        html.Label(f"{col}:"),
        dcc.Input(id=f"input-{col}", type="number" if df[col].dtype != "object" else "text", value=0, debounce=True)
        for col in columnas_entrada
    ], style={'columnCount': 2}),
    
    html.Br(),
    html.Button("Predecir", id="btn-predict", n_clicks=0),
    html.Div(id="output-prediction", style={"marginTop": 20, "fontWeight": "bold"}),

    html.H2("Matriz de Confusión (modelo entrenado)"),
    dcc.Graph(id="conf-matrix")
])

# Callback de predicción
@app.callback(
    Output("output-prediction", "children"),
    Input("btn-predict", "n_clicks"),
    [Input(f"input-{col}", "value") for col in columnas_entrada]
)
def hacer_prediccion(n_clicks, *valores):
    if n_clicks == 0:
        return ""
    entrada = pd.DataFrame([valores], columns=columnas_entrada)
    pred = modelo.predict(entrada)[0]
    return f"Predicción del modelo: {'Aprobado' if pred == 1 else 'Reprobado'}"

# Callback para matriz de confusión
@app.callback(
    Output("conf-matrix", "figure"),
    Input("btn-predict", "n_clicks")  # solo refresca cuando se hace clic
)
def actualizar_matriz(n):
    from sklearn.metrics import confusion_matrix
    y_pred = modelo.predict(X)
    cm = confusion_matrix(y, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Reprobado", "Aprobado"],
        y=["Reprobado", "Aprobado"],
        colorscale="Blues"
    ))
    fig.update_layout(title="Matriz de Confusión")
    return fig

# Ejecutar app
if __name__ == "__main__":
    app.run_server(debug=True)
