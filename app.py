import joblib
import pandas as pd
import dash
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import os

# Cargar el modelo entrenado
modelo = joblib.load("student_performance.joblib")

# Cargar el dataset limpio con variable final_score
df = pd.read_csv("student_performance_with_final_score.csv")

# Columnas de entrada (sin la variable objetivo)
columnas_entrada = [col for col in df.columns if col != "final_score"]
X = df[columnas_entrada]
y = df["final_score"]

# Inicializar app
app = Dash(__name__)
app.title = "Predicción de Rendimiento Estudiantil"

# Layout
app.layout = html.Div([
    html.H1("Predicción del Rendimiento Estudiantil", style={'textAlign': 'center'}),

    html.H2("Visualización del Dataset"),
    dcc.Graph(
        figure=px.histogram(df, x="age", color="final_score", barmode="group",
                            title="Distribución de edad según desempeño final")
    ),

    html.H2("Formulario para Predicción Individual"),
    html.Div(
        children=[
            html.Div([
                html.Label(f"{col}:"),
                dcc.Input(
                    id=f"input-{col}",
                    type="number" if pd.api.types.is_numeric_dtype(df[col]) else "text",
                    value=0,
                    debounce=True
                )
            ]) for col in columnas_entrada
        ],
        style={'columnCount': 2}
    ),

    html.Br(),
    html.Button("Predecir", id="btn-predict", n_clicks=0),
    html.Div(id="output-prediction", style={"marginTop": 20, "fontWeight": "bold"}),

    html.H2("Matriz de Confusión del Modelo"),
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
    return f"Predicción del modelo: {str(pred).upper()}"

# Callback de matriz de confusión
@app.callback(
    Output("conf-matrix", "figure"),
    Input("btn-predict", "n_clicks")
)
def actualizar_matriz(n):
    y_pred = modelo.predict(X)
    etiquetas = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=etiquetas)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=etiquetas,
        y=etiquetas,
        colorscale="Blues"
    ))
    fig.update_layout(title="Matriz de Confusión", xaxis_title="Predicción", yaxis_title="Valor Real")
    return fig

# Ejecutar la aplicación
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))  # Esto es lo que faltaba
    app.run(debug=False, host="0.0.0.0", port=port)
