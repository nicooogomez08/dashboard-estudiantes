import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Cargar el modelo correctamente
modelo = joblib.load('student_performance_model.pkl')


# Cargar dataset
df = pd.read_csv("student_performance.csv")

# Columnas predictoras
columnas_entrada = [col for col in df.columns if col != "pass"]
X = df[columnas_entrada]
y = df["pass"]

# Inicializar app
app = dash.Dash(__name__)
app.title = "POC: Predicción de Rendimiento Estudiantil"

# Layout
app.layout = html.Div([
    html.H1("Prueba de Concepto: Clasificación del Rendimiento Estudiantil", style={'textAlign': 'center'}),
    
    html.H2("Visualización del Dataset"),
    dcc.Graph(figure=px.histogram(df, x="age", color="pass", barmode="group", title="Distribución de edad según resultado")),

    html.H2("Formulario para predicción individual"),
    html.Div(
        children=[
            html.Div([
                html.Label(f"{col}:"),
                dcc.Input(
                    id=f"input-{col}",
                    type="number" if df[col].dtype != "object" else "text",
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

    html.H2("Matriz de Confusión (modelo entrenado)"),
    dcc.Graph(id="conf-matrix")
])

# Callback predicción
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

# Callback matriz de confusión
@app.callback(
    Output("conf-matrix", "figure"),
    Input("btn-predict", "n_clicks")
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
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)
