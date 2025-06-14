import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score
import plotly.figure_factory as ff
from sklearn.inspection import permutation_importance
import multiprocessing


import pickle
import streamlit as st
import pandas as pd
import os
from utils.colores import PALETA
from pathlib import Path


from utils.modulos.preparacion_datos import limpiar_datos, transformacion_datos,estandarizar_datos,dividir_datos, manejar_columnas_redundantes

# Cargamos el modelo final entrenado
def cargar_modelo_optimizado():
    ruta_archivo = os.path.join(os.path.dirname(__file__), '..', 'modelos_entrenados', 'grid_fino_v2_entrenado.pkl')
    ruta_archivo = os.path.normpath(ruta_archivo)  # Limpia la ruta para cualquier sistema operativo

    print(f"Cargando resultados desde: {ruta_archivo}")
    with open(ruta_archivo, "rb") as f:
        return pickle.load(f)



#Dividimos y dejamos todo listo para usar
def df_final_dividido(df):
    df_mod = limpiar_datos(df)
    df_mod = manejar_columnas_redundantes(df_mod) 
    df_mod = transformacion_datos(df_mod);
    
    X_train, X_test, y_train, y_test = dividir_datos(df_mod)
    X_train_scaled, X_test_scaled = estandarizar_datos(X_train,X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


    


def obtener_metricas_y_predicciones(df, modelo):
    # Preparamos los datos
    X_train, X_test, y_train, y_test = df_final_dividido(df)

    # Predicciones
    predicciones = modelo.predict(X_test)

    # Métricas
    mat_confusion = confusion_matrix(y_test, predicciones)
    accuracy = accuracy_score(y_test, predicciones)

    return mat_confusion, accuracy, y_test, predicciones


def mostrar_matriz_confusion_plotly(matriz, accuracy):
    z = matriz
    x = ["No Churn", "Churn"]
    y = ["No Churn", "Churn"]

    fig = ff.create_annotated_heatmap(
        z=z, 
        x=x, 
        y=y, 
        colorscale='Greens', 
        showscale=True
    )

    fig.update_layout(
        title_text=f"<b>Matriz de Confusión</b><br>Accuracy: {accuracy:.2%}",
        xaxis_title="Predicción",
        yaxis_title="Valor Real"
    )

    st.plotly_chart(fig, use_container_width=True)


def mostrar_matriz_confusion(df, modelo):
    mat_conf, acc, y_test, y_pred = obtener_metricas_y_predicciones(df, modelo)

    st.write("### Matriz de Confusión")
    st.write(f"Accuracy en test: **{acc:.2%}**")
    
    mostrar_matriz_confusion_plotly(mat_conf, acc)

def graficar_importancia_plotly(df_importancia):
    # Ordenar ascendente para que la barra horizontal quede similar
    df = df_importancia.sort_values('importances_mean', ascending=True)
    
    fig = go.Figure()

    # Agrego barras horizontales (para el error)
    fig.add_trace(go.Bar(
        x=df['importances_mean'],
        y=df['feature'],
        error_x=dict(
            type='data',
            array=df['importances_std'],
            visible=True,
            color=PALETA[1]
        ),
        orientation='h',
        marker=dict(
            color=PALETA[0],
            opacity=0.6
        ),
        name='Importancia media'
    ))

    # Agrego los puntos rojos como scatter (los "diamantes")
    fig.add_trace(go.Scatter(
        x=df['importances_mean'],
        y=df['feature'],
        mode='markers',
        marker=dict(
            symbol='diamond',
            size=10,
            color='red',
            opacity=0.8
        ),
        name='Importancia (puntos)'
    ))

    fig.update_layout(
        title='Importancia de los predictores (train)',
        xaxis_title='Incremento del error tras la permutación',
        yaxis_title='Características',
        yaxis=dict(autorange='reversed'),  # Para que las características queden de arriba hacia abajo igual que barh
        height=400,
        margin=dict(l=150, r=20, t=50, b=50),
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)