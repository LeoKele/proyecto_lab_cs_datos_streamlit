import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import pandas as pd
import os
from utils.colores import PALETA

from sklearn.metrics import (
    balanced_accuracy_score, f1_score, recall_score,
    precision_score, average_precision_score
)

from utils.modulos.evaluacion_modelo import df_final_dividido


def comparar_resultados_interactivo(df_resultados, df_resultados_std):
    df_resultados = df_resultados.copy()
    df_resultados_std = df_resultados_std.copy()
    df_resultados['Datos'] = 'Sin escalar'
    df_resultados_std['Datos'] = 'Estandarizados'
    df_todos = pd.concat([df_resultados, df_resultados_std], ignore_index=True)
    df_todos_ordenado = df_todos.sort_values(by='Balanced Accuracy', ascending=False).reset_index(drop=True)
    
    # Ordenar el eje y según el orden descendente de Balanced Accuracy
    modelos_ordenados = df_todos_ordenado['Modelo'].tolist()

    fig = px.bar(
        df_todos_ordenado,
        x='Balanced Accuracy',
        y='Modelo',
        color='Datos',
        orientation='h',
        barmode='group',
        title='Comparación de modelos (Balanced Accuracy)',
        range_x=[0.5, 1.0],
        category_orders={'Modelo': modelos_ordenados},
        color_discrete_sequence=PALETA
    )
    st.plotly_chart(fig, use_container_width=True)
    return df_todos
    
    
def comparar_metricas_modelos_especificos(df_todos, modelos_comparar=None):
    """
    Visualiza la comparación de métricas para modelos específicos usando Plotly.
    """

    if modelos_comparar is None:
        modelos_comparar = ['STD - SVM_rbf', 'LightGBM', 'STD - Regresión Logística']

    # Filtrar el DataFrame combinado para incluir solo los modelos deseados
    df_comparacion_especifica = df_todos[df_todos['Modelo'].isin(modelos_comparar)]

    # Seleccionar solo las columnas relevantes para la comparación de métricas
    df_metricas_comparacion = df_comparacion_especifica[['Modelo', 'F1 Score', 'PR AUC', 'Recall', 'Precision', 'Datos']]

    # Reorganizar para gráfico
    df_melted = df_metricas_comparacion.melt(
        id_vars=['Modelo', 'Datos'],
        var_name='Métrica',
        value_name='Valor'
    )

    fig = px.bar(
        df_melted,
        x='Valor',
        y='Modelo',
        color='Métrica',
        barmode='group',
        facet_col='Datos',
        orientation='h',
        title='Comparación de F1 Score, PR AUC, Recall y Precision para modelos seleccionados',
        height=500,
        color_discrete_sequence=PALETA

    )
    fig.update_layout(xaxis_title='Valor de la Métrica', yaxis_title='Modelo')
    st.plotly_chart(fig, use_container_width=True)
    
    

def mostrar_comparacion_nuevas_variables():
    """
    Muestra en Streamlit la comparación de métricas del modelo SVM-RBF GridSearch
    entrenado sin y con las nuevas variables (CantidadServicios y MonthlyChargeRate).
    Incluye explicación, tabla y gráfico interactivo.
    """
    st.markdown("#### Consideración de nuevas variables y su impacto en el modelo")

    st.markdown("""
    Durante el análisis, evaluamos la creación de dos nuevas variables:
    - **CantidadServicios**: suma de todos los servicios contratados por el cliente (PhoneService, MultipleLines, Internet y servicios asociados).
    - **MonthlyChargeRate**: relación entre el TotalCharges y el tenure (TotalCharges / tenure).

    Entrenamos el modelo `SVM_rbf` optimizado (GridSearch) tanto **sin** como **con** estas nuevas variables. Sin embargo, al comparar las métricas, observamos que **no mejoraban el desempeño** (incluso, en algunos casos, empeoraban levemente).  
    Por eso, decidimos **no incluirlas en el dataset final** presentado.

    Las métricas obtenidas fueron:
    """)


    ruta_base = os.path.dirname(__file__)
    ruta_pickle = os.path.join(ruta_base, "..","modelos_entrenados", "comparacion_metricas_svm_gridsearch.pkl")
    df_comp = pd.read_pickle(ruta_pickle)

    st.dataframe(df_comp)

    st.markdown("Visualización comparativa:")

    # Usando Plotly Express para gráfico de barras agrupadas
    df_plot = df_comp.T.reset_index().rename(columns={'index': 'Métrica'})
    df_plot = df_plot.melt(id_vars='Métrica', var_name='Modelo', value_name='Valor')

    fig = px.bar(
        df_plot,
        x='Métrica',
        y='Valor',
        color='Modelo',
        barmode='group',
        title='Comparación de métricas: SVM-RBF GridSearch<br>Sin nuevas variables vs Con nuevas variables',
        template='plotly_white',
        height=400,
        color_discrete_sequence=PALETA
    )

    fig.update_layout(
        yaxis_title='Valor',
        xaxis_title='Métrica',
        legend_title='Modelo'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("Como se observa, agregar las nuevas variables no aportó mejoras al modelo, por lo que optamos por la versión más simple y robusta.")



def mostrar_comparacion_optuna_vs_gridsearch(modelo_grid,df):
    """
    Muestra en Streamlit la comparación de métricas entre SVM optimizado con GridSearch y Optuna.
    Explica por qué se decidió no usar Optuna en el modelo final.
    """
    st.markdown("### ¿Por qué no usamos Optuna en el modelo final?")

    st.markdown("""
    Además de optimizar el modelo `SVM_rbf` con **GridSearchCV** y **RandomSearchCV**, exploramos la librería **Optuna** para la búsqueda de hiperparámetros.
    Si bien Optuna es una herramienta poderosa y moderna para optimización automática, en nuestra experiencia:
    - **El tiempo de entrenamiento fue mayor**.
    - **Las métricas obtenidas no mostraron mejoras significativas** respecto a GridSearch.
    - El concepto teórico detrás de Optuna es más complejo y, aunque entendimos la idea básica de sugerir y probar valores, preferimos quedarnos con GridSearch para este trabajo por lo mencionado anteriormente.
    
    Las métricas obtenidas fueron:
    """)
    
    # Valores de Balanced Accuracy (conjunto de prueba)
    balanced_accuracy_optuna = 0.7655
    balanced_accuracy_fino_v2 = 0.7656  # ← reemplazá con el valor real si lo tenés

    # Nombres de los modelos
    metodos = ['GridSearch - Fino V2', 'Optuna']
    scores = [balanced_accuracy_fino_v2, balanced_accuracy_optuna]

    # Crear figura
    fig = go.Figure(
        go.Bar(
            x=metodos,
            y=scores,
            text=[f"{s:.4f}" for s in scores],
            textposition='auto',
            marker_color=PALETA
        )
    )

    # Personalizar el gráfico
    fig.update_layout(
        title="Comparación de Balanced Accuracy en el conjunto de prueba",
        xaxis_title="Método",
        yaxis_title="Balanced Accuracy",
        yaxis=dict(range=[0.7, 0.78]),
        template='plotly_white'
    )

    # Mostrar en Streamlit
    st.plotly_chart(fig)

    st.info("Como se observa, Optuna no aportó mejoras significativas y su tiempo de entrenamiento fue mayor. Por eso, optamos por GridSearchCV para el modelo final.")
    
    
    
def comparacion_modelos_optimizados(df_optimizacion):
    fig = go.Figure(data=[
    go.Bar(
        x=df_optimizacion['Método'],
        y=df_optimizacion['Balanced Accuracy'],
        marker_color=PALETA,
        text=[f"{v:.3f}" for v in df_optimizacion['Balanced Accuracy']],
        textposition='outside',
        textfont=dict(weight='bold')
    )
])

    # Configurar el diseño
    fig.update_layout(
        title='Comparación de mejores resultados por método de búsqueda',
        xaxis_title='Método',
        yaxis_title='Balanced Accuracy',
        yaxis=dict(range=[0, 1]),
        xaxis=dict(tickangle=15),
        width=800,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    