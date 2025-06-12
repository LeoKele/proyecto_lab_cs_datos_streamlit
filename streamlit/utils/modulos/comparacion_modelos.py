import plotly.express as px
import streamlit as st
import pandas as pd
import os
from utils.colores import PALETA


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



def mostrar_comparacion_optuna_vs_gridsearch():
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

    # Supóniendo que ya está el archivo pickle con las métricas de GridSearch
    ruta_base = os.path.dirname(__file__)
    ruta_pickle = os.path.join(ruta_base,"modelos_entrenados", "metricas_svm_gridsearch_final.pkl")
    metricas_grid = pd.read_pickle(ruta_pickle)

    # Métricas Optuna (de las capturas)
    metricas_optuna = {
        'Balanced Accuracy': 0.7586,
        'F1 Score': 0.6328,
        'Recall': 0.7787,
        'Precision': 0.5330,
        'PR AUC': 0.6428
    }

    # Métricas GridSearch 
    if isinstance(metricas_grid, pd.DataFrame):
        metricas_grid = metricas_grid.iloc[0].to_dict()

    df_comp = pd.DataFrame(
        [metricas_grid, metricas_optuna],
        index=['SVM GridSearch', 'SVM Optuna']
    )

    st.dataframe(df_comp)

    st.markdown("Visualización comparativa:")

    df_plot = df_comp.T.reset_index().rename(columns={'index': 'Métrica'})
    df_plot = df_plot.melt(id_vars='Métrica', var_name='Método', value_name='Valor')

    fig = px.bar(
        df_plot,
        x='Métrica',
        y='Valor',
        color='Método',
        barmode='group',
        title='Comparación de métricas: SVM GridSearch vs SVM Optuna',
        template='plotly_white',
        height=400,
        color_discrete_sequence=PALETA
    )
    fig.update_layout(
        yaxis_title='Valor',
        xaxis_title='Métrica',
        legend_title='Método'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("Como se observa, Optuna no aportó mejoras significativas y su tiempo de entrenamiento fue mayor. Por eso, optamos por GridSearchCV para el modelo final.")