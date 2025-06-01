import plotly.express as px
import streamlit as st
import pandas as pd

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
        category_orders={'Modelo': modelos_ordenados}
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
        height=500
    )
    fig.update_layout(xaxis_title='Valor de la Métrica', yaxis_title='Modelo')
    st.plotly_chart(fig, use_container_width=True)
    
    
    
