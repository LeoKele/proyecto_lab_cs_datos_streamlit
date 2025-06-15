import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
from scipy import stats

import streamlit as st

from utils.colores import PALETA

def grafico_proporcion_churn(df_eda):
    """
    Crea y muestra un gráfico de barras con la proporción de la variable Churn
    """
    
    # Calculamos la proporción de observaciones de cada clase
    prop_churn = df_eda['Churn'].value_counts(normalize=True)
    
    # Convertimos a DataFrame para Plotly
    df_plot = pd.DataFrame({
        'Churn': prop_churn.index,
        'Proporción': prop_churn.values
    })
    
    # Creamos el gráfico de barras con Plotly Express
    fig = px.bar(df_plot, 
                 x='Churn', 
                 y='Proporción',
                 title='Proporción de la variable respuesta (Churn)',
                 color='Churn',
                 color_discrete_sequence=PALETA,
                 labels={'Proporción': 'Proporción de observaciones'})
    
    # Personalizamos el diseño
    fig.update_layout(
        title={'text': 'Proporción de la variable respuesta (Churn)', 'font': {'size': 18}},
        xaxis={'title': {'text': 'Churn', 'font': {'size': 14}}},
        yaxis={'title': {'text': 'Proporción de observaciones', 'font': {'size': 14}}},
        showlegend=False,  # Ocultamos la leyenda ya que es redundante
        height=500,
        width=700
    )
    
    # Mostramos el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    
import math

def percentage_stacked_plot_plotly(df, columns_to_plot, super_title):
    
    """
    Crea y muestra graficos de barras stackeadas segun la columna indicada y con categoria de churn
    """
    
    
    
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot) / 2)
    
    # Crear subplots
    fig = make_subplots(
        rows=number_of_rows, 
        cols=number_of_columns,
        subplot_titles=[f'Proporción de observaciones en {col}' for col in columns_to_plot],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Loop para cada columna
    for index, column in enumerate(columns_to_plot):
        row = (index // number_of_columns) + 1
        col = (index % number_of_columns) + 1
        
        # Calcular el crosstab y las proporciones
        crosstab = pd.crosstab(df[column], df['Churn'])
        prop_by_independent = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        
        # Obtener las categorías únicas de la variable independiente
        categories = prop_by_independent.index.tolist()
        
        # Agregar barras para cada valor de Churn
        churn_values = prop_by_independent.columns.tolist()
        
        for i, churn_val in enumerate(churn_values):
            fig.add_trace(
                go.Bar(
                    name=f'Churn {churn_val}' if index == 0 else f'Churn {churn_val}',
                    x=categories,
                    y=prop_by_independent[churn_val],
                    marker_color=PALETA[i % len(PALETA)],
                    showlegend=(index == 0),  # Solo mostrar leyenda en el primer subplot
                    legendgroup=f'churn_{churn_val}',  # Agrupar leyendas
                ),
                row=row, col=col
            )
        
        # Configurar los ejes para este subplot específico
        fig.update_yaxes(title_text="Porcentaje (%)", row=row, col=col)
        fig.update_xaxes(title_text=column, row=row, col=col)
    
    # Configurar el layout general
    fig.update_layout(
        title={
            'text': super_title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22}
        },
        barmode='stack',
        height=400 * number_of_rows,
        width=1000,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            title="Churn"
        ),
        showlegend=True
    )
    
    # Hacer que todas las barras sumen 100%
    for i in range(len(columns_to_plot)):
        row = (i // number_of_columns) + 1
        col = (i % number_of_columns) + 1
        fig.update_yaxes(range=[0, 100], row=row, col=col)
    
    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    
def boxplot_plots_plotly(df, columns_to_plot, super_title):
    
    paleta = PALETA
    
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot) / 2)
    
    # Crear subplots
    fig = make_subplots(
        rows=number_of_rows, 
        cols=number_of_columns,
        subplot_titles=[f'Distribución de {col} por Churn' for col in columns_to_plot],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Loop para cada columna
    for index, column in enumerate(columns_to_plot):
        row = (index // number_of_columns) + 1
        col = (index % number_of_columns) + 1
        
        # Preparar datos para boxplot
        data_no_churn = df[df['Churn'] == 'No'][column].dropna()
        data_yes_churn = df[df['Churn'] == 'Yes'][column].dropna()
        
        # Agregar boxplot para Churn = No
        fig.add_trace(
            go.Box(
                y=data_no_churn,
                name='No',
                marker_color=paleta[0],
                showlegend=(index == 0),
                legendgroup='churn_no',
                boxpoints='outliers'
            ),
            row=row, col=col
        )
        
        # Agregar boxplot para Churn = Yes
        fig.add_trace(
            go.Box(
                y=data_yes_churn,
                name='Yes',
                marker_color=paleta[1],
                showlegend=(index == 0),
                legendgroup='churn_yes',
                boxpoints='outliers'
            ),
            row=row, col=col
        )
        
        # Configurar los ejes para este subplot específico
        fig.update_xaxes(title_text="Churn", row=row, col=col)
        fig.update_yaxes(title_text=column, row=row, col=col)
    
    # Configurar el layout general
    fig.update_layout(
        title={
            'text': super_title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22}
        },
        height=400 * number_of_rows,
        width=1000,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            title="Churn"
        ),
        showlegend=True
    )
    
    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    
def calculate_statistical_measures_streamlit(df, numeric_columns, group_by=None):
    
    def compute_stats(data_series):
        """Función auxiliar para calcular estadísticas de una serie"""
        clean_data = data_series.dropna()

        if len(clean_data) == 0:
            return pd.Series([np.nan] * 20, index=[
                'count', 'mean', 'median', 'mode', 'std', 'var', 'min', 'max',
                'range', 'q1', 'q3', 'iqr', 'skewness', 'kurtosis',
                'cv', 'mad', 'sem', 'p10', 'p90', 'missing_count'
            ])

        # Medidas de tendencia central
        mean_val = clean_data.mean()
        median_val = clean_data.median()
        try:
            mode_val = clean_data.mode().iloc[0] if not clean_data.mode().empty else np.nan
        except:
            mode_val = np.nan

        # Medidas de dispersión
        std_val = clean_data.std()
        var_val = clean_data.var()
        min_val = clean_data.min()
        max_val = clean_data.max()
        range_val = max_val - min_val

        # Cuartiles y medidas de posición
        q1 = clean_data.quantile(0.25)
        q3 = clean_data.quantile(0.75)
        iqr = q3 - q1
        p10 = clean_data.quantile(0.10)
        p90 = clean_data.quantile(0.90)

        # Medidas de forma
        skewness_val = clean_data.skew()
        kurtosis_val = clean_data.kurtosis()

        # Otras medidas
        cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
        mad = np.mean(np.abs(clean_data - mean_val))
        sem = stats.sem(clean_data)

        # Conteos
        count = len(clean_data)
        missing_count = len(data_series) - count

        return pd.Series([
            count, mean_val, median_val, mode_val, std_val, var_val,
            min_val, max_val, range_val, q1, q3, iqr, skewness_val,
            kurtosis_val, cv, mad, sem, p10, p90, missing_count
        ], index=[
            'count', 'mean', 'median', 'mode', 'std', 'var', 'min', 'max',
            'range', 'q1', 'q3', 'iqr', 'skewness', 'kurtosis',
            'cv', 'mad', 'sem', 'p10', 'p90', 'missing_count'
        ])

    # Definir descripciones de las métricas
    metric_descriptions = {
        'count': 'Cantidad de observaciones',
        'mean': 'Media aritmética',
        'median': 'Mediana (Q2)',
        'mode': 'Moda',
        'std': 'Desviación estándar',
        'var': 'Varianza',
        'min': 'Valor mínimo',
        'max': 'Valor máximo',
        'range': 'Rango (max - min)',
        'q1': 'Primer cuartil (Q1)',
        'q3': 'Tercer cuartil (Q3)',
        'iqr': 'Rango intercuartílico (Q3-Q1)',
        'skewness': 'Asimetría',
        'kurtosis': 'Curtosis',
        'cv': 'Coeficiente de variación (%)',
        'mad': 'Desviación absoluta media',
        'sem': 'Error estándar de la media',
        'p10': 'Percentil 10',
        'p90': 'Percentil 90',
        'missing_count': 'Valores faltantes'
    }

    if group_by is None:
        # Análisis general sin agrupamiento
        stats_df = pd.DataFrame()
        for col in numeric_columns:
            stats_df[col] = compute_stats(df[col])
        
        # Mostrar en Streamlit
        
        
        # Transponer para mejor visualización
        stats_display = stats_df.T
        
        # Renombrar índices con descripciones
        stats_display_renamed = stats_display.copy()
        stats_display_renamed.index = [metric_descriptions.get(col, col) for col in stats_display.index]
        
        st.dataframe(stats_display_renamed.round(4), use_container_width=True)
        
    else:
        # Análisis agrupado
        if group_by not in df.columns:
            st.error(f"La columna '{group_by}' no existe en el DataFrame")
            return

        groups = df[group_by].unique()
        
    
        # Crear tabs para cada grupo
        tabs = st.tabs([f"{group_by}: {group}" for group in groups])
        
        group_stats = {}
        for i, group in enumerate(groups):
            with tabs[i]:
                group_data = df[df[group_by] == group]
                stats_df = pd.DataFrame()

                for col in numeric_columns:
                    stats_df[col] = compute_stats(group_data[col])
                
                group_stats[group] = stats_df
                
                # Transponer y renombrar
                stats_display = stats_df.T
                stats_display_renamed = stats_display.copy()
                stats_display_renamed.index = [metric_descriptions.get(col, col) for col in stats_display.index]
                
                st.dataframe(stats_display_renamed.round(4), use_container_width=True)
        
        # Mostrar comparación si hay exactamente 2 grupos
        if len(groups) == 2:
            st.subheader("🔍 Comparación entre Grupos")
            group1, group2 = groups
            
            comparison_df = pd.DataFrame()
            for col in numeric_columns:
                diff_series = group_stats[group1][col] - group_stats[group2][col]
                comparison_df[col] = diff_series
            
            st.write(f"**Diferencias: {group1} - {group2}**")
            st.write("Valores positivos indican que el primer grupo tiene mayor valor en esa métrica.")
            
            comparison_display = comparison_df.T
            comparison_display_renamed = comparison_display.copy()
            comparison_display_renamed.index = [metric_descriptions.get(col, col) for col in comparison_display.index]
            
            st.dataframe(comparison_display_renamed.round(4), use_container_width=True)
            
            

def generar_matriz_correlacion(df):
    # Correlación entre tenure, MonthlyCharges y TotalCharges
    correlation_matrix = df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr(method='pearson')
    
    # Crear heatmap con Plotly
    fig = px.imshow(correlation_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    color_continuous_scale='Greens',
                    title='Matriz de Correlación: Tenure, MonthlyCharges, TotalCharges')
    
    # Mostrar el gráfico
    st.plotly_chart(fig)
    
    
    

def generar_scatterplots_optimizado(df, max_points=5000):
    """
    Versión optimizada con sampling y caching para mejor performance
    """
    # Si el dataset es muy grande, hacer sampling
    if len(df) > max_points:
        df_sample = df.sample(n=max_points, random_state=42)
    else:
        df_sample = df.copy()
    
    # Crear subplot con configuración optimizada
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Tenure vs TotalCharges', 'MonthlyCharges vs TotalCharges'),
        horizontal_spacing=0.1
    )
    
    # Scatterplot 1: tenure vs TotalCharges
    fig.add_trace(
        go.Scattergl(  # Usar Scattergl para mejor performance
            x=df_sample['tenure'], 
            y=df_sample['TotalCharges'],
            mode='markers',
            name='Tenure vs TotalCharges',
            opacity=0.4,
            marker=dict(
                color=PALETA[0],
                size=4,  # Puntos más pequeños
                line=dict(width=0)  # Sin borde para mejor performance
            ),
            hovertemplate='<b>Tenure:</b> %{x}<br><b>Total Charges:</b> $%{y}<extra></extra>'
        ), 
        row=1, col=1
    )
    
    # Línea de tendencia optimizada
    x_trend = np.linspace(df_sample['tenure'].min(), df_sample['tenure'].max(), 50)
    z1 = np.polyfit(df_sample['tenure'], df_sample['TotalCharges'], 1)
    y_trend = np.polyval(z1, x_trend)
    
    fig.add_trace(
        go.Scatter(
            x=x_trend, 
            y=y_trend,
            mode='lines',
            name='Tendencia',
            line=dict(color=PALETA[5], width=2),
            hoverinfo='skip'
        ), 
        row=1, col=1
    )
    
    # Scatterplot 2: MonthlyCharges vs TotalCharges
    fig.add_trace(
        go.Scattergl(
            x=df_sample['MonthlyCharges'], 
            y=df_sample['TotalCharges'],
            mode='markers',
            name='MonthlyCharges vs TotalCharges',
            opacity=0.4,
            marker=dict(
                color=PALETA[0],
                size=4,
                line=dict(width=0)
            ),
            hovertemplate='<b>Monthly Charges:</b> $%{x}<br><b>Total Charges:</b> $%{y}<extra></extra>'
        ), 
        row=1, col=2
    )
    
    # Línea de tendencia 2 optimizada
    x_trend2 = np.linspace(df_sample['MonthlyCharges'].min(), df_sample['MonthlyCharges'].max(), 50)
    z2 = np.polyfit(df_sample['MonthlyCharges'], df_sample['TotalCharges'], 1)
    y_trend2 = np.polyval(z2, x_trend2)
    
    fig.add_trace(
        go.Scatter(
            x=x_trend2, 
            y=y_trend2,
            mode='lines',
            name='Tendencia',
            line=dict(color=PALETA[5], width=2),
            hoverinfo='skip'
        ), 
        row=1, col=2
    )
    
    # Layout optimizado
    fig.update_xaxes(title_text="Tenure (Meses)", row=1, col=1)
    fig.update_yaxes(title_text="Total Charges ($)", row=1, col=1)
    fig.update_xaxes(title_text="Monthly Charges ($)", row=1, col=2)
    fig.update_yaxes(title_text="Total Charges ($)", row=1, col=2)
    
    fig.update_layout(
        height=500, 
        showlegend=False,
        # Configuraciones para mejor performance
        hovermode='closest',
        dragmode='pan',
        modebar={'remove': ['select2d', 'lasso2d']}
    )
    
    st.plotly_chart(fig)