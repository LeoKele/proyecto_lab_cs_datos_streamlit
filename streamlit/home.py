import streamlit as st
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from utils.preparacion_datos import cargar_datos,limpiar_datos, codificar_datos_inicial, dividir_datos, estandarizar_datos, transformacion_datos
from utils.comparacion_modelos import comparar_resultados_interactivo, comparar_metricas_modelos_especificos
# from utils.modelos import evaluar_modelos_iniciales, evaluar_modelos_iniciales_estandarizados

# from utils.codigos_mostrados import (code_librerias, code_limpieza, code_limpieza2, code_evaluar_modelos, code_codificacion_rapida,
#                                         code_division_datos, code_evaluacion_modelos, code_evaluar_modelos_estandarizados
#                                      )
from utils.codigos_mostrados import *
from utils.colores import PALETA

# Configuración inicial de la página
st.set_page_config(page_title="Análisis de Abandono de Clientes", layout="wide")

# Título y presentación
st.markdown("<h1 style='text-align: center;color: #66BB6A;'>🎓 Trabajo Integrador Final - Abandono de Clientes</h1>", unsafe_allow_html=True)
st.markdown("""
👥 Integrantes:
-
-
- Evers Juan Segundo
- Kelechian Leonardo
""")


#? Cargar datos una vez
data = cargar_datos()

#! ==============================================================
#! ==============================================================

st.markdown("---")

# Introducción
st.header("✨ Introducción")
st.markdown("""
[Contenido introductorio del proyecto...]
""")

#! ==============================================================
#! ==============================================================
st.markdown("---")

# Cargado de librerías
st.header("📦 Cargado de librerias")

with st.expander("Ver código de librerías", expanded=True):
    st.code(code_librerias, language='python')

st.subheader("Datos originales")
st.dataframe(data.head())

#! ==============================================================
#! ==============================================================
st.markdown("---")


# Preparación de Datos
st.header("⚙️ Preparación de los Datos")
st.subheader("Limpieza de datos")
st.markdown("""
Realizamos las siguientes transformaciones:
- Conversión de TotalCharges a numérico
- Tratamiento de valores nulos con imputación por mediana
""")

# Mostrar dataset antes de la limpieza
df_mod_app = data.copy()
df_mod_app['TotalCharges'] = pd.to_numeric(df_mod_app['TotalCharges'], errors='coerce')

with st.expander("Ver código de limpieza", expanded=True):
    st.code(code_limpieza, language='python')
    st.code(f"Cantidad de filas con valor nulo en 'Total Charges': {df_mod_app['TotalCharges'].isnull().sum()}")


st.markdown("Como vemos, ahora existen 11 filas con valor nulo en esta variable. Si bien no son muchos valores, podríamos eliminarlos y es posible que el resultado final no varíe tanto, pero, decidimos que con el propósito de mantener todos los datos originales, se realizará una **Imputación por la mediana.**")

# Gráfico interactivo con Plotly del boxplot de TotalCharges
fig = px.box(df_mod_app, x='TotalCharges', title='Boxplot interactivo de TotalCharges',color_discrete_sequence=PALETA)
st.plotly_chart(fig, use_container_width=True)

st.markdown("Este boxplot nos ayuda a terminar de decidir si era una buena opción el uso de la mediana para la imputación y vemos rapidamente que hay un sesgo positivo, lo que hace que la media no sea tan representativa y pierda robustez como medida de resumen.")

with st.expander("Ver código de imputación", expanded=True):
    st.code(code_limpieza2, language='python')
    mediana_TotalCharges = df_mod_app['TotalCharges'].median()
    df_mod_app['TotalCharges'] = df_mod_app['TotalCharges'].fillna(mediana_TotalCharges)
    st.code(f"Cantidad de filas con valor nulo en 'Total Charges': {df_mod_app['TotalCharges'].isnull().sum()}")

st.markdown("Ahora si, la variable `TotalCharges` ya se encuentra en el tipo de dato correcto y con sus valores nulos tratados.")
st.markdown("Luego, el resto de las variables del dataset no requieren ningún cambio de tipo de dato, limpieza o tratamiento de outliers.")

#! ==============================================================
#! ==============================================================
st.markdown("---")


# Modelo Base
st.header("🔎 Búsqueda de Modelo Base")
st.markdown("""
Para abordar la tarea de modelado de manera eficiente, iniciamos con una fase de 'búsqueda de modelo base'. 
El objetivo principal aquí es probar rápidamente diferentes tipos de modelos (como regresión logística, modelos basados en árboles, etc.) con configuraciones estándar. 
Esto nos ayuda a comprender qué familias de modelos tienen un rendimiento inicial aceptable sin invertir tiempo excesivo en la optimización individual de cada uno. 
Los modelos con mejor desempeño en esta etapa serán los que consideraremos para una optimización más profunda.
""")
st.markdown("""
Sabiendo que hay modelos que necesitan estandarización, dividiremos esta busqueda en dos fases:
- Comparación rápida sin procesar -> Donde solo confiaremos en las métricas de los modelos de árboles
- Comparación con estandarización -> Usaremos la misma función pero solo con los modelos sensibles a las escalas y el data set previamente estandarizado

Por ultimo, compararemos resultados.
""")            

st.subheader("🔍 Fase 1 - Comparación rápida sin procesar")

st.markdown("""
Creamos la función `evaluar_modelos()` que se utilizará para entrenar y evaluar varios modelos de clasificación de forma rápida. 
Como habiamos mencionado, el objetivo es obtener una primera idea de qué modelos podrían funcionar mejor para el problema dado, 
utilizando configuraciones predeterminadas o ajustes iniciales básicos.            
""")
with st.expander("Ver código de la función `evaluar_modelos()`", expanded=True):
    st.code(code_evaluar_modelos, language='python')


st.markdown("""
            Si bien la idea es probar los datos sin modificar tanto, los modelos a entrenar necesitan que estos se encuentren en un formato númerico.

Por esto, realizamos una codificación rápida correspondiente a cada variable, dado que tenemos muchas categóricas.
            """)
with st.expander("Ver código de codificación rápida", expanded=True):
    st.code(code_codificacion_rapida, language='python')


st.markdown("Finalmente, dividimos los datos y evaluamos los modelos iniciales.")
with st.expander("Ver código de división de datos y entrenamiento inicial", expanded=True):
    st.code(code_division_datos, language='python')
    st.code(code_evaluacion_modelos, language='python')

# Cargar resultados de modelos iniciales
@st.cache_data
def cargar_resultados():
    ruta_base = os.path.dirname(__file__)  # Carpeta actual del archivo .py
    ruta_archivo = os.path.join(ruta_base,"utils","modelos_entrenados", "resultados_iniciales.pkl")
    print(f"Cargando resultados desde: {ruta_archivo}")
    with open(ruta_archivo, "rb") as f:
        return pickle.load(f)

df_resultados = cargar_resultados()
st.dataframe(df_resultados)


#! ==============================================================

st.subheader("🔍 Fase 2 - Comparación con estandarización")


st.markdown("Utilizamos el mismo split de datos de antes pero estandarizamos las variables numéricas.")
with st.expander("Ver código de escalado", expanded=True):
    st.code(code_escalar_inicial, language='python')

st.markdown("")
st.markdown("Ajustamos la funcion `evaluar_modelos()` pero para entrenar solo los modelos que necesitan los valores estandarizados.")
with st.expander("Ver código de `evaluar_modelos_estandarizados()` y su evaluación", expanded=True):
    st.code(code_evaluar_modelos_estandarizados, language='python')
    st.code(code_evaluacion_modelos_estandarizados, language='python')




#Cargar resultados de modelos estandarizados
@st.cache_data
def cargar_resultados():
    ruta_base = os.path.dirname(__file__)  # Carpeta actual del archivo .py
    ruta_archivo = os.path.join(ruta_base, "utils","modelos_entrenados", "resultados_iniciales_estandarizados.pkl")
    print(f"Cargando resultados desde: {ruta_archivo}")
    with open(ruta_archivo, "rb") as f:
        return pickle.load(f)


df_resultados_std = cargar_resultados()
st.dataframe(df_resultados_std)


st.markdown("---")
st.markdown("#### 🟨 Comparación de resultados")

df_todos = comparar_resultados_interactivo(df_resultados, df_resultados_std)


st.markdown("#### Conclusión")
st.markdown("""
* ✅ El modelo con mejor rendimiento general fue el `SVM_rbf` estandarizado, alcanzando la mayor Balanced Accuracy entre todos los modelos comparados.
* 🌳 El modelo `lightGBM` fue el que mejor accuracy tuvo de todos los árboles, teniendo casi el mismo valor que el SVM. Similar, solo por apenas debajo de este, se encuentra el de `Regresión Logística`. Ambos tambien podrían ser considerados como modelo final.
* ⚠️ El `SVM_linear` sin estandarizar puede no ser confiable, esto se debe a que luego de ser escalado, su accuracy es mucho menor, lo que nos hace pensar que en ese primer entrenamiento se vió sesgado por la escala de alguna variable.
""")

st.markdown("""
Entonces, los modelos de
- `SVM_rbf`,
- `lightGBM`,
- `Regresión Logística`

son nuestros candidatos para ser usados en el proyecto.

Para terminar de decidirnos por uno, evaluaremos las demás métricas calculadas, con tal de ver cual es el que mejor se ajusta a nuestro problema.
            """)

comparar_metricas_modelos_especificos(df_todos)



st.markdown("""
#### Decisión final del modelo

Luego de evaluar distintos modelos, **decidimos seleccionar el `SVM_rbf`** como el más adecuado para nuestro objetivo principal: **minimizar la pérdida de clientes reales (churners)**. Es decir, priorizamos un **recall alto**, ya que preferimos contactar a clientes con riesgo de irse antes que dejar pasar casos reales de abandono.

Entre los modelos evaluados:

- La **regresión logística estandarizada** obtuvo el **mayor recall (0.764)** —proporción de churners correctamente identificados—, pero con una **precisión más baja (0.527)** —de todos los casos predichos como churn, cuántos realmente lo son—. Su **F1 Score** —promedio armónico entre precisión y recall— fue de **0.624**.

- El modelo **`SVM_rbf` estandarizado** logró un **recall muy similar (0.752)**, pero con una **mejor precisión (0.545)** y un **F1 Score superior (0.632)**. Esto indica que detecta casi los mismos casos de churn que la regresión logística, pero con **menos falsos positivos**, lo cual es clave si las acciones de retención tienen un costo.

- Por otro lado, **`LightGBM`** presentó la **mayor precisión (0.569)**, pero con un **recall más bajo (0.717)**. Esto implica que, si bien acierta más en sus predicciones positivas, **deja pasar más casos reales de churn**, lo que no se alinea con nuestra prioridad. Su F1 Score fue de **0.635**.

En resumen, el modelo **`SVM_rbf` logra un mejor equilibrio**, manteniendo un recall alto (detectamos la mayoría de los que se van), sin sacrificar tanto la precisión (no sobreactuamos en exceso), y superando a la regresión logística en F1 Score. Por estos motivos, consideramos que es la mejor elección para nuestro caso.
            """)


#! ==============================================================
st.markdown("---")
#! ==============================================================


st.header("🛠️ Preparación de los datos para el modelo elegido") 
st.markdown("Habiendo elegido el modelo `SVM_rbf`, es necesario realizar una preparación de datos más detallada. Esto implica un manejo cuidadoso de la redundancia en las variables categóricas, su posterior codificación numérica y la estandarización de las características para asegurar el mejor rendimiento del modelo.")
st.subheader("Manejo de Redundancia en variables categóricas")