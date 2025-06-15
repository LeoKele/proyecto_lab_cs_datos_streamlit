import streamlit as st

# Configuración inicial de la página
st.set_page_config(page_title="Análisis de Abandono de Clientes", layout="centered")

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from utils.modulos.preparacion_datos import *
from utils.modulos.comparacion_modelos import *
from utils.modulos.eda import *
from utils.modulos.evaluacion_modelo import *

from utils.contenido.codigos_mostrados import *
from utils.contenido.textos_mostrados import *
from utils.colores import PALETA, colores_barras_binarias



# Título y presentación
st.markdown("<h1 style='text-align: center;'>🎓 Trabajo Integrador Final - Abandono de Clientes</h1>", unsafe_allow_html=True)
st.markdown("""
👥 Integrantes:
- Fernando Burgos
- Francisco Garcia
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
st.markdown(texto_introduccion)

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
st.header("💡 Identificar la problemática a resolver")
st.markdown(texto_problema)

#! ==============================================================
#! ==============================================================
st.markdown("---")


# Preparación de Datos
st.header("⚙️ Preparación de los Datos")
st.subheader("Limpieza de datos")
st.markdown("""
Antes de arrancar, tenemos que corroborar que los datos se encuentren limpios de forma tal que no afecten negativamente al modelo.

Luego de realizar un pequeño análisis univariado a las columnas de nuestro data set, encontramos que la columna TotalCharges era de tipo object cuando esta debería ser float. Entonces, la casteamos y encontramos también que escondía un par de valores nulos.
""")

# Mostrar dataset antes de la limpieza
df_mod_app = data.copy()
df_mod_app['TotalCharges'] = pd.to_numeric(df_mod_app['TotalCharges'], errors='coerce')

with st.expander("Ver código de limpieza", expanded=True):
    st.code(code_limpieza, language='python')
    st.code(f"Cantidad de filas con valor nulo en 'Total Charges': {df_mod_app['TotalCharges'].isnull().sum()}")


st.markdown("Como podemos observar, existen 11 filas con valor nulo en esta variable de un total de 7043 registros (aproximadamente 0.16% del dataset). Dado que este porcentaje es mínimo y no representa una pérdida significativa de información, decidimos eliminar directamente estas filas para mantener la integridad del análisis.")



with st.expander("Ver código de imputación", expanded=True):
    st.code(code_limpieza2, language='python')
    df_mod_app = df_mod_app.dropna(subset=['TotalCharges'])
    st.code(f"Cantidad de filas después de eliminar nulos: {len(df_mod_app)}\n"
            f"Cantidad de filas eliminadas: {len(data) - len(df_mod_app)}")


st.markdown("Ahora, la variable TotalCharges ya se encuentra en el tipo de dato correcto y sin valores nulos. El resto de las variables del dataset no requieren ningún cambio de tipo de dato, limpieza o tratamiento de outliers, por lo que podemos proceder con el análisis.")


#? ==============================================================
st.subheader("Limpieza de datos")
st.markdown("Para una correcta visualización de los datos tenemos que editar las etiquetas de la variable PaymentMethod.")
with st.expander("Ver código de limpieza de etiquetas", expanded=True):
    st.code(code_limpieza3, language='python')
    st.markdown("**Salida:**")
    st.write(f"Valores únicos en 'PaymentMethod' antes: `{df_mod_app['PaymentMethod'].unique()}`")
    
    st.markdown("")
    
    st.code(code_limpieza4, language='python')
    df_mod_app['PaymentMethod'] = df_mod_app['PaymentMethod'].str.replace(' (automatic)', '', regex=False)
    st.markdown("**Salida:**")
    st.write(f"Valores únicos en 'PaymentMethod' después de limpieza: `{df_mod_app['PaymentMethod'].unique()}`")

#! ==============================================================
#! ==============================================================
st.markdown("---")
st.header("📊 EDA")
st.markdown("""
En el análisis exploratorio de datos buscamos analizar las características principales del dataset a través de métodos de visualización y estadísticos de resúmen. 
El objetivo es lograr comprender cómo se comportan nuestros datos y descubrir patrones y posibles relaciones entre características y la tasa de churn.
            """)
df_eda_app = df_mod_app.copy()
grafico_proporcion_churn(df_eda_app)

st.subheader("Información demográfica")
st.markdown("A continuación analizaremos las variables de atributos demográficos (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), mostrando la proporción de `Churn` para cada categoría de cada atributo.")

# Para información demográfica
demographic_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
percentage_stacked_plot_plotly(df_eda_app, demographic_columns, 'Información Demográfica')

st.markdown("""
### Conclusiones:

• La proporción de `Churn` para adultos mayores es casi el doble que para aquellos que no pertenecen a esta categoría.

• El género no parece tener impacto en la predicción del `Churn`. Tanto hombres como mujeres tienen similar proporción de Churners.

• La proporción de Churners es mayor en clientes sin pareja.

• La proporción de Churners es mayor en clientes sin hijos.           
            
            """)

st.subheader("Información Sobre la Cuenta del Cliente - Variables Categóricas")
st.markdown("De la misma manera que hicimos con los atributos demográficos, evaluaremos la proporción de Churn para cada categoría de las características del cliente (`Contract`, `PaperlessBilling`, `PaymentMethod`).")

# Para información de cuenta
account_columns = ['Contract', 'PaperlessBilling', 'PaymentMethod']
percentage_stacked_plot_plotly(df_eda_app, account_columns, 'Información Sobre la Cuenta del Cliente')

st.markdown("""
### Conclusiones:

• Los clientes con contratos mensuales tienen mayor proporción de `Churn` que aquellos con contratos anuales o bianuales.

• En cuanto a los métodos de pago, aquellos clientes que optaron por un cheque electrónico son bastante más propensos a Churnear que los demás métodos de pago.

• Los clientes que eligieron una facturación electrónica son más propensos a Churnear.         
            """)

st.subheader("Información Sobre la Cuenta del Cliente - Variables Numéricas")
st.markdown("Los siguientes boxplots comparan la distribución de tenure (antigüedad como cliente), cargos mensuales y cargos totales entre clientes que se dieron de baja y los que permanecieron. Esto nos puede dar una idea de el comportamiento de facturación y permanencia que podrían estar relacionados con la decisión de abandonar el servicio.")
account_columns_numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
boxplot_plots_plotly(df_eda_app, account_columns_numeric, 'Información Sobre la Cuenta del Cliente')
st.markdown("""
### Conclusiones:

• En clientes que Churnean, hay mayor acumulación de registros con poca antiguedad de contrato.           

• En cuanto a los cargos mensuales, hay mayor acumulación de Churnerns en valores más altos.

• Aquellos clientes que Churnean, suelen tener cargos totales más bajos. Esto podría producirse debido a la relación encontrada con `tenure` (menos meses de antiguedad, menos cargos totales).        
            """)


st.header("Estadísticas Descriptivas Generales")
st.markdown("**Resumen completo de medidas estadísticas para variables numéricas**")
st.markdown("Análisis descriptivo que incluye medidas de tendencia central, dispersión, posición y forma para cada variable, permitiendo comprender la distribución y características de los datos.")
account_columns_numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
calculate_statistical_measures_streamlit(df_eda_app, account_columns_numeric)

st.subheader("Análisis de los Resultados: ")
st.markdown("""
`TENURE` (Antiguedad en meses):

- Distribución: Ligeramente asimétrica hacia la derecha (skewness: 0.24)

- Variabilidad: Muy alta (CV: 75.7%) - hay clientes muy nuevos y muy antiguos

- Curtosis negativa (-1.39): Distribución más plana que la normal, sin concentración en el centro



`MONTHLY CHARGES` (Cargos mensuales):

- Distribución: Ligeramente asimétrica hacia la izquierda (skewness: -0.22)

- Variabilidad: Moderada (CV: 46.4%)

- Interpretación: La mediana (USD 70.35) es mayor que la media (USD 64.80), sugiere concentración en valores altos

- Rango: De $18.25 a $118.75 - amplio espectro de planes



`TOTAL CHARGES` (Cargos totales):

- Distribución: Fuertemente asimétrica hacia la derecha (curtosis: 0.96)

- Variabilidad: Extremadamente alta (CV: 99.3%) - la más variable de las tres

- Interpretación: Media (USD 2,283) muy superior a la mediana (USD 1,397) - muchos valores bajos y algunos muy altos
          
            
            
            """)


st.header("Correlacion Lineal y No Lineal")
generar_matriz_correlacion(df_eda_app)
st.subheader("Conclusión: ")
st.markdown("""
- Podemos observar una relación fuerte ente `tenure` y `TotalCharges`.

- Podemos observar una relación moderada entre `MonthlyCharges` y `TotalCharges`.

Para avanzar sobre el análisis de estas relaciones vamos a graficarlas.           
            """)
generar_scatterplots_optimizado(df_eda_app)
st.subheader("Conclusión: ")
st.markdown("""
Vemos en el primer gráfico la relación entre `tenure` y `TotalCharges`. Podemos observar una relación lineal fuerte positiva. Esto tiene sentido, ya que a medida que aumenta la antiguedad de un cliente, sus gastos totales en el servicio van a haber aumentado dado el paso del tiempo.

Por otro lado, podemos observar una relación similar entre `MonthlyCharges` y `TotalCharges`. También tiene un sentido lógico, ya que aquellos clientes que tienen mayores gastos mensuales, a lo largo del tiempo, tendrán un aún mayor gasto total.         
            
            """)


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
Entonces, los modelos:
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

- La **`Regresión Logística` estandarizada** obtuvo el **mayor recall (0.764)** —proporción de churners correctamente identificados—, pero con una **precisión más baja (0.527)** —de todos los casos predichos como churn, cuántos realmente lo son—. Su **F1 Score** —promedio armónico entre precisión y recall— fue de **0.624**.

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
st.markdown("""
            
            """)

st.markdown("""
A la hora de utilizar un modelo basado en SVM, menos es más: un conjunto de variables bien elegidas, no redundantes y bien escaladas, permite encontrar un hiperplano más limpio, más generalizable y más eficiente computacionalmente. Tratar la redundancia mejora tanto el rendimiento como la robustez del modelo.
            
Se reconoció que un cliente no puede tener múltiples líneas si no tiene servicio telefónico. Por ello, se creó una nueva variable binaria HasMultipleLines que toma el valor 1 solo si el cliente tiene servicio telefónico y múltiples líneas. Luego se eliminó la variable original MultipleLines, eliminando así la redundancia.
            
Se observó que servicios como OnlineSecurity, StreamingTV o TechSupport solo pueden contratarse si el cliente tiene servicio de internet. Para resolver esta dependencia lógica, se crearon variables binarias (HasOnlineSecurity, HasStreamingTV, etc.) que indican si el cliente tiene el servicio y además tiene internet. Las variables originales fueron eliminadas, conservando únicamente la información relevante y no redundante.
            """)

with st.expander("Ver código de manejo de redundancia", expanded=True):
    st.code(code_manejo_redun_1, language='python')
    st.code(code_manejo_redun_2, language='python')


st.subheader("Transformación de datos")
st.markdown("""
Aplicamos binary encoding a las variables recién creadas y luego las transformamos de booleano a numérico.
""")
with st.expander("Ver código de transformación de datos", expanded=True):
    st.code(code_transformacion_1, language='python')
    st.code(code_transformacion_2, language='python')
    st.code(code_transformacion_3, language='python')

st.markdown("")

#? ===========

mostrar_comparacion_nuevas_variables()


    


#? ===========
st.subheader("DataFrame final")
with st.expander("Ver código de dataframe final", expanded=True):
    st.code(code_df_final, language='python')   

st.subheader("División de datos")
st.markdown("""
            Antes de aplicar cualquier transformación, debemos dividir los datos en training y testing sets para evitar un Data Leakage.
            """)
with st.expander("Ver código de división de datos", expanded=True):
    st.code(code_split, language='python')

st.subheader("Estandarización de datos")
st.markdown("...")
with st.expander("Ver código de estandarización", expanded=True):
    st.code(code_estandarizacion, language='python')

st.markdown("Finalmente, tenemos nuestro dataset listo para ser utilizado por el modelo `SVM_rbf`.")


#! ==============================================================
st.markdown("Se aplicó un proceso de escalado o estandarización sobre las variables numéricas `tenure`, `MonthlyCharges` y `TotalCharges`, utilizando la técnica de StandardScaler. Esta transformación consiste en restar la media y dividir por la desviación estándar de cada variable, lo que da como resultado nuevas variables con media 0 y desviación estándar 1.")
#! ==============================================================


st.header("🧠 Entrenamiento del modelo")
st.markdown("Ya estamos en condiciones de optimizar el modelo elegido con el propósito de mejorar aún más esas métricas iniciales obtenidas.")

st.markdown("Hacemos una función donde vamos a guardar los mejores hiperparámetros de cada entrenamiento para luego comparar todo junto.")
with st.expander("Ver código función para el DataFrame de resultados", expanded=True):
    st.code(code_resultados_busquedas, language='python')
    st.write("Creamos el DataFrame donde guardaremos estos resultados:")
    st.code("df_resultados_optimizacion = pd.DataFrame()",language='python')

st.subheader("GridSearchCV() - Hiperparámetros básicos")
st.markdown("Se realiza una búsqueda de hiperparámetros utilizando un grid básico con valores comúnmente utilizados. Este grid permite obtener una primera aproximación al rendimiento del modelo sin requerir mucho tiempo de cómputo. Sirve como punto de partida para identificar si existe una mejora del modelo inicial.")
with st.expander("Ver código función para el GridSearchCV Básico", expanded=True):
    st.code(code_gridSearchBasico,language='python')
    st.write("Agregamos los resultados al DF para luego comparar:")
    st.code("df_resultados_optimizacion = agregar_resultado_busqueda(df_resultados_optimizacion,grid_basico,'GridSearch - Basico')",language='python')

st.subheader("GridSearchCV() - Hiperparámetros Fino")
st.markdown("""
En base a los resultados obtenidos con el grid básico, se define un grid más fino con valores logarítmicamente espaciados para los hiperparámetros `C` y `gamma`. Esta estrategia permite explorar con mayor detalle combinaciones en una región más amplia, incrementando la probabilidad de encontrar un mejor conjunto de hiperparámetros.  
            """)
with st.expander("Ver código función para el GridSearchCV Fino", expanded=True):
    st.code(code_gridSearchFino,language='python')
    st.write("Agregamos los resultados al DF para luego comparar:")
    st.code("df_resultados_optimizacion = agregar_resultado_busqueda(df_resultados_optimizacion,grid_fino,'GridSearch - Fino')",language='python')


st.subheader("GridSearchCV() - Hiperparámetros Fino V2")
st.markdown("""
A partir de los mejores valores encontrados en la búsqueda anterior, se realiza un ajuste más localizado, centrado en un rango reducido alrededor del mejor `C` y `gamma`. Este enfoque permite afinar aún más el modelo, evaluando con mayor precisión pequeñas variaciones que podrían mejorar ligeramente el rendimiento. Básicamente, hacemos un "zoom" entorno al mejor valor encontrado por el entrenamiento anterior.
""")
with st.expander("Ver código función para el GridSearchCV Fino", expanded=True):
    st.code(code_gridSearchFinoV2,language='python')
    st.write("Agregamos los resultados al DF para luego comparar:")
    st.code("df_resultados_optimizacion = agregar_resultado_busqueda(df_resultados_optimizacion,grid_fino_v2,'GridSearch - Fino V2')",language='python')


st.subheader("RandomizedSearchCV()")
st.markdown("""
Se aplica una búsqueda aleatoria con distribuciones continuas. Esta técnica permite cubrir un espacio de búsqueda más amplio, con menor costo computacional que un grid exhaustivo. Si bien no garantiza encontrar el mejor valor absoluto, puede descubrir combinaciones efectivas que un grid regular no contempla.
""")
with st.expander("Ver código función para el RandomizedSearchCV()", expanded=True):
    st.code(code_randomizedSearch,language='python')
    st.write("Agregamos los resultados al DF para luego comparar:")
    st.code("df_resultados_optimizacion = agregar_resultado_busqueda(df_resultados_optimizacion,random_search,'Random Search')",language='python')

st.subheader("Comparación de las métricas obtenidas en cada entrenamiento")
st.markdown("En la siguiente tabla se resumen los resultados de distintas estrategias de búsqueda de hiperparámetros aplicadas al modelo SVM, evaluadas mediante la métrica de Balanced Accuracy, que resulta adecuada dado el desbalance de clases presente en el dataset:")

#Cargar resultados de modelos optimizados
@st.cache_data
def cargar_resultados_optimizacion():
    ruta_base = os.path.dirname(__file__)  # Carpeta actual del archivo .py
    ruta_archivo = os.path.join(ruta_base, "utils","modelos_entrenados", "df_optimizacion.pkl")
    print(f"Cargando resultados desde: {ruta_archivo}")
    with open(ruta_archivo, "rb") as f:
        return pickle.load(f)

df_optimizacion = cargar_resultados_optimizacion()
st.dataframe(df_optimizacion)
comparacion_modelos_optimizados(df_optimizacion)

st.markdown("""Como se puede observar, el mejor rendimiento se obtuvo con la versión ajustada del Grid Search fino (V2), alcanzando una balanced accuracy del 76.56%. Esto demuestra que realizar una búsqueda más específica en torno a los valores óptimos mejora el desempeño del modelo.

También se observa que Random Search presentó un rendimiento inferior al de las variantes de Grid Search, lo cual refuerza la idea de que una exploración sistemática del espacio de hiperparámetros, aunque más costosa, puede generar mejores resultados.

En todos los casos se mantuvo el `class_weight='balanced'`, lo que permitió compensar la desproporción entre clases durante el entrenamiento.""")

df_final_app = data.copy()
modelo_comparacion = cargar_modelo_optimizado()
mostrar_comparacion_optuna_vs_gridsearch(modelo_comparacion,df_final_app)

#! ==============================================================
st.markdown("---")
#! ==============================================================


st.header("🔮 Predicción y Evaluación del modelo")
st.markdown("""
En esta sección vamos a mostrar el resultado final luego de entrenar nuestro modelo optimizado de `SVM` analizando todas sus métricas resultantes y comparando si hubo una mejora, o no, respecto al primer entrenamiento hecho.        
            """)

st.subheader("Predicción y evaluación del modelo")

modelo_final = cargar_modelo_optimizado().best_estimator_

mostrar_matriz_confusion(df_final_app, modelo_final)
st.markdown("### Conclusión sobre el modelo final")
st.markdown(texto_conclusion_modelo_final)

st.subheader("Importancia de predictores")


#Cargar resultados de la importancia de predictores
@st.cache_data
def cargar_resultados_importancia():
    ruta_base = os.path.dirname(__file__)  # Carpeta actual del archivo .py
    ruta_archivo = os.path.join(ruta_base, "utils","modelos_entrenados", "importancia_permutacion.pkl")
    print(f"Cargando resultados desde: {ruta_archivo}")
    with open(ruta_archivo, "rb") as f:
        return pickle.load(f)
st.write("Importancia de características por permutación:")

df_importancia = cargar_resultados_importancia()
st.dataframe(df_importancia)

graficar_importancia_plotly(df_importancia)

st.markdown("### Variables más importantes")
st.markdown(texto_variables_mas_importantes)

#! ==============================================================
st.markdown("---")
#! ==============================================================
st.header("✅ Conclusión Final")
