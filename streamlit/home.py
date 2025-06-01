import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preparacion_datos import limpiar_datos, codificar_datos_inicial, dividir_datos, estandarizar_datos, transformacion_datos
from utils.comparacion_modelos import comparar_resultados_interactivo, comparar_metricas_modelos_especificos
from utils.modelos import evaluar_modelos_iniciales, evaluar_modelos_iniciales_estandarizados

# Configuración inicial de la página
st.set_page_config(page_title="Análisis de Abandono de Clientes", layout="wide")

# Título y presentación
st.title("🎓 Trabajo Integrador Final - Abandono de Clientes")
st.markdown("""
## Integrantes:
-
-
- Evers Juan Segundo
- Kelechian Leonardo
""")

# Cargar datos (esto podrías moverlo a preparacion_datos.py)
@st.cache_data
def cargar_datos():
    ruta_archivo = "data\Grupo 3 - Abandono de Clientes.csv"  # Ajustar ruta
    data = pd.read_csv(ruta_archivo, sep=',', encoding='utf-8')
    return data

# Cargar datos una vez
data = cargar_datos()



# Introducción
st.header("Introducción")
st.markdown("""
[Contenido introductorio del proyecto...]
""")

# Preparación de Datos
st.header("⚙️ Preparación de los Datos")
st.subheader("Limpieza de datos")
st.markdown("""
Realizamos las siguientes transformaciones:
- Conversión de TotalCharges a numérico
- Tratamiento de valores nulos con imputación por mediana
""")

# Mostrar código de preparación (simplificado)
with st.expander("Ver código de limpieza"):
    st.code("""
    # Crear una copia del DataFrame
    df_mod = data.copy()

    # Convertir TotalCharges a numérico y manejar nulos
    df_mod['TotalCharges'] = pd.to_numeric(df_mod['TotalCharges'], errors='coerce')
    mediana = df_mod['TotalCharges'].median()
    df_mod['TotalCharges'] = df_mod['TotalCharges'].fillna(mediana)
    
    # Codificación de variables...
    """, language='python')

# Mostrar datos antes/después
st.subheader("Datos originales")
st.dataframe(data.head())


# Modelo Base
st.header("🔎 Búsqueda de Modelo Base")
st.markdown("""
En esta sección comparamos el rendimiento de diferentes modelos para predecir el abandono de clientes.
""")

st.subheader("Fase 1 - Comparación rápida sin procesar")
code_evaluar_modelos = """
#Esta función agarraria la BBDD sin realizar transformaciones y luego nos devuelve las métricas de cada modelo puesto
def evaluar_modelos(X_train, y_train, X_test, y_test):
    modelos = {
        #Estos son "tweaks iniciales seguros" que no se consideran tuning agresivo, sino buenas prácticas para establecer un baseline justo
        'Regresión Logística': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'HistGradientBoosting': HistGradientBoostingClassifier(),
        'XGBoost': XGBClassifier(eval_metric='logloss', scale_pos_weight=2.773, use_label_encoder=False), #scale_pos_weight = 'Churn' 73.5% (yes) / 26.5% (no) = 2.773
        'LightGBM': LGBMClassifier(verbose=-1, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'SVM_linear': SVC(kernel = 'linear', class_weight='balanced'),
        'SVM_rbf': SVC(kernel = 'rbf', class_weight='balanced')
    }

    resultados = []

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        # Asegurarse de que el modelo tiene el atributo predict_proba antes de usarlo
        if hasattr(modelo, "predict_proba"):
            y_proba = modelo.predict_proba(X_test)[:, 1]
        elif hasattr(modelo, "decision_function"):
            y_proba = modelo.decision_function(X_test)
        else:
            y_proba = None


        resultados.append({
            'Modelo': nombre,
            'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred), #Calcula el promedio de recall entre clases (recall clase 0 y clase 1).
            'F1 Score': f1_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred), #Indica cuántos de los que realmente se fueron (churn=1) tu modelo detectó.
            'Precision': precision_score(y_test, y_pred),
            # 'ROC AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None
            #PR AUC
            'PR AUC': average_precision_score(y_test,y_proba) if y_proba is not None else None #es más informativa cuando te interesa detectar bien la clase 1.
        })

    return pd.DataFrame(resultados).sort_values(by='Balanced Accuracy', ascending=False).reset_index(drop=True)
    """.strip()
st.markdown("### Función para evaluar modelos")
st.code(code_evaluar_modelos, language='python')


code_codificacion_rapida = """
# Elimino costumerID
df_inicial = df_mod.copy()
df_inicial.drop(columns='customerID', inplace=True)

# Realizamos modificaciones rapidas porque nuestros modelos necesitan variables numericas
binary_cols_quick = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

# Variables categóricas con más de dos opciones (ajusta la lista si es necesario)
categorical_cols_quick = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                          'Contract', 'PaymentMethod']

# Aplicar codificación binaria a las columnas Yes/No
for col in binary_cols_quick:
    if col in df_inicial.columns:
        if df_inicial[col].dtype == 'object': # Solo codificar si son strings
             # Manejar el caso de 'No internet service' o 'No phone service' si aplica
            df_inicial[col] = df_inicial[col].replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})


# Aplicar One-Hot Encoding a las columnas con más de dos opciones
df_inicial = pd.get_dummies(df_inicial, columns=categorical_cols_quick, drop_first=True)

# Convertir las nuevas columnas booleanas a enteros (si pd.get_dummies las crea como bool)
bool_cols_quick = df_inicial.select_dtypes(include='bool').columns
df_inicial[bool_cols_quick] = df_inicial[bool_cols_quick].astype(int)

#Convertimos tambien la variable objetivo
df_inicial['Churn'] = df_inicial['Churn'].replace({'Yes': 1, 'No': 0})

# --- Fin de la codificación rápida ---
""".strip()
st.markdown("### Codificación rápida de variables")
st.code(code_codificacion_rapida, language='python')




code_division_datos = """
# División de los datos en train y test
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
                                        df_inicial.drop(columns = 'Churn'),
                                        df_inicial['Churn'],
                                        random_state = 22
                                    )
""".strip()
st.markdown("### División de los datos en train y test")
st.code(code_division_datos, language='python')

code_evaluar_modelos = """
# Evaluamos los modelos inciales
# ==============================================================================
df_resultados = evaluar_modelos(X_train, y_train, X_test, y_test)
df_resultados
""".strip()
st.markdown("### Evaluación de los modelos iniciales")
st.code(code_evaluar_modelos, language='python')

df_mod = limpiar_datos(data)
df_inicial = codificar_datos_inicial(df_mod)

# Dividir datos
X_train, X_test, y_train, y_test = dividir_datos(df_inicial)
df_resultados = evaluar_modelos_iniciales(X_train, X_test, y_train, y_test)
st.dataframe(df_resultados)

st.subheader("Fase 2 - Comparación con estandarización")
st.markdown("Utilizamos el mismo split de datos de antes...")
code_escalar_inicial = """
# Variables a escalar
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Inicializar el escalador
scaler = StandardScaler()

# Ajustar el escalador solo con datos de entrenamiento
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
""".strip()
st.markdown("### Escalado de variables numéricas")  
st.code(code_escalar_inicial, language='python')

st.markdown("Ajustamos la otra función para entrenar nuevamente pero sin los arboles...")
code_evaluar_modelos_estandarizados = """
def evaluar_modelos_estandarizados(X_train, y_train, X_test, y_test):
    modelos = {
        #Estos son "tweaks iniciales seguros" que no se consideran tuning agresivo, sino buenas prácticas para establecer un baseline justo
        'STD - Regresión Logística': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'STD - SVM_linear': SVC(kernel = 'linear', class_weight='balanced'),
        'STD - SVM_rbf': SVC(kernel = 'rbf', class_weight='balanced')
    }

    resultados = []

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        # Asegurarse de que el modelo tiene el atributo predict_proba antes de usarlo
        if hasattr(modelo, "predict_proba"):
            y_proba = modelo.predict_proba(X_test)[:, 1]
        elif hasattr(modelo, "decision_function"):
            y_proba = modelo.decision_function(X_test)
        else:
            y_proba = None


        resultados.append({
            'Modelo': nombre,
            'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred), #Calcula el promedio de recall entre clases (recall clase 0 y clase 1).
            'F1 Score': f1_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred), #Indica cuántos de los que realmente se fueron (churn=1) tu modelo detectó.
            'Precision': precision_score(y_test, y_pred),
            # 'ROC AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None
            #PR AUC
            'PR AUC': average_precision_score(y_test,y_proba) if y_proba is not None else None #es más informativa cuando te interesa detectar bien la clase 1.
        })

    return pd.DataFrame(resultados).sort_values(by='Balanced Accuracy', ascending=False).reset_index(drop=True)
""".strip()
st.markdown("### Función para evaluar modelos estandarizados")
st.code(code_evaluar_modelos_estandarizados, language='python')

code_evaluar_modelos_estandarizados = """
# Evaluamos los modelos inciales
# ==============================================================================
df_resultados_std = evaluar_modelos_estandarizados(X_train_scaled, y_train, X_test_scaled, y_test)
df_resultados_std
""".strip()
st.markdown("### Evaluación de los modelos iniciales estandarizados")
st.code(code_evaluar_modelos_estandarizados, language='python')


df_mod = limpiar_datos(data)
df_inicial = codificar_datos_inicial(df_mod)

# Dividir datos
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled, X_test_scaled = estandarizar_datos(X_train_scaled, X_test_scaled)
df_resultados_std = evaluar_modelos_iniciales_estandarizados(X_train_scaled, X_test_scaled, y_train, y_test)
st.dataframe(df_resultados_std)


st.markdown("---")
st.subheader("Comparación de resultados")

df_todos = comparar_resultados_interactivo(df_resultados, df_resultados_std)


st.subheader("Conclusiones")
st.markdown("""
* ✅ El modelo con mejor rendimiento general fue el `SVM_rbf` estandarizado, alcanzando la mayor Balanced Accuracy entre todos los modelos comparados.
* 🌳 El modelo `lightGBM` fue el que mejor accuracy tuvo de todos los árboles, teniendo casi el mismo valor que el SVM. Similar, solo por apenas debajo de este, se encuentra el de `Regresión Logística`. Ambos tambien podrían ser considerados como modelo final.
* ⚠️ El `SVM_linear` sin estandarizar puede no ser confiable, esto se debe a que luego de ser escalado, su accuracy es mucho menor, lo que nos hace pensar que en ese primer entrenamiento se vió sesgado por la escala de alguna variable.
""")

st.subheader("Comparación de métricas específicas")
comparar_metricas_modelos_especificos(df_todos)



md_conclusion_final = """
<h3>Conclusion final</h3>

Entonces, creo que el `LightGBM` porque, si bien no tiene el mayor recall (proporción de churners reales sí detectados), la precisión es la mas alta (de los que detectamos como churn, cuantos realmente lo eran). Esta ultima, si es baja, estaríamos "sobrealertando" o etiquetando como churners a muchos que en realidad no lo eran (falsos positivos).

Este balance hace que se detecten bien a los que se van (recall aceptable) y no sobreactúe marcando como "churn" a muchos que no lo son (mayor precision)

❗Pero, si nuestro objetivo fuera maximizar el recall (ej.:"prefiero contactar 10 clientes de más que perder 1 que se va") podríamos optar por la `Regresión Logística`.

Si la prioridad es no perder clientes, entonces:

- `SVM_rbf` ofrece un casi idéntico recall a la `regresión logística`, pero mejora en F1 Score y precisión.

- Esto sugiere que detectamos casi los mismos casos de churn, pero con menos falsos positivos, lo cual es deseable si las acciones de retención tienen costo.
""".strip()
st.markdown(md_conclusion_final, unsafe_allow_html=True)


st.markdown("---")


st.subheader("Manejo de Redundancia en variables categóricas") 