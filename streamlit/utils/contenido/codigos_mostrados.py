code_librerias = """
# Conexión con Google Drive
# ==============================================================================
from google.colab import drive
import os

# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np
import pickle

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# Análisis Estadísticos
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.stats import uniform
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')
import math

# PreProcesado y modelado
# ==============================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedKFold, GridSearchCV, RandomizedSearchCV
import multiprocessing
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, average_precision_score, make_scorer, ConfusionMatrixDisplay, confusion_matrix
""".strip()


code_limpieza = """
# Creamos una copia del DataFrame para no modificar el original
df_mod = data.copy()

# Convertimos TotalCharges a tipo numérico (float) y manejar errores
df_mod['TotalCharges'] = pd.to_numeric(df_mod['TotalCharges'], errors='coerce')

# Revisamos valores faltantes o nulos en TotalCharges
print(f"Cantidad de filas con valor nulo en 'Total Charges': {df_mod['TotalCharges'].isnull().sum()}");
""".strip()

code_limpieza2 = """
# Eliminamos las filas con valores nulos en TotalCharges
df_mod = df_mod.dropna(subset=['TotalCharges'])
print(f"Cantidad de filas después de eliminar nulos: {len(df_mod)}")
print(f"Filas eliminadas: {len(data) - len(df_mod)}")
""".strip()

code_limpieza3 = """
df_mod.PaymentMethod.unique()
""".strip()
code_limpieza4 = """
df_mod['PaymentMethod'] = df_mod['PaymentMethod'].str.replace(' (automatic)', '', regex=False)
df_mod.PaymentMethod.unique()
""".strip()



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

code_division_datos = """
# División de los datos en train y test
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
                                        df_inicial.drop(columns = 'Churn'),
                                        df_inicial['Churn'],
                                        random_state = 22
                                    )
""".strip()

code_evaluacion_modelos = """
# Evaluamos los modelos inciales
# ==============================================================================
df_resultados = evaluar_modelos(X_train, y_train, X_test, y_test)
df_resultados
""".strip()


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

code_evaluacion_modelos_estandarizados = """
# Evaluamos los modelos inciales
# ==============================================================================
df_resultados_std = evaluar_modelos_estandarizados(X_train_scaled, y_train, X_test_scaled, y_test)
df_resultados_std
""".strip()



#! ================================================
#! ================================================
#! ================================================
#? Preparacion de los datos para el modelo
code_manejo_redun_1 = """
df_mod_1 = df_mod.copy()
df_mod_1['HasMultipleLines'] = ((df_mod_1['PhoneService'] == 1) & (df_mod_1['MultipleLines'] == 'Yes')).astype(int)
df_mod_1.drop(columns='MultipleLines', inplace=True)
""".strip()

code_manejo_redun_2 = """
df_mod_2 = df_mod_1.copy()
# Lista de columnas relacionadas con servicios que dependen de tener Internet
internet_services = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

# Creamos nuevas variables binarias
for col in internet_services:
    new_col = 'Has' + col  # Ej: 'HasOnlineSecurity'
    df_mod_2[new_col] = ((df_mod_2['InternetService'] != 'No') & (df_mod_2[col] == 'Yes')).astype(int)

# Eliminamos columnas originales
df_mod_2.drop(columns=internet_services, inplace=True)
""".strip()


code_transformacion_1 = """
# Transformamos variables categóricas de 3 o más opciones con el método dummy coding.
df_mod_3 = df_mod_2.copy()

# Lista de columnas categóricas a codificar
categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']

# Aplicamos One-Hot Encoding
df_mod_3 = pd.get_dummies(df_mod_3, columns=categorical_cols, drop_first=True)
""".strip()

code_transformacion_2 = """
# Todas las nuevas columnas creadas por el dummy coding son booleanas, las transformamos a numéricas
df_mod_4 = df_mod_3.copy()
bool_cols = df_mod_4.select_dtypes(include='bool').columns
df_mod_4[bool_cols] = df_mod_4[bool_cols].astype(int)
""".strip()

code_transformacion_3 = """
# Reemplazo específico para 'gender'
df_mod_5 = df_mod_4.copy()
df_mod_5['gender'] = df_mod_5['gender'].replace({'Male': 1, 'Female': 0})

# Otras variables categóricas binarias (Yes/No)
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling','Churn']
df_mod_5[binary_cols] = df_mod_5[binary_cols].replace({'Yes': 1, 'No': 0})
""".strip()

code_df_final = """
# Eliminamos costumerID
df_final = df_mod_5.copy()
df_final.drop(columns='customerID', inplace=True)
""".strip()

code_split = """
# División de los datos en train y test
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
                                        df_final.drop(columns = 'Churn'),
                                        df_final['Churn'],
                                        random_state = 22
                                    )
""".strip()

code_estandarizacion = """
# Variables a escalar
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Inicializamos el escalador
scaler = StandardScaler()

# Ajustamos el escalador solo con datos de entrenamiento
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
""".strip()


code_resultados_busquedas = """
def agregar_resultado_busqueda(df_resultados, search_obj, nombre_metodo):
    nueva_fila = {
        'Método': nombre_metodo,
        'Mejores Hiperparámetros': search_obj.best_params_,
        'Mejor Score': search_obj.best_score_,
        'Scoring': search_obj.scoring
    }
    return pd.concat([df_resultados, pd.DataFrame([nueva_fila])], ignore_index=True)
""".strip()


code_gridSearchBasico = """
# Grid de hiperparámetros evaluados
# ==============================================================================
param_grid_basico = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
    'class_weight': ['balanced']
}

# Búsqueda por grid search con validación cruzada
# ==============================================================================
grid_basico = GridSearchCV(
        estimator  = SVC(kernel='rbf',random_state=22),
        param_grid = param_grid_basico,
        scoring    = 'balanced_accuracy',
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = StratifiedKFold(n_splits=10, shuffle=True, random_state=22), #Mantiene la proporción de clases en cada fold
        refit      = True,
        verbose    = 0,
        return_train_score = True
       )

grid_basico.fit(X = X_train_scaled, y = y_train)
""".strip()

code_gridSearchFino = """
# Grid de hiperparámetros evaluados
# ==============================================================================
# Esto hace que el rango de los parametros sea mas 'fino' agregando valores intermedios
param_grid_fino = {
    'C': np.logspace(-2, 4, 10),         # [0.01, 0.046, ..., 10000]
    'gamma': np.logspace(-4, 1, 10),     # [0.0001, ..., 10]
    'class_weight': ['balanced']
}

# Búsqueda por grid search con validación cruzada
# ==============================================================================
grid_fino = GridSearchCV(
        estimator  = SVC(kernel='rbf',random_state=22),
        param_grid = param_grid_fino,
        scoring    = 'balanced_accuracy',
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = StratifiedKFold(n_splits=10, shuffle=True, random_state=22), #Mantiene la proporción de clases en cada fold
        refit      = True,
        verbose    = 0,
        return_train_score = True
       )

grid_fino.fit(X = X_train_scaled, y = y_train)
""".strip()


code_gridSearchFinoV2 = """
# Grid de hiperparámetros evaluados
# ==============================================================================
#Hacemos un "zoom" local alrededor del mejor gamma y c para intentar encontrar una mejora de rendimiento
param_grid_fino_v2 = {
    'C': np.linspace(0.5, 2, 10),
    'gamma': np.linspace(0.01, 0.025, 10),
    'class_weight': ['balanced']
}


# Búsqueda por grid search con validación cruzada
# ==============================================================================
grid_fino_v2 = GridSearchCV(
        estimator  = SVC(kernel='rbf',random_state=22),
        param_grid = param_grid_fino_v2,
        scoring    = 'balanced_accuracy',
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = StratifiedKFold(n_splits=10, shuffle=True, random_state=22), #Mantiene la proporción de clases en cada fold
        refit      = True,
        verbose    = 0,
        return_train_score = True
       )

grid_fino_v2.fit(X = X_train_scaled, y = y_train)

""".strip()


code_randomizedSearch ="""
# Grid de hiperparámetros evaluados
# ==============================================================================
param_randomS = {
    'C': uniform(loc=0.1, scale=100),        # valores entre 0.1 y 100.1
    'gamma': uniform(loc=0.0001, scale=1),   # valores entre 0.0001 y 1.0001
    'class_weight': ['balanced']
}

# Búsqueda por grid search con validación cruzada
# ==============================================================================
random_search = RandomizedSearchCV(
    estimator=SVC(kernel='rbf', random_state=22),
    param_distributions=param_randomS,
    n_iter=30,  # cantidad de combinaciones a probar
    scoring='balanced_accuracy',
    n_jobs=multiprocessing.cpu_count() - 1,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=22),
    refit=True,
    verbose=0,
    random_state=22,
    return_train_score=True
)

random_search.fit(X_train_scaled, y_train)
""".strip()