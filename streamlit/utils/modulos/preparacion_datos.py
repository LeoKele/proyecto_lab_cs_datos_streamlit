import pandas as pd
import streamlit as st
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

@st.cache_data
def cargar_datos():
    # print("Directorio actual:", os.getcwd())
    ruta_archivo = os.path.join(os.getcwd(), 'data', 'Grupo 3 - Abandono de Clientes.csv')
    data = pd.read_csv(ruta_archivo, sep=',', encoding='utf-8')
    return data



#Vamos a utilizar esta funcion para limpiar los datos
def limpiar_datos(data):
    # Crear una copia del DataFrame para no modificar el original
    df_mod = data.copy()

# Convertir TotalCharges a tipo numérico (float) y manejar errores
    df_mod['TotalCharges'] = pd.to_numeric(df_mod['TotalCharges'], errors='coerce')
    # Eliminamos las filas con valores nulos en TotalCharges
    df_mod = df_mod.dropna(subset=['TotalCharges'])
    print(f"Cantidad de filas después de eliminar nulos: {len(df_mod)}")
    print(f"Filas eliminadas: {len(data) - len(df_mod)}")

    return df_mod

def codificar_datos_inicial(data):
    # Elimino costumerID
    df_inicial = data.copy()
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
    return df_inicial

def manejar_columnas_redundantes(df):
    df_mod = df.copy()
    
    # HasMultipleLines
    if 'MultipleLines' in df_mod.columns:
        df_mod['HasMultipleLines'] = ((df_mod['PhoneService'] == 1) & (df_mod['MultipleLines'] == 'Yes')).astype(int)
        df_mod.drop(columns='MultipleLines', inplace=True, errors='ignore')

    # Columnas relacionadas con InternetService
    internet_services = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    for col in internet_services:
        if col in df_mod.columns:
            new_col = f'Has{col}'
            df_mod[new_col] = ((df_mod['InternetService'] != 'No') & (df_mod[col] == 'Yes')).astype(int)

    df_mod.drop(columns=internet_services, inplace=True, errors='ignore')
    
    return df_mod



def transformacion_datos(data):
    # Copia del DataFrame original
    df_mod = data.copy()
    
    # Unificar etiquetas de PaymentMethod para coincidir con el modelo entrenado
    df_mod['PaymentMethod'] = df_mod['PaymentMethod'].replace({
        'Credit card (automatic)': 'Credit card',
        'Bank transfer (automatic)': 'Bank transfer'
    })
    
    # Convertir TotalCharges a numérico y manejar nulos
    df_mod['TotalCharges'] = pd.to_numeric(df_mod['TotalCharges'], errors='coerce')
    mediana = df_mod['TotalCharges'].median()
    df_mod['TotalCharges'] = df_mod['TotalCharges'].fillna(mediana)
    
    # Eliminar customerID
    df_mod.drop(columns='customerID', inplace=True)

    # Codificación binaria básica
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df_mod[col] = df_mod[col].replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    
    # Aplicar manejo de columnas redundantes ANTES del get_dummies
    df_mod = manejar_columnas_redundantes(df_mod)

    # One-Hot Encoding
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df_mod = pd.get_dummies(df_mod, columns=categorical_cols, drop_first=True)

    # Convertir Churn a binario
    df_mod['Churn'] = df_mod['Churn'].replace({'Yes': 1, 'No': 0})
    
    return df_mod

def estandarizar_datos(X_train, X_test):
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    return X_train_scaled, X_test_scaled

def dividir_datos(df):
    X = df.drop(columns='Churn')
    y = df['Churn']
    return train_test_split(X, y, random_state=22)