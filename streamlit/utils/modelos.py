# import streamlit as st
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, precision_score, average_precision_score
# import pandas as pd

# @st.cache_data
# def evaluar_modelos_iniciales(X_train, X_test, y_train, y_test):
#     modelos = {
#         'Regresión Logística': LogisticRegression(max_iter=1000, class_weight='balanced'),
#         'HistGradientBoosting': HistGradientBoostingClassifier(),
#         'XGBoost': XGBClassifier(eval_metric='logloss', scale_pos_weight=2.773, use_label_encoder=False),
#         'LightGBM': LGBMClassifier(verbose=-1, class_weight='balanced'),
#         'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
#         'SVM_linear': SVC(kernel='linear', class_weight='balanced'),
#         'SVM_rbf': SVC(kernel='rbf', class_weight='balanced')
#     }

#     resultados = []
    
#     for nombre, modelo in modelos.items():
#         modelo.fit(X_train, y_train)
#         y_pred = modelo.predict(X_test)
        
#         if hasattr(modelo, "predict_proba"):
#             y_proba = modelo.predict_proba(X_test)[:, 1]
#         elif hasattr(modelo, "decision_function"):
#             y_proba = modelo.decision_function(X_test)
#         else:
#             y_proba = None

#         resultados.append({
#             'Modelo': nombre,
#             'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
#             'F1 Score': f1_score(y_test, y_pred),
#             'Recall': recall_score(y_test, y_pred),
#             'Precision': precision_score(y_test, y_pred),
#             'PR AUC': average_precision_score(y_test, y_proba) if y_proba is not None else None
#         })

#     return pd.DataFrame(resultados).sort_values(by='Balanced Accuracy', ascending=False)

# @st.cache_data
# def evaluar_modelos_iniciales_estandarizados(X_train, X_test, y_train, y_test):
#     modelos = {
#         'STD - Regresión Logística': LogisticRegression(max_iter=1000, class_weight='balanced'),
#         'STD - SVM_linear': SVC(kernel='linear', class_weight='balanced'),
#         'STD - SVM_rbf': SVC(kernel='rbf', class_weight='balanced')
#     }

#     resultados = []
    
#     for nombre, modelo in modelos.items():
#         modelo.fit(X_train, y_train)
#         y_pred = modelo.predict(X_test)
        
#         if hasattr(modelo, "predict_proba"):
#             y_proba = modelo.predict_proba(X_test)[:, 1]
#         elif hasattr(modelo, "decision_function"):
#             y_proba = modelo.decision_function(X_test)
#         else:
#             y_proba = None

#         resultados.append({
#             'Modelo': nombre,
#             'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
#             'F1 Score': f1_score(y_test, y_pred),
#             'Recall': recall_score(y_test, y_pred),
#             'Precision': precision_score(y_test, y_pred),
#             'PR AUC': average_precision_score(y_test, y_proba) if y_proba is not None else None
#         })

#     return pd.DataFrame(resultados).sort_values(by='Balanced Accuracy', ascending=False)