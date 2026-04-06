"""
Atenea AI - Entrenamiento del Modelo Stacking
Layer 1: RandomForest + XGBoost + GradientBoosting
Layer 2: Logistic Regression (meta-learner)
Exporta: model.pkl, explainer.pkl, students_predictions.csv
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap


FEATURES = [
    'promedio_general', 'materias_reprobadas', 'asistencia_pct',
    'tareas_entregadas_pct', 'participacion_clase',
    'reportes_disciplinarios', 'llegadas_tarde', 'dias_suspension',
    'nivel_estres', 'motivacion_escolar', 'apoyo_familiar',
    'satisfaccion_escolar', 'ansiedad_academica'
]


def train_model(base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_path = os.path.join(base_dir, 'students_dataset.csv')
    print(f"Cargando dataset desde {dataset_path}...")
    df = pd.read_csv(dataset_path)

    X = df[FEATURES]
    y = df['desercion']

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Modelo Stacking
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                    eval_metric='logloss', random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                           learning_rate=0.1, random_state=42)),
    ]

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        stack_method='predict_proba',
        cv=5,
        n_jobs=-1
    )

    print("\nEntrenando modelo Stacking (RF + XGBoost + GB → Logistic Regression)...")
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*45}")
    print(f"  RESULTADOS DEL MODELO")
    print(f"{'='*45}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"{'='*45}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=['No Deserta', 'Deserta']))

    # Guardar métricas
    metrics = {'accuracy': round(acc, 4), 'f1_score': round(f1, 4), 'auc_roc': round(auc, 4)}
    with open(os.path.join(base_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # SHAP sobre RandomForest base
    print("\nCalculando SHAP values (RandomForest base)...")
    rf_model = model.named_estimators_['rf']
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)

    # Handle both old SHAP (list) and new SHAP (single array)
    if isinstance(shap_values, list):
        sv = shap_values[1]  # old: [class0, class1]
    else:
        sv = shap_values     # new: single array for positive class

    shap_importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': np.abs(sv).mean(axis=0)
    }).sort_values('importance', ascending=False)
    print("\nTop 5 features por importancia SHAP:")
    print(shap_importance.head())

    # Predicciones para todos los alumnos
    print("\nGenerando predicciones para todos los alumnos...")
    all_probs = model.predict_proba(X)[:, 1]
    scores = (all_probs * 100).round(1)

    def get_nivel(score):
        if score > 65:   return '🔴 Alto'
        elif score > 35: return '🟡 Medio'
        else:            return '🟢 Bajo'
    top3_factors = []
    for i in range(len(X)):
        row_shap = pd.Series(sv[i], index=FEATURES)
        top3 = row_shap.nlargest(3).index.tolist()
        top3_factors.append(', '.join(top3))

    df_out = df[['id', 'nombre', 'grupo', 'semestre']].copy()
    df_out['score_riesgo']   = scores
    df_out['nivel_riesgo']   = df_out['score_riesgo'].apply(get_nivel)
    df_out['top3_factores']  = top3_factors
    df_out['prob_desercion'] = all_probs.round(4)

    predictions_path = os.path.join(base_dir, 'students_predictions.csv')
    model_path       = os.path.join(base_dir, 'model.pkl')
    explainer_path   = os.path.join(base_dir, 'explainer.pkl')

    df_out.to_csv(predictions_path, index=False)
    print(f"✅ Predicciones guardadas: {predictions_path}")

    joblib.dump(model, model_path)
    joblib.dump({'explainer': explainer, 'features': FEATURES, 'shap_values': sv}, explainer_path)
    print(f"✅ Modelo guardado: {model_path}")
    print(f"✅ Explainer guardado: {explainer_path}")
    print("\n🎉 Entrenamiento completado.")


if __name__ == '__main__':
    train_model()
