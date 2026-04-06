# 🦉 Atenea AI — MVP Demo

Sistema de predicción de deserción escolar usando Machine Learning (Stacking Ensemble).

## Demo en Streamlit Cloud

El app se auto-configura en el primer arranque: genera el dataset sintético y entrena el modelo automáticamente (~60 s). No se necesita ningún paso previo.

## Ejecutar localmente

### 1. Instalar dependencias
```bash
cd EduPredict_MVP
pip install -r requirements.txt
```

### 2. Lanzar la demo
```bash
streamlit run app.py
```
Abre → http://localhost:8501

El primer arranque genera el dataset y entrena el modelo de forma automática.

---

## Estructura del proyecto
```
EduPredict_MVP/
├── generate_dataset.py    → Genera dataset simulado
├── train_model.py         → Entrena Stacking + SHAP
├── app.py                 → Dashboard Streamlit
├── requirements.txt       → Dependencias
├── students_dataset.csv   → Dataset (generado)
├── students_predictions.csv → Predicciones (generado)
├── model.pkl              → Modelo entrenado (generado)
└── explainer.pkl          → SHAP explainer (generado)
```

## Modelo
- **Layer 1:** RandomForest + XGBoost + GradientBoosting (CV 5-fold)
- **Layer 2:** Logistic Regression (meta-learner)
- **Interpretabilidad:** SHAP TreeExplainer

## Variables
- **Académicas:** promedio, materias reprobadas, asistencia, tareas, participación
- **Conductuales:** reportes disciplinarios, llegadas tarde, suspensiones
- **Psicoemocionales:** estrés, motivación, apoyo familiar, satisfacción, ansiedad

## Score de riesgo
- 🔴 **Alto:** score > 65
- 🟡 **Medio:** score 35–65
- 🟢 **Bajo:** score < 35

---
EduPredict AI v0.1 | Luis Pardo | UAQ DTE
