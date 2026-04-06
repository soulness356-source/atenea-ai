"""
Atenea AI - Generador de Dataset Simulado
Genera 500 estudiantes con variables académicas, conductuales y psicoemocionales.
"""

import numpy as np
import pandas as pd
from faker import Faker
import os


def generate_dataset(output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    fake = Faker('es_MX')
    np.random.seed(42)

    N = 500

    def clip(val, lo, hi):
        return np.clip(val, lo, hi)

    # Factor de riesgo latente (0 = bajo riesgo, 1 = alto riesgo)
    riesgo_latente = np.random.beta(a=2, b=4, size=N)  # ~25-30% alta deserción

    # Variables académicas (correlacionan con riesgo)
    promedio_general      = clip(10 - riesgo_latente * 7 + np.random.normal(0, 0.8, N), 0, 10).round(1)
    materias_reprobadas   = clip(np.round(riesgo_latente * 7 + np.random.normal(0, 0.5, N)).astype(int), 0, 8)
    asistencia_pct        = clip(100 - riesgo_latente * 60 + np.random.normal(0, 5, N), 0, 100).round(1)
    tareas_entregadas_pct = clip(100 - riesgo_latente * 55 + np.random.normal(0, 6, N), 0, 100).round(1)
    participacion_clase   = clip(np.round(5 - riesgo_latente * 3.5 + np.random.normal(0, 0.3, N)).astype(int), 1, 5)

    # Variables conductuales
    reportes_disciplinarios = clip(np.round(riesgo_latente * 8 + np.random.normal(0, 0.5, N)).astype(int), 0, 10)
    llegadas_tarde          = clip(np.round(riesgo_latente * 25 + np.random.normal(0, 2, N)).astype(int), 0, 30)
    dias_suspension         = clip(np.round(riesgo_latente * 12 + np.random.normal(0, 0.8, N)).astype(int), 0, 15)

    # Variables psicoemocionales
    nivel_estres         = clip(np.round(1 + riesgo_latente * 3.8 + np.random.normal(0, 0.3, N)).astype(int), 1, 5)
    motivacion_escolar   = clip(np.round(5 - riesgo_latente * 3.5 + np.random.normal(0, 0.3, N)).astype(int), 1, 5)
    apoyo_familiar       = clip(np.round(5 - riesgo_latente * 3 + np.random.normal(0, 0.4, N)).astype(int), 1, 5)
    satisfaccion_escolar = clip(np.round(5 - riesgo_latente * 3.2 + np.random.normal(0, 0.3, N)).astype(int), 1, 5)
    ansiedad_academica   = clip(np.round(1 + riesgo_latente * 3.6 + np.random.normal(0, 0.3, N)).astype(int), 1, 5)

    # Variable objetivo (deserción)
    prob_desercion = (
        0.25 * (1 - promedio_general / 10) +
        0.15 * (materias_reprobadas / 8) +
        0.15 * (1 - asistencia_pct / 100) +
        0.10 * (nivel_estres / 5) +
        0.10 * (1 - motivacion_escolar / 5) +
        0.08 * (1 - apoyo_familiar / 5) +
        0.07 * (reportes_disciplinarios / 10) +
        0.05 * (llegadas_tarde / 30) +
        0.05 * (ansiedad_academica / 5)
    )
    prob_desercion = clip(prob_desercion + np.random.normal(0, 0.05, N), 0, 1)
    desercion = (prob_desercion > 0.45).astype(int)

    # Grupos y semestres
    grupos    = np.random.choice(['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B'], size=N)
    semestres = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=N)

    # Nombres e IDs
    nombres = [fake.name() for _ in range(N)]
    ids     = [f"EST-{str(i+1).zfill(4)}" for i in range(N)]

    df = pd.DataFrame({
        'id': ids,
        'nombre': nombres,
        'grupo': grupos,
        'semestre': semestres,
        'promedio_general': promedio_general,
        'materias_reprobadas': materias_reprobadas,
        'asistencia_pct': asistencia_pct,
        'tareas_entregadas_pct': tareas_entregadas_pct,
        'participacion_clase': participacion_clase,
        'reportes_disciplinarios': reportes_disciplinarios,
        'llegadas_tarde': llegadas_tarde,
        'dias_suspension': dias_suspension,
        'nivel_estres': nivel_estres,
        'motivacion_escolar': motivacion_escolar,
        'apoyo_familiar': apoyo_familiar,
        'satisfaccion_escolar': satisfaccion_escolar,
        'ansiedad_academica': ansiedad_academica,
        'desercion': desercion
    })

    out_path = os.path.join(output_dir, 'students_dataset.csv')
    df.to_csv(out_path, index=False)
    print(f"✅ Dataset generado: {N} estudiantes")
    print(f"   Tasa de deserción: {desercion.mean():.1%}")
    print(f"   Alumnos en riesgo: {desercion.sum()} de {N}")
    print(f"   Guardado en: {out_path}")
    return out_path


if __name__ == '__main__':
    generate_dataset()
