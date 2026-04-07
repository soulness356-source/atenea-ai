"""
Atenea AI — Dashboard Streamlit
Demo MVP: Predicción de deserción escolar
Corre con: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import os

# ── Directorio base (funciona igual en local y en Streamlit Cloud) ─────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Atenea AI",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Credenciales por escuela ──────────────────────────────────────────────────
CREDENTIALS = {
    "demo":      {"password": "demo2024",   "school": "Escuela Demo"},
    "plantel_a": {"password": "atenea2024", "school": "Plantel A"},
    "admin":     {"password": "admin2024",  "school": "Admin"},
}

# ── Pantalla de login ─────────────────────────────────────────────────────────
def show_login():
    st.markdown("""
    <style>
        .login-container {
            max-width: 420px;
            margin: 4rem auto 0 auto;
            padding: 2.5rem 2rem;
            background: #ffffff;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(124, 58, 237, 0.12);
            text-align: center;
        }
        .login-owl   { font-size: 5rem; margin-bottom: 0.2rem; }
        .login-title { font-size: 2rem; font-weight: 800; color: #7c3aed; margin: 0; }
        .login-sub   { font-size: 1rem; color: #6c757d; margin-top: 0.3rem; margin-bottom: 2rem; }
    </style>
    <div class="login-container">
        <div class="login-owl">🦉</div>
        <p class="login-title">Atenea AI</p>
        <p class="login-sub">Sistema de Inteligencia Educativa</p>
    </div>
    """, unsafe_allow_html=True)

    # Centrar los campos de login
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("<br>", unsafe_allow_html=True)
        usuario = st.text_input("Usuario", placeholder="usuario", key="login_user")
        password = st.text_input("Contraseña", type="password", placeholder="contraseña", key="login_pass")
        entrar = st.button("Entrar", use_container_width=True, type="primary")

        if entrar:
            cred = CREDENTIALS.get(usuario)
            if cred and cred["password"] == password:
                st.session_state["authenticated"] = True
                st.session_state["school_name"] = cred["school"]
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

# ── Verificar autenticación ───────────────────────────────────────────────────
if not st.session_state.get("authenticated", False):
    show_login()
    st.stop()

# ── Onboarding para docentes nuevos ──────────────────────────────────────────
def show_onboarding():
    if "onboarding_step" not in st.session_state:
        st.session_state["onboarding_step"] = 1

    step = st.session_state["onboarding_step"]
    total_steps = 4

    st.markdown("""
    <style>
        .ob-card {
            max-width: 680px;
            margin: 3rem auto;
            padding: 2.8rem 2.5rem 2rem 2.5rem;
            background: #ffffff;
            border-radius: 24px;
            box-shadow: 0 12px 48px rgba(124, 58, 237, 0.13);
        }
        .ob-step-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #7c3aed;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }
        .ob-title {
            font-size: 1.9rem;
            font-weight: 800;
            color: #1a1a2e;
            margin: 0 0 0.6rem 0;
        }
        .ob-body {
            font-size: 1.05rem;
            color: #444;
            line-height: 1.65;
            margin-bottom: 1.4rem;
        }
        .ob-progress-bar-bg {
            background: #e9ecef;
            border-radius: 99px;
            height: 6px;
            margin-bottom: 1.8rem;
        }
        .ob-progress-bar-fill {
            background: linear-gradient(90deg, #7c3aed, #a78bfa);
            border-radius: 99px;
            height: 6px;
        }
        .ob-risk-row {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            margin-bottom: 0.6rem;
            font-size: 1rem;
            font-weight: 600;
        }
        .ob-risk-alto  { background: #fff0f0; color: #c0392b; border: 1.5px solid #f5c6c6; }
        .ob-risk-medio { background: #fffbf0; color: #9a6400; border: 1.5px solid #ffe08a; }
        .ob-risk-bajo  { background: #f0fff4; color: #1a7c3e; border: 1.5px solid #b2dfdb; }
    </style>
    """, unsafe_allow_html=True)

    progress_pct = int((step / total_steps) * 100)

    st.markdown(f"""
    <div class="ob-card">
        <div class="ob-step-label">Paso {step} de {total_steps}</div>
        <div class="ob-progress-bar-bg">
            <div class="ob-progress-bar-fill" style="width:{progress_pct}%;"></div>
        </div>
    """, unsafe_allow_html=True)

    if step == 1:
        st.markdown("""
        <div class="ob-title">¡Bienvenido a Atenea AI 🦉</div>
        <div style="font-size:1rem; color:#7c3aed; font-weight:600; margin-bottom:0.8rem;">
            Sistema de Inteligencia Educativa
        </div>
        <div class="ob-body">
            Atenea AI analiza el riesgo de deserción escolar de tus alumnos usando inteligencia artificial.
            En menos de 2 minutos aprenderás a usar el sistema.
        </div>
        </div>
        """, unsafe_allow_html=True)

        col_skip, _, col_next = st.columns([2, 3, 2])
        with col_skip:
            if st.button("Saltar tutorial", use_container_width=True):
                st.session_state["onboarding_done"] = True
                st.rerun()
        with col_next:
            if st.button("Siguiente →", use_container_width=True, type="primary"):
                st.session_state["onboarding_step"] = 2
                st.rerun()

    elif step == 2:
        st.markdown("""
        <div class="ob-title">¿Qué es el Score de Riesgo?</div>
        <div class="ob-body">
            Cada alumno recibe un <strong>score de 0 a 100</strong> calculado por el modelo de IA.
            Cuanto más alto el número, mayor es la probabilidad de que el alumno abandone sus estudios.
            El sistema clasifica a cada alumno en uno de tres niveles:
        </div>
        <div class="ob-risk-row ob-risk-alto">
            🔴&nbsp; Alto (65 – 100) — Requiere acción inmediata
        </div>
        <div class="ob-risk-row ob-risk-medio">
            🟡&nbsp; Medio (35 – 65) — Monitorear de cerca
        </div>
        <div class="ob-risk-row ob-risk-bajo">
            🟢&nbsp; Bajo (0 – 35) — Sin intervención urgente
        </div>
        </div>
        """, unsafe_allow_html=True)

        _, col_next = st.columns([5, 2])
        with col_next:
            if st.button("Siguiente →", use_container_width=True, type="primary"):
                st.session_state["onboarding_step"] = 3
                st.rerun()

    elif step == 3:
        st.markdown("""
        <div class="ob-title">Filtra y encuentra a tus alumnos</div>
        <div class="ob-body">
            En el <strong>sidebar izquierdo</strong> puedes filtrar la lista por <em>nivel de riesgo</em>,
            <em>grupo</em> y <em>semestre</em> para enfocarte en quienes más lo necesitan.<br><br>
            Haz clic en <strong>Ver detalle</strong> junto a cualquier alumno para abrir su perfil completo:
            score, factores de riesgo (análisis SHAP) y un plan de intervención personalizado.<br><br>
            Desde el perfil también puedes <strong>exportar el reporte individual en CSV</strong>
            para compartirlo con orientadores o directivos.
        </div>
        </div>
        """, unsafe_allow_html=True)

        _, col_next = st.columns([5, 2])
        with col_next:
            if st.button("Siguiente →", use_container_width=True, type="primary"):
                st.session_state["onboarding_step"] = 4
                st.rerun()

    elif step == 4:
        st.markdown("""
        <div class="ob-title">Todo listo 🎉</div>
        <div class="ob-body">
            Ya tienes todo lo que necesitas para identificar a los alumnos que más te necesitan.
            El sistema se actualiza con cada ciclo escolar para mantenerse preciso y relevante.<br><br>
            <strong>¡Mucho éxito — cada intervención a tiempo puede cambiar una historia!</strong>
        </div>
        </div>
        """, unsafe_allow_html=True)

        _, col_enter = st.columns([5, 2])
        with col_enter:
            if st.button("¡Entrar al sistema!", use_container_width=True, type="primary"):
                st.session_state["onboarding_done"] = True
                st.rerun()

if not st.session_state.get("onboarding_done", False):
    show_onboarding()
    st.stop()

# ── Setup automático si no existen los archivos generados ─────────────────────
def setup_if_needed():
    predictions_path = os.path.join(BASE_DIR, 'students_predictions.csv')
    if not os.path.exists(predictions_path):
        with st.spinner("🦉 Preparando Atenea AI por primera vez... (puede tardar ~60 segundos)"):
            from generate_dataset import generate_dataset
            from train_model import train_model
            generate_dataset(output_dir=BASE_DIR)
            train_model(base_dir=BASE_DIR)
        st.success("✅ Modelo listo. ¡Bienvenido a Atenea AI!")
        st.rerun()

setup_if_needed()

# ── Estilos CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card.alto {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    }
    .metric-card.medio {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
    }
    .metric-card.bajo {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-number {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }
    .score-badge-alto  { color: #dc3545; font-weight: 700; }
    .score-badge-medio { color: #fd7e14; font-weight: 700; }
    .score-badge-bajo  { color: #28a745; font-weight: 700; }
    .stDataFrame { border-radius: 8px; }
    div[data-testid="stSidebarContent"] { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

FEATURES_ES = {
    'promedio_general': 'Promedio general',
    'materias_reprobadas': 'Materias reprobadas',
    'asistencia_pct': 'Asistencia (%)',
    'tareas_entregadas_pct': 'Tareas entregadas (%)',
    'participacion_clase': 'Participación en clase',
    'reportes_disciplinarios': 'Reportes disciplinarios',
    'llegadas_tarde': 'Llegadas tarde',
    'dias_suspension': 'Días de suspensión',
    'nivel_estres': 'Nivel de estrés',
    'motivacion_escolar': 'Motivación escolar',
    'apoyo_familiar': 'Apoyo familiar',
    'satisfaccion_escolar': 'Satisfacción escolar',
    'ansiedad_academica': 'Ansiedad académica',
}

RECOMENDACIONES = {
    'Alto': [
        "🚨 **Atención inmediata** — Contactar al tutor esta semana",
        "🧠 Derivar a valoración psicológica",
        "👨‍👩‍👧 Citar a padres/tutores para reunión urgente",
        "📋 Crear plan de intervención personalizado",
        "📚 Apoyo académico intensivo (asesorías)",
    ],
    'Medio': [
        "📅 **Seguimiento semanal** — Reunión con orientador",
        "📊 Monitorear asistencia y calificaciones cada 2 semanas",
        "💬 Sesión de mentoría con estudiante",
        "🏠 Verificar situación familiar básica",
    ],
    'Bajo': [
        "✅ **Monitoreo mensual** — Mantener estrategias actuales",
        "🌟 Reconocer logros académicos",
        "📈 Continuar seguimiento regular del grupo",
    ]
}

# ── Cargar datos ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'students_predictions.csv'))
    return df

@st.cache_resource
def load_explainer():
    path = os.path.join(BASE_DIR, 'explainer.pkl')
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def load_raw():
    path = os.path.join(BASE_DIR, 'students_dataset.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

df = load_data()
explainer_data = load_explainer()
df_raw = load_raw()

def get_nivel_clean(nivel_str):
    """Extrae 'Alto', 'Medio' o 'Bajo' del string con emoji"""
    if 'Alto' in nivel_str:  return 'Alto'
    if 'Medio' in nivel_str: return 'Medio'
    return 'Bajo'

df['nivel_clean'] = df['nivel_riesgo'].apply(get_nivel_clean)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🦉 Atenea AI")
    st.markdown("*Sistema de inteligencia educativa*")
    st.divider()
    st.markdown(f"🏫 **{st.session_state.get('school_name', '')}**")
    if st.button("Cerrar sesión", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["school_name"] = ""
        st.rerun()
    st.divider()

    buscar = st.text_input("🔍 Buscar alumno", placeholder="Nombre o ID...")

    nivel_filtro = st.multiselect(
        "Nivel de riesgo",
        options=['Alto', 'Medio', 'Bajo'],
        default=['Alto', 'Medio', 'Bajo']
    )

    grupos_disponibles = ['Todos'] + sorted(df['grupo'].unique().tolist())
    grupo_filtro = st.selectbox("Grupo", grupos_disponibles)

    semestres_disponibles = ['Todos'] + sorted(df['semestre'].unique().tolist())
    semestre_filtro = st.selectbox("Semestre", semestres_disponibles)

    st.divider()
    st.caption("Atenea AI v0.1 — MVP Demo")
    st.caption("Modelo: Stacking (RF + XGBoost + GB)")

# ── Filtrar datos ─────────────────────────────────────────────────────────────
df_filtered = df.copy()

if buscar:
    mask = (df_filtered['nombre'].str.contains(buscar, case=False, na=False) |
            df_filtered['id'].str.contains(buscar, case=False, na=False))
    df_filtered = df_filtered[mask]

if nivel_filtro:
    df_filtered = df_filtered[df_filtered['nivel_clean'].isin(nivel_filtro)]

if grupo_filtro != 'Todos':
    df_filtered = df_filtered[df_filtered['grupo'] == grupo_filtro]

if semestre_filtro != 'Todos':
    df_filtered = df_filtered[df_filtered['semestre'] == semestre_filtro]

df_filtered = df_filtered.sort_values('score_riesgo', ascending=False)

# ── Vista: Detalle de alumno ──────────────────────────────────────────────────
def show_student_detail(student_id):
    stu = df[df['id'] == student_id].iloc[0]
    nivel = get_nivel_clean(stu['nivel_riesgo'])
    score = stu['score_riesgo']

    st.markdown(f"### 👤 {stu['nombre']}")
    st.markdown(f"**ID:** {stu['id']} | **Grupo:** {stu['grupo']} | **Semestre:** {stu['semestre']}")
    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        color = {'Alto': '#dc3545', 'Medio': '#fd7e14', 'Bajo': '#28a745'}[nivel]
        st.markdown(f"""
        <div style="text-align:center; padding: 1.5rem; background: {color}15;
                    border: 3px solid {color}; border-radius: 16px; margin-bottom: 1rem;">
            <div style="font-size: 3.5rem; font-weight: 900; color: {color};">{score:.0f}</div>
            <div style="font-size: 1rem; color: {color}; font-weight: 600;">SCORE DE RIESGO</div>
            <div style="font-size: 1.4rem; margin-top: 0.5rem;">
                {'🔴' if nivel=='Alto' else '🟡' if nivel=='Medio' else '🟢'} Riesgo {nivel}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(score))

        if df_raw is not None:
            raw = df_raw[df_raw['id'] == student_id]
            if not raw.empty:
                r = raw.iloc[0]
                st.markdown("**Variables del alumno:**")
                data_display = {
                    'Promedio': f"{r['promedio_general']:.1f}/10",
                    'Asistencia': f"{r['asistencia_pct']:.0f}%",
                    'Mat. reprobadas': r['materias_reprobadas'],
                    'Nivel estrés': f"{r['nivel_estres']}/5",
                    'Motivación': f"{r['motivacion_escolar']}/5",
                    'Apoyo familiar': f"{r['apoyo_familiar']}/5",
                }
                for k, v in data_display.items():
                    st.markdown(f"- **{k}:** {v}")

    with col2:
        if explainer_data is not None and df_raw is not None:
            st.markdown("**Factores que más influyen (SHAP):**")
            idx = df_raw[df_raw['id'] == student_id].index
            if len(idx) > 0:
                sv = explainer_data['shap_values'][idx[0]]
                features = explainer_data['features']
                shap_df = pd.DataFrame({'feature': features, 'shap': sv})
                shap_df['feature_es'] = shap_df['feature'].map(FEATURES_ES)
                shap_df = shap_df.sort_values('shap', key=abs, ascending=True).tail(8)

                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#dc3545' if v > 0 else '#28a745' for v in shap_df['shap']]
                ax.barh(shap_df['feature_es'], shap_df['shap'], color=colors, height=0.6)
                ax.axvline(0, color='black', linewidth=0.8)
                ax.set_xlabel('Impacto en probabilidad de deserción', fontsize=9)
                ax.set_title('Factores de riesgo (SHAP values)', fontsize=10, fontweight='bold')
                ax.tick_params(axis='y', labelsize=8)
                ax.tick_params(axis='x', labelsize=8)
                red_patch = mpatches.Patch(color='#dc3545', label='Aumenta riesgo')
                green_patch = mpatches.Patch(color='#28a745', label='Reduce riesgo')
                ax.legend(handles=[red_patch, green_patch], fontsize=8, loc='lower right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.info("Cargando análisis SHAP...")
            top3 = stu['top3_factores']
            st.markdown(f"**Top 3 factores:** {top3}")

        st.markdown(f"**📋 Plan de intervención — Riesgo {nivel}:**")
        for rec in RECOMENDACIONES[nivel]:
            st.markdown(f"- {rec}")

    st.divider()

    if df_raw is not None:
        raw_stu = df_raw[df_raw['id'] == student_id]
        if not raw_stu.empty:
            reporte = raw_stu.copy()
            reporte['score_riesgo'] = score
            reporte['nivel_riesgo'] = nivel
            reporte['top3_factores'] = stu['top3_factores']
            csv_bytes = reporte.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exportar reporte del alumno (CSV)",
                data=csv_bytes,
                file_name=f"reporte_{student_id}.csv",
                mime='text/csv'
            )

# ── Vista principal ───────────────────────────────────────────────────────────
if 'selected_student' not in st.session_state:
    st.session_state.selected_student = None

if st.session_state.selected_student:
    if st.button("← Volver al Dashboard"):
        st.session_state.selected_student = None
        st.rerun()
    show_student_detail(st.session_state.selected_student)
else:
    st.markdown('<p class="main-header">🦉 Atenea AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema de inteligencia educativa — Dashboard</p>', unsafe_allow_html=True)

    total   = len(df_filtered)
    n_alto  = len(df_filtered[df_filtered['nivel_clean'] == 'Alto'])
    n_medio = len(df_filtered[df_filtered['nivel_clean'] == 'Medio'])
    n_bajo  = len(df_filtered[df_filtered['nivel_clean'] == 'Bajo'])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number">{total}</div>
            <div class="metric-label">Total alumnos</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card alto">
            <div class="metric-number">{n_alto}</div>
            <div class="metric-label">🔴 Riesgo Alto</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card medio">
            <div class="metric-number">{n_medio}</div>
            <div class="metric-label">🟡 Riesgo Medio</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card bajo">
            <div class="metric-number">{n_bajo}</div>
            <div class="metric-label">🟢 Riesgo Bajo</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"### 📋 Alumnos ({total})")
    st.caption("Haz clic en 'Ver detalle' para ver el análisis completo de cada alumno")

    if total == 0:
        st.info("No se encontraron alumnos con los filtros seleccionados.")
    else:
        color_map = {
            'Alto':  '#dc3545',
            'Medio': '#fd7e14',
            'Bajo':  '#28a745',
        }
        for _, row in df_filtered.iterrows():
            nivel_clean = get_nivel_clean(row['nivel_riesgo'])
            score = row['score_riesgo']
            color = color_map[nivel_clean]

            with st.container():
                cols = st.columns([3, 1.5, 2, 3, 1.5])
                with cols[0]:
                    st.markdown(f"**{row['nombre']}**")
                    st.caption(f"{row['id']} | Grupo {row['grupo']} | Sem. {row['semestre']}")
                with cols[1]:
                    st.markdown(f'<span style="color:{color}; font-size:1.6rem; font-weight:800;">{score:.0f}</span>', unsafe_allow_html=True)
                    st.caption("score")
                with cols[2]:
                    st.markdown(f'<span style="color:{color}; font-weight:700;">{row["nivel_riesgo"]}</span>', unsafe_allow_html=True)
                with cols[3]:
                    factores = row['top3_factores'] if pd.notna(row['top3_factores']) else "—"
                    for k, v in FEATURES_ES.items():
                        factores = factores.replace(k, v)
                    st.caption(f"📌 {factores}")
                with cols[4]:
                    if st.button("Ver detalle", key=f"btn_{row['id']}"):
                        st.session_state.selected_student = row['id']
                        st.rerun()
            st.divider()
