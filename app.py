"""
Atenea AI — Dashboard Streamlit con Supabase
Sistema de predicción de deserción escolar
Corre con: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import hashlib
from datetime import date

# ── Directorio base ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Atenea AI",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS global (tema oscuro profesional) ──────────────────────────────────────
st.markdown("""
<style>
    /* Fondo oscuro global */
    .stApp { background-color: #0f0f1a; color: #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #0d0d1a; border-right: 1px solid #1e1e3a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .block-container { padding-top: 1.5rem; }

    /* Cards métricas */
    .metric-card {
        background: linear-gradient(135deg, #1e1e3a 0%, #16213e 100%);
        border: 1px solid #2d2d5e;
        padding: 1.4rem 1.2rem;
        border-radius: 14px;
        text-align: center;
    }
    .metric-card.alto  { border-color: #dc3545; background: linear-gradient(135deg, #2a0d12 0%, #1a0810 100%); }
    .metric-card.medio { border-color: #fd7e14; background: linear-gradient(135deg, #2a1800 0%, #1a1000 100%); }
    .metric-card.bajo  { border-color: #28a745; background: linear-gradient(135deg, #0a2010 0%, #061408 100%); }
    .metric-number { font-size: 2.6rem; font-weight: 800; line-height: 1; color: #ffffff; }
    .metric-label  { font-size: 0.82rem; opacity: 0.85; margin-top: 0.3rem; color: #b0b8d0; }
    .metric-card.alto .metric-number  { color: #ff6b7a; }
    .metric-card.medio .metric-number { color: #ffa94d; }
    .metric-card.bajo .metric-number  { color: #69db7c; }

    /* Badges nivel riesgo */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .badge-alto  { background: #2a0d12; color: #ff6b7a; border: 1px solid #dc3545; }
    .badge-medio { background: #2a1800; color: #ffa94d; border: 1px solid #fd7e14; }
    .badge-bajo  { background: #0a2010; color: #69db7c; border: 1px solid #28a745; }

    /* Tabla alumnos */
    .student-row {
        background: #12122a;
        border: 1px solid #1e1e3a;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s;
    }
    .student-row:hover { border-color: #4f46e5; }

    /* Score bar */
    .score-bar-bg {
        background: #1e1e3a;
        border-radius: 4px;
        height: 6px;
        width: 100%;
        margin-top: 4px;
    }
    .score-bar-fill { height: 6px; border-radius: 4px; }

    /* Inputs y selectbox oscuros */
    .stTextInput input, .stSelectbox select, .stNumberInput input, .stTextArea textarea {
        background-color: #12122a !important;
        color: #e2e8f0 !important;
        border-color: #2d2d5e !important;
    }
    .stRadio label { color: #e2e8f0 !important; }

    /* Login card */
    .login-card {
        max-width: 440px;
        margin: 3rem auto;
        padding: 2.5rem 2rem;
        background: #12122a;
        border: 1px solid #2d2d5e;
        border-radius: 20px;
        text-align: center;
    }

    /* Perfil score card */
    .score-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        border: 3px solid;
    }
    .score-big { font-size: 4rem; font-weight: 900; line-height: 1; }

    /* Nav sidebar */
    .nav-item {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        cursor: pointer;
        margin-bottom: 0.2rem;
        color: #b0b8d0;
        transition: background 0.15s;
    }
    .nav-item.active { background: #1e1e3a; color: #ffffff; font-weight: 600; }
    .nav-item:hover  { background: #1a1a2e; }

    /* Divider */
    hr { border-color: #1e1e3a !important; }

    /* Recomendaciones */
    .rec-card {
        background: #12122a;
        border-left: 4px solid;
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
    }
    .rec-alto  { border-color: #dc3545; }
    .rec-medio { border-color: #fd7e14; }
    .rec-bajo  { border-color: #28a745; }

    h1, h2, h3, h4, h5 { color: #ffffff !important; }
    p, label, span { color: #e2e8f0; }
    .stCaption { color: #8892a4 !important; }
</style>
""", unsafe_allow_html=True)

# ── Supabase client ────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    from supabase import create_client
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)

def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

# ── Carga del modelo ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "model.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

@st.cache_resource
def load_explainer():
    exp_path = os.path.join(BASE_DIR, "explainer.pkl")
    if not os.path.exists(exp_path):
        return None
    return joblib.load(exp_path)

# ── Orden exacto de features del modelo ───────────────────────────────────────
MODEL_FEATURES = [
    'semestre', 'promedio_general', 'asistencia_pct', 'materias_reprobadas',
    'tareas_entregadas_pct', 'llegadas_tarde', 'reportes_disciplinarios',
    'motivacion_escolar', 'nivel_estres', 'apoyo_familiar',
    'satisfaccion_escolar', 'participacion_clases', 'apoyo_familiar_int'
]

FEATURES_ES = {
    'semestre': 'Semestre',
    'promedio_general': 'Promedio general',
    'asistencia_pct': 'Asistencia (%)',
    'materias_reprobadas': 'Materias reprobadas',
    'tareas_entregadas_pct': 'Tareas entregadas (%)',
    'llegadas_tarde': 'Llegadas tarde',
    'reportes_disciplinarios': 'Reportes disciplinarios',
    'motivacion_escolar': 'Motivación escolar',
    'nivel_estres': 'Nivel de estrés',
    'apoyo_familiar': 'Apoyo familiar',
    'satisfaccion_escolar': 'Satisfacción escolar',
    'participacion_clases': 'Participación en clase',
    'apoyo_familiar_int': 'Apoyo familiar (intensidad)',
}

def map_1_4_to_1_5(val: int) -> float:
    """Convierte escala 1-4 a escala 1-5 (el modelo fue entrenado en 1-5)."""
    mapping = {1: 1.0, 2: 2.33, 3: 3.67, 4: 5.0}
    return mapping.get(val, 2.5)

def build_feature_vector(
    semestre, promedio_general, asistencia_pct, materias_reprobadas,
    tareas_entregadas_pct, llegadas_tarde, reportes_disciplinarios,
    motivacion, nivel_estres, apoyo_familiar, sentido_pertenencia, expectativas_futuro
) -> np.ndarray:
    """
    Construye el vector de 13 features en el orden exacto que espera el modelo.
    Las variables psicoemocionales (1-4) se convierten a escala 1-5.
    """
    mot_5  = map_1_4_to_1_5(motivacion)
    est_5  = map_1_4_to_1_5(nivel_estres)
    apo_5  = map_1_4_to_1_5(apoyo_familiar)
    per_5  = map_1_4_to_1_5(sentido_pertenencia)
    # expectativas_futuro no tiene equivalente directo → promedio de motivación y sentido_pertenencia
    exp_avg = (mot_5 + per_5) / 2.0  # noqa: F841 — disponible por si el modelo cambia

    return np.array([[
        semestre,
        promedio_general,
        asistencia_pct,
        materias_reprobadas,
        tareas_entregadas_pct,
        llegadas_tarde,
        reportes_disciplinarios,
        mot_5,          # motivacion_escolar
        est_5,          # nivel_estres
        apo_5,          # apoyo_familiar
        per_5,          # satisfaccion_escolar
        mot_5,          # participacion_clases (mismo valor que motivacion)
        apo_5,          # apoyo_familiar_int (mismo valor que apoyo_familiar)
    ]])

def predict_risk(feature_vector: np.ndarray):
    """Devuelve (score 0-100, nivel, top3_factores_str)."""
    model = load_model()
    if model is None:
        return None, None, None

    proba = model.predict_proba(feature_vector)[0][1]
    score = round(proba * 100, 1)

    if score >= 55:
        nivel = "Alto"
    elif score >= 30:
        nivel = "Medio"
    else:
        nivel = "Bajo"

    # Top 3 factores via SHAP
    top3_str = ""
    explainer_data = load_explainer()
    if explainer_data is not None:
        try:
            if hasattr(explainer_data, 'shap_values'):
                # objeto dict guardado por el script de entrenamiento
                sv = explainer_data['shap_values']
                # calcular SHAP para este vector usando el explainer almacenado
                top3_str = _top3_from_explainer_dict(feature_vector, explainer_data)
            elif callable(explainer_data):
                shap_vals = explainer_data(feature_vector)
                vals = np.abs(shap_vals.values[0])
                idxs = np.argsort(vals)[::-1][:3]
                top3_str = ", ".join([FEATURES_ES.get(MODEL_FEATURES[i], MODEL_FEATURES[i]) for i in idxs])
        except Exception:
            top3_str = ""

    return score, nivel, top3_str

def _top3_from_explainer_dict(feature_vector: np.ndarray, explainer_data) -> str:
    """Extrae top 3 factores del explainer guardado como dict."""
    try:
        import shap as shap_lib
        # El explainer puede ser un TreeExplainer o un dict con shap_values
        if isinstance(explainer_data, dict):
            # Fallback: usar correlaciones simples entre la nueva observación y shap_values históricos
            return ""
        else:
            shap_vals = explainer_data.shap_values(feature_vector)
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]  # clase positiva
            else:
                sv = shap_vals[0]
            idxs = np.argsort(np.abs(sv))[::-1][:3]
            return ", ".join([FEATURES_ES.get(MODEL_FEATURES[i], MODEL_FEATURES[i]) for i in idxs])
    except Exception:
        return ""

# ── Regla educativa de emergencia ─────────────────────────────────────────────
def _apply_emergency_rule(score, nivel, promedio_general, asistencia_pct,
                           motivacion, apoyo_familiar, nivel_estres,
                           sentido_pertenencia, expectativas_futuro):
    """
    Override basado en umbrales educativos reales (México).
    promedio ≤ 6 ya es reprobatorio; asistencia ≤ 60% es alarma.
    Considera las 5 variables psicoemocionales en su totalidad.
    """
    adversos_psico = 0
    if motivacion <= 2:           adversos_psico += 1
    if apoyo_familiar <= 2:       adversos_psico += 1
    if nivel_estres >= 3:         adversos_psico += 1
    if sentido_pertenencia <= 2:  adversos_psico += 1
    if expectativas_futuro <= 2:  adversos_psico += 1

    promedio_reprobatorio = promedio_general <= 6.0
    asistencia_critica    = asistencia_pct <= 60.0
    asistencia_muy_baja   = asistencia_pct <= 50.0
    psico_mayoria_adversa = adversos_psico >= 3

    if promedio_reprobatorio and asistencia_critica and psico_mayoria_adversa:
        nivel = "Alto"
        score = max(score, 70.0)
    elif promedio_reprobatorio and asistencia_muy_baja:
        nivel = "Alto"
        score = max(score, 65.0)
    elif promedio_reprobatorio and adversos_psico >= 2:
        if nivel == "Bajo":
            nivel = "Medio"
            score = max(score, 35.0)

    return score, nivel

# ── RECOMENDACIONES ────────────────────────────────────────────────────────────
RECOMENDACIONES = {
    'Alto': [
        "🚨 **Atención inmediata** — Contactar al tutor esta semana",
        "🧠 Derivar a valoración psicológica / orientación vocacional",
        "👨‍👩‍👧 Citar a padres o tutores para reunión urgente",
        "📋 Crear plan de intervención personalizado con metas semanales",
        "📚 Inscribir en programa de apoyo académico (asesorías)",
    ],
    'Medio': [
        "📅 **Seguimiento quincenal** — Reunión breve con orientador",
        "📊 Monitorear asistencia y calificaciones cada 2 semanas",
        "💬 Sesión de mentoría entre pares o con un docente de confianza",
        "🏠 Verificar situación de apoyo en casa",
    ],
    'Bajo': [
        "✅ **Monitoreo mensual** — Mantener estrategias actuales",
        "🌟 Reconocer logros académicos públicamente",
    ],
}

# ── Setup automático ───────────────────────────────────────────────────────────
def setup_if_needed():
    """Genera dataset y entrena modelo si no existen los archivos."""
    model_path = os.path.join(BASE_DIR, "model.pkl")
    if not os.path.exists(model_path):
        with st.spinner("🦉 Iniciando sistema por primera vez... esto tarda ~2 minutos"):
            try:
                from generate_dataset import generate_dataset
                from train_model import train_model
                generate_dataset(output_dir=BASE_DIR)
                train_model(base_dir=BASE_DIR)
            except Exception as e:
                st.error(f"Error al inicializar el sistema: {e}")
                st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PANTALLA DE AUTH
# ══════════════════════════════════════════════════════════════════════════════

def show_auth():
    st.markdown("""
    <div class="login-card">
        <div style="font-size:4rem; margin-bottom:0.3rem;">🦉</div>
        <h1 style="font-size:2rem; font-weight:800; color:#a78bfa; margin:0;">Atenea AI</h1>
        <p style="color:#8892a4; margin-top:0.3rem;">Sistema de Inteligencia Educativa</p>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        tab_login, tab_register = st.tabs(["Iniciar sesión", "Crear cuenta"])

        # ── TAB: LOGIN ─────────────────────────────────────────────────────────
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            email_l = st.text_input("Email", key="login_email", placeholder="tu@email.com")
            pwd_l   = st.text_input("Contraseña", type="password", key="login_pwd")
            if st.button("Entrar", use_container_width=True, type="primary", key="btn_login"):
                if not email_l or not pwd_l:
                    st.error("Completa todos los campos.")
                else:
                    _do_login(email_l.strip().lower(), pwd_l)

        # ── TAB: REGISTRO ──────────────────────────────────────────────────────
        with tab_register:
            st.markdown("<br>", unsafe_allow_html=True)
            r_nombre    = st.text_input("Nombre", key="reg_nombre")
            r_apellidos = st.text_input("Apellidos", key="reg_apellidos")
            r_email     = st.text_input("Email", key="reg_email", placeholder="tu@email.com")
            r_escuela   = st.text_input("Nombre de tu escuela", key="reg_escuela")
            r_rol       = st.selectbox("Rol", ["maestro", "directivo"], key="reg_rol")
            r_grupos    = ""
            if r_rol == "maestro":
                r_grupos = st.text_input("Grupos asignados (ej: 3A, 3B)", key="reg_grupos")
            r_pwd1 = st.text_input("Contraseña", type="password", key="reg_pwd1")
            r_pwd2 = st.text_input("Confirmar contraseña", type="password", key="reg_pwd2")
            if st.button("Crear cuenta", use_container_width=True, type="primary", key="btn_register"):
                _do_register(
                    r_nombre.strip(), r_apellidos.strip(),
                    r_email.strip().lower(), r_escuela.strip(),
                    r_rol, r_grupos.strip(), r_pwd1, r_pwd2
                )

def _do_login(email: str, pwd: str):
    sb = get_supabase()
    try:
        res = sb.table("usuarios").select("*").eq("email", email).eq("password_hash", hash_password(pwd)).execute()
        if not res.data:
            st.error("Email o contraseña incorrectos.")
            return
        u = res.data[0]
        # Obtener nombre de escuela
        escuela_nombre = ""
        if u.get("escuela_id"):
            esc = sb.table("escuelas").select("nombre").eq("id", u["escuela_id"]).execute()
            if esc.data:
                escuela_nombre = esc.data[0]["nombre"]
        st.session_state.update({
            "authenticated": True,
            "user_id": u["id"],
            "user_name": f"{u['nombre']} {u['apellidos']}",
            "user_rol": u["rol"],
            "escuela_id": u.get("escuela_id"),
            "escuela_nombre": escuela_nombre,
            "grupos_asignados": u.get("grupos_asignados", ""),
            "page": "dashboard",
        })
        st.rerun()
    except Exception as e:
        st.error(f"Error al conectar con la base de datos: {e}")

def _do_register(nombre, apellidos, email, escuela_nombre, rol, grupos, pwd1, pwd2):
    if not all([nombre, apellidos, email, escuela_nombre, pwd1]):
        st.error("Completa todos los campos obligatorios.")
        return
    if pwd1 != pwd2:
        st.error("Las contraseñas no coinciden.")
        return
    if len(pwd1) < 6:
        st.error("La contraseña debe tener al menos 6 caracteres.")
        return

    sb = get_supabase()
    try:
        # Verificar email duplicado
        existe = sb.table("usuarios").select("id").eq("email", email).execute()
        if existe.data:
            st.error("Ya existe una cuenta con ese email.")
            return

        # Buscar o crear escuela
        esc = sb.table("escuelas").select("id").eq("nombre", escuela_nombre).execute()
        if esc.data:
            escuela_id = esc.data[0]["id"]
        else:
            nueva_esc = sb.table("escuelas").insert({"nombre": escuela_nombre}).execute()
            escuela_id = nueva_esc.data[0]["id"]

        # Insertar usuario
        sb.table("usuarios").insert({
            "nombre": nombre,
            "apellidos": apellidos,
            "email": email,
            "password_hash": hash_password(pwd1),
            "rol": rol,
            "escuela_id": escuela_id,
            "grupos_asignados": grupos if rol == "maestro" else None,
        }).execute()

        st.success("¡Cuenta creada! Ya puedes iniciar sesión.")
    except Exception as e:
        st.error(f"Error al crear la cuenta: {e}")

setup_if_needed()

# ── GUARD: Auth ────────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated", False):
    show_auth()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR (cuando autenticado)
# ══════════════════════════════════════════════════════════════════════════════

def _nav_button(label: str, page_key: str):
    active = st.session_state.get("page") == page_key
    style = "background:#1e1e3a; color:#fff; font-weight:600;" if active else "color:#b0b8d0;"
    if st.button(label, use_container_width=True, key=f"nav_{page_key}"):
        st.session_state["page"] = page_key
        # Limpiar selección alumno al cambiar de página
        if page_key != "perfil":
            st.session_state.pop("alumno_id", None)
        st.rerun()

with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 0.2rem 0;">
        <span style="font-size:1.8rem;">🦉</span>
        <span style="font-size:1.3rem; font-weight:800; color:#a78bfa; margin-left:0.4rem;">Atenea AI</span>
    </div>
    """, unsafe_allow_html=True)
    esc = st.session_state.get("escuela_nombre", "")
    if esc:
        st.caption(f"🏫 {esc}")
    st.divider()

    _nav_button("📊 Dashboard", "dashboard")
    _nav_button("➕ Agregar alumno", "agregar")
    st.markdown('<span style="color:#4a5568; font-size:0.82rem; padding:0.5rem 1rem;">👥 Mis grupos <em>(próximamente)</em></span>', unsafe_allow_html=True)
    st.markdown('<span style="color:#4a5568; font-size:0.82rem; padding:0.5rem 1rem;">📋 Reportes <em>(próximamente)</em></span>', unsafe_allow_html=True)
    st.markdown('<span style="color:#4a5568; font-size:0.82rem; padding:0.5rem 1rem;">⚙️ Configuración <em>(próximamente)</em></span>', unsafe_allow_html=True)

    # Espaciador al fondo
    st.markdown("<br>" * 4, unsafe_allow_html=True)
    st.divider()

    rol_str   = st.session_state.get("user_rol", "").capitalize()
    grupos_str = st.session_state.get("grupos_asignados", "") or "—"
    st.markdown(f"""
    <div style="padding:0.5rem 0;">
        <div style="font-size:0.95rem; font-weight:600; color:#e2e8f0;">
            👤 {st.session_state.get('user_name', '')}
        </div>
        <div style="font-size:0.75rem; color:#8892a4; margin-top:2px;">
            {rol_str} · {grupos_str}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Cerrar sesión", use_container_width=True, key="logout_btn"):
        for key in ["authenticated", "user_id", "user_name", "user_rol",
                    "escuela_id", "escuela_nombre", "grupos_asignados", "page", "alumno_id"]:
            st.session_state.pop(key, None)
        st.rerun()

# ── Inicializar página por defecto ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "dashboard"

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS SUPABASE
# ══════════════════════════════════════════════════════════════════════════════

def get_alumnos_escuela() -> pd.DataFrame:
    sb = get_supabase()
    escuela_id = st.session_state.get("escuela_id")
    user_rol   = st.session_state.get("user_rol")
    user_id    = st.session_state.get("user_id")

    try:
        query = sb.table("alumnos").select(
            "id, nombre, apellidos, matricula, grupo, semestre, maestro_id, "
            "registros_riesgo(score_riesgo, nivel_riesgo, factores_principales, fecha)"
        ).eq("escuela_id", escuela_id).order("created_at", desc=True)

        if user_rol == "maestro":
            query = query.eq("maestro_id", user_id)

        res = query.execute()
        if not res.data:
            return pd.DataFrame()

        rows = []
        for a in res.data:
            registros = a.get("registros_riesgo") or []
            # El último registro (más reciente)
            ultimo = None
            if registros:
                ultimo = sorted(registros, key=lambda x: x.get("fecha", ""), reverse=True)[0]

            rows.append({
                "id": a["id"],
                "nombre": f"{a['nombre']} {a['apellidos']}",
                "matricula": a["matricula"],
                "grupo": a["grupo"],
                "semestre": a["semestre"],
                "score_riesgo": ultimo["score_riesgo"] if ultimo else None,
                "nivel_riesgo": ultimo["nivel_riesgo"] if ultimo else None,
                "factores": ultimo["factores_principales"] if ultimo else None,
                "fecha_analisis": ultimo["fecha"] if ultimo else None,
            })

        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error al cargar alumnos: {e}")
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    st.markdown("## 📊 Dashboard")

    df = get_alumnos_escuela()

    # Métricas
    total  = len(df)
    n_alto  = len(df[df["nivel_riesgo"] == "Alto"]) if total > 0 else 0
    n_medio = len(df[df["nivel_riesgo"] == "Medio"]) if total > 0 else 0
    n_bajo  = len(df[df["nivel_riesgo"] == "Bajo"]) if total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{total}</div><div class="metric-label">Total alumnos</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card alto"><div class="metric-number">{n_alto}</div><div class="metric-label">🔴 Riesgo Alto</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card medio"><div class="metric-number">{n_medio}</div><div class="metric-label">🟡 Riesgo Medio</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card bajo"><div class="metric-number">{n_bajo}</div><div class="metric-label">🟢 Riesgo Bajo</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if total == 0:
        st.info("Aún no hay alumnos registrados. Ve a **Agregar alumno** para comenzar.")
        return

    # Filtros
    col_bus, col_grupo, col_sem, col_nivel = st.columns([3, 2, 2, 2])
    with col_bus:
        buscar = st.text_input("🔍 Buscar", placeholder="Nombre o matrícula...")
    with col_grupo:
        grupos = ["Todos"] + sorted(df["grupo"].dropna().unique().tolist())
        grupo_f = st.selectbox("Grupo", grupos)
    with col_sem:
        sems = ["Todos"] + sorted(df["semestre"].dropna().unique().tolist())
        sem_f = st.selectbox("Semestre", sems)
    with col_nivel:
        nivel_f = st.multiselect("Nivel de riesgo", ["Alto", "Medio", "Bajo"], default=["Alto", "Medio", "Bajo"])

    # Aplicar filtros
    df_f = df.copy()
    if buscar:
        mask = (df_f["nombre"].str.contains(buscar, case=False, na=False) |
                df_f["matricula"].str.contains(buscar, case=False, na=False))
        df_f = df_f[mask]
    if grupo_f != "Todos":
        df_f = df_f[df_f["grupo"] == grupo_f]
    if sem_f != "Todos":
        df_f = df_f[df_f["semestre"] == sem_f]
    if nivel_f:
        df_f = df_f[df_f["nivel_riesgo"].isin(nivel_f) | df_f["nivel_riesgo"].isna()]

    # Ordenar por score desc (alumnos sin score al final)
    df_f = df_f.sort_values("score_riesgo", ascending=False, na_position="last")

    st.markdown(f"### 📋 Alumnos ({len(df_f)})")
    st.caption("Haz clic en 'Ver perfil' para ver el análisis completo del alumno.")

    if len(df_f) == 0:
        st.info("No hay alumnos con los filtros seleccionados.")
        return

    COLOR_MAP = {"Alto": "#dc3545", "Medio": "#fd7e14", "Bajo": "#28a745"}

    for _, row in df_f.iterrows():
        nivel = row["nivel_riesgo"] or "—"
        score = row["score_riesgo"]
        color = COLOR_MAP.get(nivel, "#6c757d")
        score_display = f"{score:.0f}" if score is not None else "—"
        badge_class  = f"badge-{nivel.lower()}" if nivel in COLOR_MAP else ""
        bar_pct      = int(score) if score is not None else 0
        factores     = row["factores"] or "Sin análisis"
        fecha_str    = row["fecha_analisis"] or "—"

        cols = st.columns([3, 1.5, 2, 3, 1.5])
        with cols[0]:
            st.markdown(f"**{row['nombre']}**")
            st.caption(f"Matrícula: {row['matricula']} | Grupo {row['grupo']} | Sem. {row['semestre']}")
        with cols[1]:
            st.markdown(
                f'<span style="font-size:1.7rem; font-weight:900; color:{color};">{score_display}</span>',
                unsafe_allow_html=True
            )
            if score is not None:
                st.markdown(
                    f'<div class="score-bar-bg"><div class="score-bar-fill" '
                    f'style="width:{bar_pct}%; background:{color};"></div></div>',
                    unsafe_allow_html=True
                )
        with cols[2]:
            if badge_class:
                st.markdown(f'<span class="badge {badge_class}">{nivel}</span>', unsafe_allow_html=True)
            else:
                st.caption("Sin análisis")
        with cols[3]:
            st.caption(f"📌 {factores}")
            st.caption(f"🗓 {fecha_str}")
        with cols[4]:
            if st.button("Ver perfil →", key=f"ver_{row['id']}"):
                st.session_state["page"] = "perfil"
                st.session_state["alumno_id"] = row["id"]
                st.rerun()
        st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: AGREGAR ALUMNO
# ══════════════════════════════════════════════════════════════════════════════

PLANTILLA_COLS = [
    "nombre", "apellidos", "matricula", "grupo", "semestre",
    "promedio_general", "asistencia_pct", "materias_reprobadas",
    "tareas_entregadas_pct", "llegadas_tarde", "reportes_disciplinarios",
    "motivacion", "apoyo_familiar", "nivel_estres",
    "sentido_pertenencia", "expectativas_futuro"
]

def _analizar_y_mostrar(
    nombre, apellidos, matricula, grupo, semestre,
    promedio_general, asistencia_pct, materias_reprobadas,
    tareas_entregadas_pct, llegadas_tarde, reportes_disciplinarios,
    motivacion, apoyo_familiar, nivel_estres, sentido_pertenencia, expectativas_futuro,
    guardar=True
):
    """Corre el modelo y opcionalmente guarda en Supabase. Devuelve (score, nivel, top3)."""
    fv = build_feature_vector(
        semestre, promedio_general, asistencia_pct, materias_reprobadas,
        tareas_entregadas_pct, llegadas_tarde, reportes_disciplinarios,
        motivacion, nivel_estres, apoyo_familiar, sentido_pertenencia, expectativas_futuro
    )

    score, nivel, top3 = predict_risk(fv)
    if score is None:
        st.error("El modelo no está disponible. Asegúrate de que model.pkl existe en el directorio.")
        return None, None, None

    score, nivel = _apply_emergency_rule(
        score, nivel, promedio_general, asistencia_pct,
        motivacion, apoyo_familiar, nivel_estres,
        sentido_pertenencia, expectativas_futuro
    )

    if guardar:
        sb = get_supabase()
        escuela_id = st.session_state.get("escuela_id")
        user_id    = st.session_state.get("user_id")

        try:
            # Buscar alumno existente por matrícula + escuela
            existe = sb.table("alumnos").select("id").eq("matricula", matricula).eq("escuela_id", escuela_id).execute()
            if existe.data:
                alumno_id = existe.data[0]["id"]
                # Actualizar datos básicos
                sb.table("alumnos").update({
                    "nombre": nombre, "apellidos": apellidos,
                    "grupo": grupo, "semestre": semestre,
                    "updated_at": date.today().isoformat()
                }).eq("id", alumno_id).execute()
            else:
                nuevo = sb.table("alumnos").insert({
                    "nombre": nombre, "apellidos": apellidos,
                    "matricula": matricula, "grupo": grupo,
                    "semestre": semestre, "escuela_id": escuela_id,
                    "maestro_id": user_id
                }).execute()
                alumno_id = nuevo.data[0]["id"]

            # Insertar registro de riesgo
            sb.table("registros_riesgo").insert({
                "alumno_id": alumno_id,
                "fecha": date.today().isoformat(),
                "promedio_general": promedio_general,
                "asistencia_pct": asistencia_pct,
                "materias_reprobadas": materias_reprobadas,
                "tareas_entregadas_pct": tareas_entregadas_pct,
                "llegadas_tarde": llegadas_tarde,
                "reportes_disciplinarios": reportes_disciplinarios,
                "motivacion": motivacion,
                "apoyo_familiar": apoyo_familiar,
                "nivel_estres": nivel_estres,
                "sentido_pertenencia": sentido_pertenencia,
                "expectativas_futuro": expectativas_futuro,
                "score_riesgo": score,
                "nivel_riesgo": nivel,
                "factores_principales": top3,
            }).execute()

            st.session_state["ultimo_alumno_id"] = alumno_id
        except Exception as e:
            st.error(f"Error al guardar en Supabase: {e}")
            return score, nivel, top3

    return score, nivel, top3

def _show_resultado(score, nivel, top3, nombre_alumno=""):
    COLOR_MAP = {"Alto": "#dc3545", "Medio": "#fd7e14", "Bajo": "#28a745"}
    color = COLOR_MAP.get(nivel, "#6c757d")
    emoji = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢"}.get(nivel, "")

    st.markdown("---")
    st.markdown("### Resultado del análisis")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
        <div class="score-card" style="border-color:{color}; background:{color}18;">
            <div class="score-big" style="color:{color};">{score:.0f}</div>
            <div style="color:{color}; font-weight:700; margin-top:0.4rem;">SCORE DE RIESGO</div>
            <div style="font-size:1.3rem; margin-top:0.5rem;">{emoji} Riesgo {nivel}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if top3:
            st.markdown("**Top factores de riesgo:**")
            for f in top3.split(", "):
                st.markdown(f"- {f}")
        st.markdown(f"**Recomendaciones ({nivel}):**")
        for rec in RECOMENDACIONES.get(nivel, []):
            rec_class = f"rec-{nivel.lower()}"
            st.markdown(f'<div class="rec-card {rec_class}">{rec}</div>', unsafe_allow_html=True)

    if st.session_state.get("ultimo_alumno_id"):
        if st.button("Ver perfil completo →", type="primary"):
            st.session_state["page"] = "perfil"
            st.session_state["alumno_id"] = st.session_state["ultimo_alumno_id"]
            st.rerun()

def page_agregar():
    st.markdown("## ➕ Agregar alumno")

    tab_manual, tab_archivo = st.tabs(["📝 Captura manual", "📂 Carga de archivo"])

    # ── TAB 1: CAPTURA MANUAL ──────────────────────────────────────────────────
    with tab_manual:
        st.markdown("#### 1. Datos de identificación")
        c1, c2 = st.columns(2)
        with c1:
            nombre     = st.text_input("Nombre", key="m_nombre")
            matricula  = st.text_input("Matrícula", key="m_matricula")
            semestre   = st.number_input("Semestre / Año", min_value=1, max_value=6, value=1, step=1, key="m_semestre", help="Semestre o año escolar que cursa el alumno (ej: 3, 4°, etc.)")
        with c2:
            apellidos  = st.text_input("Apellidos", key="m_apellidos")
            grupo      = st.text_input("Grupo (ej: 3A)", key="m_grupo")

        st.markdown("#### 2. Variables académicas")
        c3, c4 = st.columns(2)
        with c3:
            promedio        = st.number_input("Promedio general (0–10)", min_value=0.0, max_value=10.0, value=7.0, step=0.1, key="m_promedio")
            mat_reprobadas  = st.number_input("Materias reprobadas", min_value=0, max_value=20, value=0, key="m_mat_rep")
            llegadas_tarde  = st.number_input("Llegadas tarde en el mes", min_value=0, max_value=50, value=0, key="m_llegadas")
        with c4:
            asistencia      = st.number_input("Asistencia (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.5, key="m_asistencia")
            tareas_pct      = st.number_input("Tareas entregadas (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.5, key="m_tareas")
            reportes_disc   = st.number_input("Reportes disciplinarios", min_value=0, max_value=20, value=0, key="m_reportes")

        st.markdown("#### 3. 💬 Seguimiento de Bienestar Escolar")
        st.info("Realiza estas preguntas al alumno en una conversación de 5 minutos.")

        OPTS_MOT = {1: "Casi nunca", 2: "A veces", 3: "Frecuentemente", 4: "Siempre"}
        OPTS_APO = {1: "Sin apoyo", 2: "Poco", 3: "Suficiente", 4: "Mucho"}
        OPTS_EST = {1: "Nunca", 2: "Raramente", 3: "Con frecuencia", 4: "Muy seguido"}
        OPTS_PER = {1: "Para nada", 2: "Poco", 3: "Bastante", 4: "Totalmente"}
        OPTS_EXP = {1: "No lo sabe", 2: "Vagamente", 3: "Tiene ideas", 4: "Muy claro"}

        def radio_1_4(label, pregunta, opciones_dict, key):
            st.markdown(f"**{label}:** *{pregunta}*")
            opts = list(opciones_dict.values())
            sel = st.radio("", opts, horizontal=True, key=key, label_visibility="collapsed")
            return [k for k, v in opciones_dict.items() if v == sel][0]

        motivacion_val = radio_1_4(
            "Motivación",
            "¿Con qué frecuencia muestra interés en aprender?",
            OPTS_MOT, "m_mot"
        )
        apoyo_fam_val = radio_1_4(
            "Apoyo familiar",
            "¿Tiene apoyo en casa para estudiar?",
            OPTS_APO, "m_apo"
        )
        estres_val = radio_1_4(
            "Estrés",
            "¿Con qué frecuencia se siente estresado por la escuela?",
            OPTS_EST, "m_est"
        )
        pertenencia_val = radio_1_4(
            "Pertenencia",
            "¿Se siente parte del grupo y la escuela?",
            OPTS_PER, "m_per"
        )
        expectativas_val = radio_1_4(
            "Expectativas",
            "¿Tiene claro qué quiere hacer al terminar el bachillerato?",
            OPTS_EXP, "m_exp"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Analizar riesgo", type="primary", use_container_width=True, key="btn_analizar"):
            if not nombre or not apellidos or not matricula or not grupo:
                st.error("Completa los datos de identificación del alumno.")
            else:
                with st.spinner("Analizando..."):
                    score, nivel, top3 = _analizar_y_mostrar(
                        nombre, apellidos, matricula, grupo, int(semestre),
                        promedio, asistencia, int(mat_reprobadas),
                        tareas_pct, int(llegadas_tarde), int(reportes_disc),
                        motivacion_val, apoyo_fam_val, estres_val,
                        pertenencia_val, expectativas_val
                    )
                if score is not None:
                    _show_resultado(score, nivel, top3, nombre)

    # ── TAB 2: CARGA DE ARCHIVO ────────────────────────────────────────────────
    with tab_archivo:
        st.markdown("#### Carga masiva desde CSV / Excel")

        # Botón descargar plantilla
        plantilla_df = pd.DataFrame(columns=PLANTILLA_COLS)
        csv_plantilla = plantilla_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Descargar plantilla CSV",
            data=csv_plantilla,
            file_name="plantilla_alumnos.csv",
            mime="text/csv"
        )

        archivo = st.file_uploader("Sube tu archivo (CSV o Excel)", type=["csv", "xlsx", "xls"])
        if archivo:
            try:
                if archivo.name.endswith(".csv"):
                    df_up = pd.read_csv(archivo)
                else:
                    df_up = pd.read_excel(archivo)
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")
                return

            required = {"nombre", "apellidos", "matricula", "grupo", "semestre",
                        "promedio_general", "asistencia_pct", "materias_reprobadas",
                        "tareas_entregadas_pct", "llegadas_tarde", "reportes_disciplinarios",
                        "motivacion", "apoyo_familiar", "nivel_estres",
                        "sentido_pertenencia", "expectativas_futuro"}
            missing = required - set(df_up.columns)
            if missing:
                st.error(f"Columnas faltantes en el archivo: {', '.join(missing)}")
                return

            st.markdown("**Vista previa:**")
            st.dataframe(df_up.head(5), use_container_width=True)

            if st.button("🔍 Procesar y analizar todos", type="primary"):
                resultados = []
                progress = st.progress(0)
                for i, row in df_up.iterrows():
                    try:
                        fv = build_feature_vector(
                            int(row["semestre"]), float(row["promedio_general"]),
                            float(row["asistencia_pct"]), int(row["materias_reprobadas"]),
                            float(row["tareas_entregadas_pct"]), int(row["llegadas_tarde"]),
                            int(row["reportes_disciplinarios"]), int(row["motivacion"]),
                            int(row["nivel_estres"]), int(row["apoyo_familiar"]),
                            int(row["sentido_pertenencia"]), int(row["expectativas_futuro"])
                        )
                        score, nivel, top3 = predict_risk(fv)
                    except Exception:
                        score, nivel, top3 = None, "Error", ""

                    resultados.append({**row.to_dict(), "score_riesgo": score, "nivel_riesgo": nivel, "factores": top3})
                    progress.progress((i + 1) / len(df_up))

                df_res = pd.DataFrame(resultados)
                st.markdown("**Resultados:**")
                st.dataframe(df_res[["nombre", "apellidos", "matricula", "grupo", "semestre", "score_riesgo", "nivel_riesgo"]], use_container_width=True)

                if st.button("💾 Guardar todos en Supabase"):
                    sb = get_supabase()
                    escuela_id = st.session_state.get("escuela_id")
                    user_id    = st.session_state.get("user_id")
                    guardados = 0
                    for _, row in df_res.iterrows():
                        if row.get("nivel_riesgo") == "Error":
                            continue
                        try:
                            existe = sb.table("alumnos").select("id").eq("matricula", str(row["matricula"])).eq("escuela_id", escuela_id).execute()
                            if existe.data:
                                alumno_id = existe.data[0]["id"]
                            else:
                                nuevo = sb.table("alumnos").insert({
                                    "nombre": row["nombre"], "apellidos": row["apellidos"],
                                    "matricula": str(row["matricula"]), "grupo": row["grupo"],
                                    "semestre": int(row["semestre"]), "escuela_id": escuela_id,
                                    "maestro_id": user_id
                                }).execute()
                                alumno_id = nuevo.data[0]["id"]

                            sb.table("registros_riesgo").insert({
                                "alumno_id": alumno_id,
                                "fecha": date.today().isoformat(),
                                "promedio_general": float(row["promedio_general"]),
                                "asistencia_pct": float(row["asistencia_pct"]),
                                "materias_reprobadas": int(row["materias_reprobadas"]),
                                "tareas_entregadas_pct": float(row["tareas_entregadas_pct"]),
                                "llegadas_tarde": int(row["llegadas_tarde"]),
                                "reportes_disciplinarios": int(row["reportes_disciplinarios"]),
                                "motivacion": int(row["motivacion"]),
                                "apoyo_familiar": int(row["apoyo_familiar"]),
                                "nivel_estres": int(row["nivel_estres"]),
                                "sentido_pertenencia": int(row["sentido_pertenencia"]),
                                "expectativas_futuro": int(row["expectativas_futuro"]),
                                "score_riesgo": row["score_riesgo"],
                                "nivel_riesgo": row["nivel_riesgo"],
                                "factores_principales": row["factores"],
                            }).execute()
                            guardados += 1
                        except Exception:
                            pass
                    st.success(f"✅ {guardados} alumnos guardados en Supabase.")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: PERFIL DEL ALUMNO
# ══════════════════════════════════════════════════════════════════════════════

def page_perfil():
    alumno_id = st.session_state.get("alumno_id")
    if not alumno_id:
        st.warning("No se seleccionó ningún alumno.")
        if st.button("← Volver al Dashboard"):
            st.session_state["page"] = "dashboard"
            st.rerun()
        return

    sb = get_supabase()
    try:
        alumno_res = sb.table("alumnos").select("*").eq("id", alumno_id).execute()
        if not alumno_res.data:
            st.error("Alumno no encontrado.")
            return
        alumno = alumno_res.data[0]

        registros_res = sb.table("registros_riesgo").select("*").eq("alumno_id", alumno_id).order("fecha", desc=True).execute()
        registros = registros_res.data or []
        hist = sb.table("registros_riesgo").select(
            "fecha,score_riesgo,nivel_riesgo,promedio_general,asistencia_pct,"
            "materias_reprobadas,tareas_entregadas_pct,llegadas_tarde,reportes_disciplinarios,"
            "motivacion,apoyo_familiar,nivel_estres,sentido_pertenencia,expectativas_futuro"
        ).eq("alumno_id", alumno_id).order("fecha").execute()
    except Exception as e:
        st.error(f"Error al cargar perfil: {e}")
        return

    ultimo = registros[0] if registros else None

    # ── HEADER ─────────────────────────────────────────────────────────────────
    if st.button("← Volver al Dashboard", key="back_perfil"):
        st.session_state["page"] = "dashboard"
        st.rerun()

    nombre_completo = f"{alumno['nombre']} {alumno['apellidos']}"
    c_title, c_btns = st.columns([3, 1])
    with c_title:
        st.markdown(f"## 👤 {nombre_completo}")
        st.caption(f"Matrícula: {alumno['matricula']} | Grupo: {alumno['grupo']} | Semestre: {alumno['semestre']}")
    with c_btns:
        if ultimo:
            export_data = {
                "nombre": nombre_completo,
                "matricula": alumno["matricula"],
                "grupo": alumno["grupo"],
                "semestre": alumno["semestre"],
                **{k: v for k, v in ultimo.items() if k not in ("id", "alumno_id")}
            }
            csv_exp = pd.DataFrame([export_data]).to_csv(index=False).encode("utf-8")
            st.download_button("📥 Exportar CSV", data=csv_exp, file_name=f"perfil_{alumno['matricula']}.csv", mime="text/csv")

    st.divider()

    if not ultimo:
        st.info("Este alumno aún no tiene análisis de riesgo. Ve a **Agregar alumno** para registrar sus datos.")
        return

    nivel = ultimo.get("nivel_riesgo", "—")
    score = ultimo.get("score_riesgo")
    top3  = ultimo.get("factores_principales", "")

    COLOR_MAP = {"Alto": "#dc3545", "Medio": "#fd7e14", "Bajo": "#28a745"}
    color = COLOR_MAP.get(nivel, "#6c757d")
    emoji = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢"}.get(nivel, "")

    # ── Tendencia (Cambio 4) ────────────────────────────────────────────────────
    tendencia_badge = ""
    if len(registros) >= 2:
        first_score_t = registros[-1].get("score_riesgo") or 0
        last_score_t  = registros[0].get("score_riesgo") or 0
        delta_t = last_score_t - first_score_t
        if delta_t > 5:
            tendencia_badge = "📈 En aumento"
        elif delta_t < -5:
            tendencia_badge = "📉 Mejorando"
        else:
            tendencia_badge = "↔️ Estable"

    col_left, col_right = st.columns([1, 2])

    with col_left:
        tendencia_html = (
            f'<div style="margin-top:0.6rem; font-size:0.85rem; font-weight:600; color:#b0b8d0;">{tendencia_badge}</div>'
            if tendencia_badge else ""
        )
        st.markdown(f"""
        <div class="score-card" style="border-color:{color}; background:{color}18;">
            <div class="score-big" style="color:{color};">{f"{score:.0f}" if score else '—'}</div>
            <div style="color:{color}; font-weight:700; font-size:0.9rem; margin-top:0.4rem;">SCORE DE RIESGO</div>
            <div style="font-size:1.2rem; margin-top:0.5rem;">{emoji} Riesgo {nivel}</div>
            <div style="color:#8892a4; font-size:0.75rem; margin-top:0.4rem;">Último análisis: {ultimo.get('fecha', '—')}</div>
            {tendencia_html}
        </div>
        """, unsafe_allow_html=True)

        if top3:
            st.markdown("<br>**Top factores de riesgo:**", unsafe_allow_html=True)
            for f in top3.split(", "):
                if f.strip():
                    st.markdown(f"- {f.strip()}")

    with col_right:
        st.markdown("**Variables del último análisis:**")

        VAR_LABELS = [
            ("promedio_general",      "Promedio general",       10.0),
            ("asistencia_pct",        "Asistencia (%)",         100.0),
            ("materias_reprobadas",   "Materias reprobadas",    20.0),
            ("tareas_entregadas_pct", "Tareas entregadas (%)",  100.0),
            ("llegadas_tarde",        "Llegadas tarde",         50.0),
            ("reportes_disciplinarios","Reportes disciplinarios",20.0),
        ]
        PSI_LABELS = [
            ("motivacion",          "Motivación",           4.0),
            ("apoyo_familiar",      "Apoyo familiar",       4.0),
            ("nivel_estres",        "Nivel de estrés",      4.0),
            ("sentido_pertenencia", "Sentido de pertenencia",4.0),
            ("expectativas_futuro", "Expectativas futuro",  4.0),
        ]

        st.markdown("*Académicas:*")
        for campo, label, maximo in VAR_LABELS:
            val = ultimo.get(campo)
            if val is not None:
                pct = min(float(val) / maximo, 1.0)
                bar_col = "#dc3545" if campo in ("materias_reprobadas", "llegadas_tarde", "reportes_disciplinarios") else "#4f46e5"
                st.markdown(
                    f'<div style="margin-bottom:6px;">'
                    f'<span style="font-size:0.8rem; color:#b0b8d0;">{label}: <b style="color:#fff;">{val}</b></span>'
                    f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{int(pct*100)}%; background:{bar_col};"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("*Bienestar escolar (1–4):*")
        for campo, label, maximo in PSI_LABELS:
            val = ultimo.get(campo)
            if val is not None:
                pct = float(val) / maximo
                bar_col = "#dc3545" if campo == "nivel_estres" else "#10b981"
                st.markdown(
                    f'<div style="margin-bottom:6px;">'
                    f'<span style="font-size:0.8rem; color:#b0b8d0;">{label}: <b style="color:#fff;">{val}/4</b></span>'
                    f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{int(pct*100)}%; background:{bar_col};"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    st.divider()

    # ── RECOMENDACIONES ─────────────────────────────────────────────────────────
    st.markdown(f"### 📋 Plan de intervención — Riesgo {nivel}")
    rec_class = f"rec-{nivel.lower()}"
    for rec in RECOMENDACIONES.get(nivel, []):
        st.markdown(f'<div class="rec-card {rec_class}">{rec}</div>', unsafe_allow_html=True)

    st.divider()

    # ── HISTORIAL ───────────────────────────────────────────────────────────────
    if len(registros) > 1:
        st.markdown("### 📈 Historial de análisis")
        hist_df = pd.DataFrame([
            {"Fecha": r.get("fecha", "—"), "Score": r.get("score_riesgo"), "Nivel": r.get("nivel_riesgo", "—")}
            for r in registros
        ])
        hist_df = hist_df.sort_values("Fecha")
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # ── EVOLUCIÓN LONGITUDINAL (Cambio 3) ──────────────────────────────────────
    if hist.data and len(hist.data) >= 2:
        st.markdown("---")
        st.markdown("### 📈 Evolución del riesgo")
        df_hist = pd.DataFrame(hist.data)
        df_hist["fecha"] = pd.to_datetime(df_hist["fecha"])
        df_hist = df_hist.sort_values("fecha")
        chart_data = df_hist.set_index("fecha")[["score_riesgo"]]
        st.line_chart(chart_data, use_container_width=True)

        first_score = df_hist["score_riesgo"].iloc[0]
        last_score  = df_hist["score_riesgo"].iloc[-1]
        delta = last_score - first_score
        if delta > 5:
            trend_msg   = f"⚠️ El riesgo **aumentó {delta:.0f} puntos** desde el primer registro."
            trend_color = "#dc3545"
        elif delta < -5:
            trend_msg   = f"✅ El riesgo **disminuyó {abs(delta):.0f} puntos** desde el primer registro."
            trend_color = "#28a745"
        else:
            trend_msg   = "↔️ El riesgo se ha **mantenido estable**."
            trend_color = "#fd7e14"
        st.markdown(f'<p style="color:{trend_color}; font-weight:600;">{trend_msg}</p>', unsafe_allow_html=True)

    # ── NUEVO ANÁLISIS (Cambio 3 Parte B) ──────────────────────────────────────
    st.markdown("---")
    with st.expander("➕ Registrar nuevo análisis", expanded=False):
        st.markdown("Actualiza las variables del alumno para registrar un nuevo punto de seguimiento.")

        ultimo_hist = hist.data[-1] if hist.data else {}

        with st.form("form_nuevo_analisis"):
            col1, col2, col3 = st.columns(3)
            with col1:
                na_promedio   = st.number_input("Promedio general", 0.0, 10.0,
                                  value=float(ultimo_hist.get("promedio_general", 7.0)), step=0.1)
                na_asistencia = st.number_input("Asistencia (%)", 0.0, 100.0,
                                  value=float(ultimo_hist.get("asistencia_pct", 80.0)), step=1.0)
                na_materias   = st.number_input("Materias reprobadas", 0, 10,
                                  value=int(ultimo_hist.get("materias_reprobadas", 0)))
            with col2:
                na_tareas     = st.number_input("Tareas entregadas (%)", 0.0, 100.0,
                                  value=float(ultimo_hist.get("tareas_entregadas_pct", 80.0)), step=1.0)
                na_tarde      = st.number_input("Llegadas tarde", 0, 30,
                                  value=int(ultimo_hist.get("llegadas_tarde", 0)))
                na_reportes   = st.number_input("Reportes disciplinarios", 0, 20,
                                  value=int(ultimo_hist.get("reportes_disciplinarios", 0)))
            with col3:
                OPTS_MOT = ["Casi nunca (1)", "Pocas veces (2)", "Frecuentemente (3)", "Siempre (4)"]
                OPTS_APO = ["Sin apoyo (1)", "Poco apoyo (2)", "Buen apoyo (3)", "Apoyo total (4)"]
                OPTS_EST = ["Casi nada (1)", "Algo (2)", "Bastante (3)", "Muy seguido (4)"]
                OPTS_PER = ["Para nada (1)", "Poco (2)", "Bastante (3)", "Totalmente (4)"]
                OPTS_EXP = ["No lo sabe (1)", "Tal vez (2)", "Probablemente (3)", "Seguro (4)"]

                _mot_idx = int(ultimo_hist.get("motivacion", 2)) - 1 if ultimo_hist.get("motivacion") else 1
                _apo_idx = int(ultimo_hist.get("apoyo_familiar", 2)) - 1 if ultimo_hist.get("apoyo_familiar") else 1
                _est_idx = int(ultimo_hist.get("nivel_estres", 2)) - 1 if ultimo_hist.get("nivel_estres") else 1
                _per_idx = int(ultimo_hist.get("sentido_pertenencia", 2)) - 1 if ultimo_hist.get("sentido_pertenencia") else 1
                _exp_idx = int(ultimo_hist.get("expectativas_futuro", 2)) - 1 if ultimo_hist.get("expectativas_futuro") else 1

                na_motivacion   = st.selectbox("Motivación", OPTS_MOT, index=_mot_idx)
                na_apoyo        = st.selectbox("Apoyo familiar", OPTS_APO, index=_apo_idx)
                na_estres       = st.selectbox("Nivel de estrés", OPTS_EST, index=_est_idx)
                na_pertenencia  = st.selectbox("Sentido de pertenencia", OPTS_PER, index=_per_idx)
                na_expectativas = st.selectbox("Expectativas futuro", OPTS_EXP, index=_exp_idx)

            submit_nuevo = st.form_submit_button("Guardar análisis", type="primary", use_container_width=True)

        if submit_nuevo:
            def parse_sel(s): return int(s[-2])

            al_data = sb.table("alumnos").select("semestre,grupo").eq("id", alumno_id).execute()
            al = al_data.data[0] if al_data.data else {}

            fv_n = build_feature_vector(
                al.get("semestre", 1), na_promedio, na_asistencia, na_materias,
                na_tareas, na_tarde, na_reportes,
                parse_sel(na_motivacion), parse_sel(na_estres),
                parse_sel(na_apoyo), parse_sel(na_pertenencia), parse_sel(na_expectativas)
            )
            score_n, nivel_n, top3_n = predict_risk(fv_n)
            score_n, nivel_n = _apply_emergency_rule(
                score_n, nivel_n, na_promedio, na_asistencia,
                parse_sel(na_motivacion), parse_sel(na_apoyo), parse_sel(na_estres),
                parse_sel(na_pertenencia), parse_sel(na_expectativas)
            )

            try:
                sb.table("registros_riesgo").insert({
                    "alumno_id": alumno_id,
                    "escuela_id": st.session_state.get("escuela_id"),
                    "fecha": date.today().isoformat(),
                    "promedio_general": na_promedio,
                    "asistencia_pct": na_asistencia,
                    "materias_reprobadas": na_materias,
                    "tareas_entregadas_pct": na_tareas,
                    "llegadas_tarde": na_tarde,
                    "reportes_disciplinarios": na_reportes,
                    "motivacion": parse_sel(na_motivacion),
                    "apoyo_familiar": parse_sel(na_apoyo),
                    "nivel_estres": parse_sel(na_estres),
                    "sentido_pertenencia": parse_sel(na_pertenencia),
                    "expectativas_futuro": parse_sel(na_expectativas),
                    "score_riesgo": score_n,
                    "nivel_riesgo": nivel_n,
                    "factores_principales": top3_n or "",
                }).execute()
                st.success(f"✅ Análisis guardado — Riesgo: {nivel_n} ({score_n:.0f}/100)")
                st.rerun()
            except Exception as e:
                st.error(f"Error al guardar: {e}")

    # ── BOTÓN INTERVENCIÓN ──────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("✅ Intervención realizada — registrar nota"):
        try:
            sb.table("registros_riesgo").insert({
                "alumno_id": alumno_id,
                "fecha": date.today().isoformat(),
                "score_riesgo": score,
                "nivel_riesgo": nivel,
                "factores_principales": f"[Intervención registrada] {top3}",
            }).execute()
            st.success("Nota de intervención registrada.")
        except Exception as e:
            st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# ROUTER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

page = st.session_state.get("page", "dashboard")

if page == "dashboard":
    page_dashboard()
elif page == "agregar":
    page_agregar()
elif page == "perfil":
    page_perfil()
else:
    page_dashboard()
