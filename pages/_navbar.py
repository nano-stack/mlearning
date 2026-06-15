"""Barra de navegación con pasos clicables para volver atrás."""
import streamlit as st

STEPS = [
    ("Tipo",        "select_type"),
    ("Modelo",      "select_model"),
    ("Datos",       "data"),
    ("Parámetros",  "params"),
    ("Resultados",  "results"),
]

MODEL_LABELS = {
    "linear_regression":   "Reg. Lineal",
    "logistic_regression": "Reg. Logística",
    "svm":    "SVM",
    "kmeans": "K-Means",
    "pca":    "PCA",
}


def navbar(active: int = -1):
    model_badge = ""
    if st.session_state.get("model_key"):
        lbl = MODEL_LABELS.get(st.session_state.model_key, st.session_state.model_key)
        model_badge = f'<span style="color:#f0a500;font-size:13px;margin-left:8px">· {lbl}</span>'

    algo_badge = ""
    if st.session_state.get("algo_type"):
        at = "Supervisado" if st.session_state.algo_type == "supervisado" else "No Supervisado"
        algo_badge = f'<span style="color:#a8c4e0;font-size:12px;margin-left:8px">({at})</span>'

    # Pasos como botones Streamlit reales (clicables)
    steps_html = ""
    for i, (label, _) in enumerate(STEPS):
        if i < active:
            css = "done"
            tip = f"Volver a {label}"
        elif i == active:
            css = "active"
            tip = ""
        else:
            css = ""
            tip = ""
        steps_html += f'<div class="ml-step {css}" title="{tip}">{label}</div>'

    st.markdown(f"""
    <div class="ml-navbar">
      <div>
        <span class="ml-navbar-brand">ML<span style="color:#f0a500">earning</span></span>
        {algo_badge}{model_badge}
      </div>
      <div class="ml-navbar-steps">{steps_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Barra de navegación inferior con botones clicables ───────────────────
    if active > 0:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        nav_cols = st.columns(len(STEPS) + 1)

        # Botón volver
        with nav_cols[0]:
            prev_label, prev_page = STEPS[active - 1]
            if st.button(f"← {prev_label}", key="nav_back_btn",
                         help=f"Volver a {prev_label}",
                         use_container_width=True):
                st.session_state.result = None
                st.session_state.plots  = None
                st.session_state.page   = prev_page
                st.rerun()

        # Botones de cada paso completado (para saltar)
        for i, (label, page) in enumerate(STEPS):
            with nav_cols[i + 1]:
                if i < active:
                    if st.button(label, key=f"nav_step_{i}",
                                 help=f"Ir a {label}",
                                 use_container_width=True):
                        st.session_state.result = None
                        st.session_state.plots  = None
                        st.session_state.page   = page
                        st.rerun()
                elif i == active:
                    st.markdown(
                        f'<div style="text-align:center;font-size:12px;font-weight:700;'
                        f'color:#f0a500;padding:6px 0;border-bottom:2px solid #f0a500">'
                        f'{label}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div style="text-align:center;font-size:12px;color:#5a7a9f;'
                        f'padding:6px 0">{label}</div>',
                        unsafe_allow_html=True
                    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
