"""
Pantalla 5 — Configuración de hiperparámetros con explicaciones didácticas.
"""
import streamlit as st
import numpy as np
from knowledge.models_knowledge import MODELS_KNOWLEDGE, HYPERPARAM_EXPLAINER
from pages._navbar import navbar


def render():
    navbar(active=3)

    model_key = st.session_state.model_key
    info = MODELS_KNOWLEDGE.get(model_key, {})

    st.markdown(f"""
    <div class="ml-section-header">
        Configuración de parámetros — {info.get('name', model_key)}
    </div>
    <div class="ml-section-sub">
        Configura cómo entrenará el modelo. Cada campo incluye una explicación de qué es y cómo afecta el resultado.
    </div>
    """, unsafe_allow_html=True)

    # ── Qué es un hiperparámetro ─────────────────────────────────────────────
    with st.expander("¿Qué es un hiperparámetro y qué NO es un hiperparámetro?"):
        he = HYPERPARAM_EXPLAINER
        st.markdown(f"""
        <div class="ml-info-box"><strong>Definición:</strong> {he['definition']}</div>
        <div class="ml-warn-box">{he['not_hyperparam'].replace(chr(10), '<br>')}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px">
            <div class="ml-info-box">
                <strong>SON hiperparámetros:</strong><br>
                {"<br>".join("• " + x for x in he['examples']['son_hiperparametros'])}
            </div>
            <div class="ml-warn-box">
                <strong>NO son hiperparámetros:</strong><br>
                {"<br>".join("• " + x for x in he['examples']['no_son_hiperparametros'])}
            </div>
        </div>
        <div class="ml-info-box" style="margin-top:12px">
            <strong>Cómo optimizarlos:</strong><br>
            {he['how_to_tune'].replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

    # ── Parámetros del modelo ─────────────────────────────────────────────────
    params_cfg = info.get("params", {})
    collected  = {}

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    for param_key, cfg in params_cfg.items():
        is_hyper = cfg.get("is_hyperparam", False)
        badge_html = (
            '<span class="ml-param-badge">Hiperparámetro</span>'
            if is_hyper else
            '<span class="ml-param-badge not-hyper">No es hiperparámetro</span>'
        )

        st.markdown(f"""
        <div class="ml-param-label">
            {cfg['label']}
            {badge_html}
        </div>
        """, unsafe_allow_html=True)

        # Widget según tipo
        ptype = cfg.get("type", "slider")

        if ptype == "slider":
            val = st.slider(
                label=param_key,
                min_value=float(cfg["min"]),
                max_value=float(cfg["max"]),
                value=float(cfg["default"]),
                step=float(cfg.get("step", 0.01)) if cfg.get("step") else 0.01,
                key=f"param_{param_key}",
                label_visibility="collapsed",
            )
            collected[param_key] = val

        elif ptype == "number":
            val = st.number_input(
                label=param_key,
                min_value=int(cfg["min"]),
                max_value=int(cfg["max"]),
                value=int(cfg["default"]),
                key=f"param_{param_key}",
                label_visibility="collapsed",
            )
            collected[param_key] = val

        elif ptype == "checkbox":
            val = st.checkbox(
                label=cfg["label"],
                value=cfg["default"],
                key=f"param_{param_key}",
            )
            collected[param_key] = val

        elif ptype == "select":
            val = st.selectbox(
                label=param_key,
                options=cfg["options"],
                index=cfg["options"].index(cfg["default"]) if cfg["default"] in cfg["options"] else 0,
                key=f"param_{param_key}",
                label_visibility="collapsed",
            )
            collected[param_key] = val

        # Info boxes
        cols_info = st.columns(2)
        with cols_info[0]:
            st.markdown(f"""
            <div class="ml-info-box">
                <strong>¿Qué es?</strong><br>{cfg.get('what_it_is', '—')}
            </div>
            """, unsafe_allow_html=True)
        with cols_info[1]:
            st.markdown(f"""
            <div class="ml-info-box">
                <strong>Efecto:</strong><br>{cfg.get('effect', '—')}<br><br>
                <strong>Regla práctica:</strong> {cfg.get('rule_of_thumb', '—')}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Métricas que se calcularán ────────────────────────────────────────────
    metrics_info = info.get("metrics", {})
    if metrics_info:
        with st.expander("Métricas que se calcularán"):
            for metric, desc in metrics_info.items():
                st.markdown(f"""
                <div class="ml-info-box"><strong>{metric}:</strong> {desc}</div>
                """, unsafe_allow_html=True)

    # ── Código preview ────────────────────────────────────────────────────────
    code_tpl = info.get("code_template", "")
    if code_tpl:
        # Reemplaza placeholders con valores actuales
        code_filled = code_tpl
        for k, v in collected.items():
            code_filled = code_filled.replace(f"{{{k}}}", str(v))

        with st.expander("Vista previa del código que se ejecutará"):
            st.code(code_filled, language="python")

    # ── Botón Entrenar ────────────────────────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1.5, 3])
    with col_btn:
        if st.button("Entrenar modelo →", key="train_btn", use_container_width=True):
            st.session_state.params = collected
            st.session_state.page  = "results"
            st.rerun()
