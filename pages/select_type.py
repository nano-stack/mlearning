"""
Pantalla 2 — Selección de tipo de aprendizaje.
"""
import streamlit as st
from assets.icons.icons import ICON_SUPERVISED, ICON_UNSUPERVISED
from pages._navbar import navbar


def render():
    navbar(active=0)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 3, 1])
    with col:
        st.markdown("""
        <div class="ml-section-header" style="text-align:center">¿Qué tipo de aprendizaje quieres explorar?</div>
        <div class="ml-section-sub" style="text-align:center;margin-bottom:36px">
            El tipo de aprendizaje determina si tus datos tienen etiquetas conocidas o no.
        </div>
        """, unsafe_allow_html=True)

        # ── Info conceptual expandible ────────────────────────────────────────
        with st.expander("¿Cuál es la diferencia? (click para expandir)"):
            st.markdown("""
            <div class="ml-info-box">
            <strong>Aprendizaje Supervisado</strong><br>
            Entrenas el modelo con datos etiquetados: para cada entrada <b>X</b> existe una salida
            conocida <b>y</b>. El modelo aprende a mapear X → y para predecir en datos nuevos.<br>
            <i>Ejemplo: predecir el precio de una casa (y continua) o si un email es spam (y categórica).</i>
            </div>
            <div class="ml-info-box">
            <strong>Aprendizaje No Supervisado</strong><br>
            No hay etiquetas. El modelo encuentra estructura, patrones o grupos en los datos por sí solo.<br>
            <i>Ejemplo: agrupar clientes similares (clustering) o reducir 50 variables a 3 (PCA).</i>
            </div>
            <div class="ml-warn-box">
            <strong>¿Cuándo usar cada uno?</strong><br>
            Si tienes y (etiquetas) → Supervisado.<br>
            Si no tienes y o quieres explorar la estructura → No Supervisado.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    _, c1, gap, c2, _ = st.columns([0.3, 2, 0.5, 2, 0.3])

    card_style = """
    background: #1a2a3f;
    border-radius: 20px;
    padding: 36px 32px;
    border: 2px solid {border};
    box-shadow: 0 4px 24px rgba(0,0,0,0.30);
    height: 430px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    gap: 14px;
    """

    with c1:
        border = "#f0a500" if st.session_state.algo_type == "supervisado" else "#2a3f5a"
        st.markdown(f"""
        <div style="{card_style.format(border=border)}">
          <div style="text-align:center; margin-bottom:8px">{ICON_SUPERVISED}</div>
          <div style="font-size:22px; font-weight:800; color:white; text-align:center">
            Supervisado
          </div>
          <div style="font-size:13px; color:#6a8faf; text-align:center; margin-bottom:4px">
            Datos con etiquetas · Predicción directa
          </div>
          <div style="background:rgba(74,126,200,0.12); border-radius:12px; padding:18px 20px;
                      font-size:13px; color:#a8c4e0; line-height:2.2; flex:1">
            <span style="color:white;font-weight:700">Modelos disponibles:</span><br>
            &nbsp;&nbsp;• Regresión Lineal<br>
            &nbsp;&nbsp;• Regresión Logística<br>
            &nbsp;&nbsp;• SVM (Support Vector Machine)
          </div>
          <div style="font-size:12px; color:#6a8faf; line-height:1.7; padding: 0 4px">
            Necesitas una columna <b style="color:#a8c4e0">objetivo (y)</b> en tus datos.
            El modelo aprende de ejemplos etiquetados para predecir casos nuevos.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        if st.button("Elegir Supervisado", key="btn_sup", use_container_width=True):
            st.session_state.algo_type = "supervisado"
            st.session_state.page = "select_model"
            st.rerun()

    with c2:
        border = "#f0a500" if st.session_state.algo_type == "no_supervisado" else "#2a3f5a"
        st.markdown(f"""
        <div style="{card_style.format(border=border)}">
          <div style="text-align:center; margin-bottom:8px">{ICON_UNSUPERVISED}</div>
          <div style="font-size:22px; font-weight:800; color:white; text-align:center">
            No Supervisado
          </div>
          <div style="font-size:13px; color:#6a8faf; text-align:center; margin-bottom:4px">
            Datos sin etiquetas · Descubrimiento de estructura
          </div>
          <div style="background:rgba(74,126,200,0.12); border-radius:12px; padding:18px 20px;
                      font-size:13px; color:#a8c4e0; line-height:2.2; flex:1">
            <span style="color:white;font-weight:700">Modelos disponibles:</span><br>
            &nbsp;&nbsp;• K-Means Clustering<br>
            &nbsp;&nbsp;• PCA — Análisis de Componentes Principales
          </div>
          <div style="font-size:12px; color:#6a8faf; line-height:1.7; padding: 0 4px">
            <b style="color:#a8c4e0">No necesitas</b> columna objetivo. El modelo descubre
            patrones ocultos en los datos sin guía externa.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        if st.button("Elegir No Supervisado", key="btn_nosup", use_container_width=True):
            st.session_state.algo_type = "no_supervisado"
            st.session_state.page = "select_model"
            st.rerun()
