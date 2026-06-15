"""
Pantalla 3 — Selección de modelo.
"""
import streamlit as st
from assets.icons.icons import (
    ICON_LINEAR, ICON_LOGISTIC, ICON_SVM, ICON_KMEANS, ICON_PCA,
    ICON_DECISION_TREE, ICON_RANDOM_FOREST, ICON_KNN,
    ICON_GRADIENT_BOOSTING, ICON_NAIVE_BAYES, ICON_DBSCAN, ICON_HIERARCHICAL,
)
from knowledge.models_knowledge import MODELS_KNOWLEDGE, ALGO_SELECTION_GUIDE
from pages._navbar import navbar


SUPERVISED_MODELS = [
    ("linear_regression",   "Regresión Lineal",      ICON_LINEAR,            "Regresión · Salida continua",
     "La opción más interpretable. Aprende coeficientes lineales para predecir valores numéricos."),
    ("logistic_regression", "Regresión Logística",   ICON_LOGISTIC,          "Clasificación · Probabilístico",
     "Predice la probabilidad de pertenencia a una clase. Rápido, interpretable y robusto."),
    ("svm",                 "SVM",                   ICON_SVM,               "Clasificación · Margen máximo",
     "Maximiza el margen entre clases. Excelente en espacios de alta dimensión con kernel."),
    ("decision_tree",       "Árbol de Decisión",     ICON_DECISION_TREE,     "Clasificación/Regresión · Interpretable",
     "Divide los datos en reglas if-else aprendidas. El modelo más visual e interpretable."),
    ("random_forest",       "Random Forest",         ICON_RANDOM_FOREST,     "Clasificación/Regresión · Ensemble",
     "Conjunto de cientos de árboles que votan. Robusto, preciso y resistente al sobreajuste."),
    ("knn",                 "K-Nearest Neighbors",   ICON_KNN,               "Clasificación/Regresión · Distancia",
     "Clasifica basándose en los K vecinos más cercanos. Sin entrenamiento, muy intuitivo."),
    ("gradient_boosting",   "Gradient Boosting",     ICON_GRADIENT_BOOSTING, "Clasificación/Regresión · Boosting",
     "Árboles secuenciales que corrigen errores del anterior. Alta precisión en datos tabulares."),
    ("naive_bayes",         "Naive Bayes",           ICON_NAIVE_BAYES,       "Clasificación · Probabilístico",
     "Clasificador bayesiano ultrarrápido. Sorprendentemente efectivo en texto y clases claras."),
]

UNSUPERVISED_MODELS = [
    ("kmeans",       "K-Means",                ICON_KMEANS,       "Clustering · Grupos esféricos",
     "Agrupa observaciones en K clusters según similitud. El algoritmo más popular de clustering."),
    ("pca",          "PCA",                    ICON_PCA,          "Reducción dimensional · Varianza",
     "Proyecta datos en menos dimensiones preservando la mayor varianza. Ideal para visualización."),
    ("dbscan",       "DBSCAN",                 ICON_DBSCAN,       "Clustering · Densidad · Outliers",
     "Detecta clusters de cualquier forma y marca outliers automáticamente. No necesita definir K."),
    ("hierarchical", "Clustering Jerárquico",  ICON_HIERARCHICAL, "Clustering · Dendrograma",
     "Construye un árbol de clusters. Revela estructura anidada y no requiere K fijo de antemano."),
]


def render():
    navbar(active=1)

    algo_type = st.session_state.algo_type
    models = SUPERVISED_MODELS if algo_type == "supervisado" else UNSUPERVISED_MODELS

    label = "Supervisado" if algo_type == "supervisado" else "No Supervisado"
    st.markdown(f"""
    <div class="ml-section-header">Elige el modelo — {label}</div>
    <div class="ml-section-sub">Cada modelo tiene sus fortalezas. Lee la descripción antes de elegir.</div>
    """, unsafe_allow_html=True)

    # ── Guía de selección ────────────────────────────────────────────────────
    with st.expander("Guía: ¿Cuándo usar cada modelo?"):
        cheat = ALGO_SELECTION_GUIDE["cheatsheet"]
        rows = [(k, v) for k, v in cheat.items()
                if (algo_type == "supervisado" and k in ["Regresión Lineal","Regresión Logística","SVM"])
                or (algo_type == "no_supervisado" and k in ["K-Means","PCA"])]
        for name, desc in rows:
            st.markdown(f"""
            <div class="ml-info-box">
                <strong>{name}</strong> — {desc}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    n_cols = 4
    for row_start in range(0, len(models), n_cols):
        row_models = models[row_start:row_start + n_cols]
        cols = st.columns(n_cols)
        for col, (key, name, icon, tag, desc) in zip(cols, row_models):
            with col:
                selected = st.session_state.model_key == key
                border = "2px solid #f0a500" if selected else "2px solid #e8f0f8"
                bg     = "#f0f5fb" if selected else "#ffffff"

                st.markdown(f"""
                <div style="background:{bg};border:{border};border-radius:18px;padding:22px 16px;
                            text-align:center;transition:all 0.2s;min-height:240px;
                            display:flex;flex-direction:column;align-items:center;gap:8px">
                    <div>{icon}</div>
                    <div style="font-size:15px;font-weight:700;color:#1e3a5f;line-height:1.3">{name}</div>
                    <div style="display:inline-block;background:#e8f0f8;border-radius:8px;
                                padding:2px 10px;font-size:10px;font-weight:600;color:#2a5298">{tag}</div>
                    <div style="font-size:12px;color:#5a7a9f;line-height:1.55">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                if st.button(
                    f"{'✓ Seleccionado' if selected else 'Seleccionar'}",
                    key=f"btn_{key}",
                    use_container_width=True,
                ):
                    st.session_state.model_key = key
                    st.session_state.page = "data"
                    st.rerun()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Preview del modelo seleccionado ─────────────────────────────────────
    if st.session_state.model_key:
        mk = st.session_state.model_key
        if mk in MODELS_KNOWLEDGE:
            info = MODELS_KNOWLEDGE[mk]
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="ml-card">
                <div class="ml-card-title">
                    {info['name']} — Vista previa
                </div>
                <div class="ml-card-sub">{info['one_liner']}</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
                    <div>
                        <div style="font-size:12px;font-weight:700;color:#1e3a5f;margin-bottom:8px">
                            CUÁNDO USARLO
                        </div>
                        {"".join(f"<div style='font-size:12px;color:#5a7a9f;padding:3px 0'>• {u}</div>" for u in info['when_to_use'])}
                    </div>
                    <div>
                        <div style="font-size:12px;font-weight:700;color:#c0392b;margin-bottom:8px">
                            CUÁNDO NO USARLO
                        </div>
                        {"".join(f"<div style='font-size:12px;color:#5a7a9f;padding:3px 0'>• {u}</div>" for u in info['when_not_to_use'])}
                    </div>
                </div>
                <div style="margin-top:18px;padding:16px;background:#0d1b2a;border-radius:10px;
                            font-family:monospace;font-size:12px;color:#a8c4e0;line-height:1.7">
                    <div style="color:#5a7a9f;margin-bottom:6px"># Matemática del modelo</div>
                    {info['math_concept'].replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
