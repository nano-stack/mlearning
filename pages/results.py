"""
Pantalla 6 — Resultados: dashboard de métricas + visualización Manim + código.
"""
import streamlit as st
import numpy as np
import subprocess
import shutil
from pathlib import Path
from pages._navbar import navbar
from knowledge.models_knowledge import MODELS_KNOWLEDGE


ANIM_SCENE_MAP = {
    "linear_regression":   "LinearRegressionScene",
    "logistic_regression": "LogisticRegressionScene",
    "svm":                 "SVMScene",
    "kmeans":              "KMeansScene",
    "pca":                 "PCAScene",
    "decision_tree":       "DecisionTreeScene",
    "random_forest":       "RandomForestScene",
    "knn":                 "KNNScene",
    "gradient_boosting":   "GradientBoostingScene",
    "naive_bayes":         "NaiveBayesScene",
    "dbscan":              "DBSCANScene",
    "hierarchical":        "HierarchicalScene",
}


def _get_animation_path(scene_name: str) -> Path:
    return Path(f"assets/videos/{scene_name}.mp4")


def _render_animation(scene_name: str) -> bool:
    path = _get_animation_path(scene_name)
    if path.exists():
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["python", "-m", "manim", "-qm", "--media_dir", "assets",
             "animations/manim_scenes.py", scene_name],
            capture_output=True, timeout=180
        )
        rendered = list(Path("assets").rglob(f"{scene_name}*.mp4"))
        if rendered:
            shutil.copy(rendered[0], path)
            return True
    except Exception:
        pass
    return False


def _run_model(model_key, X, y, params):
    from utils.model_runner import (
        run_linear_regression, plot_linear_regression,
        run_logistic_regression, plot_logistic_regression,
        run_svm, plot_svm,
        run_kmeans, plot_kmeans,
        run_pca, plot_pca,
        run_decision_tree, plot_decision_tree,
        run_random_forest, plot_random_forest,
        run_knn, plot_knn,
        run_gradient_boosting, plot_gradient_boosting,
        run_naive_bayes, plot_naive_bayes,
        run_dbscan, plot_dbscan,
        run_hierarchical, plot_hierarchical,
    )
    feature_names = st.session_state.feature_cols

    if model_key == "linear_regression":
        result = run_linear_regression(X, y, params)
        plots  = plot_linear_regression(result, feature_names)
    elif model_key == "logistic_regression":
        result = run_logistic_regression(X, y, params)
        plots  = plot_logistic_regression(result)
    elif model_key == "svm":
        result = run_svm(X, y, params)
        plots  = plot_svm(result)
    elif model_key == "kmeans":
        result = run_kmeans(X, params)
        plots  = plot_kmeans(result, feature_names)
    elif model_key == "pca":
        result = run_pca(X, params)
        plots  = plot_pca(result, feature_names)
    elif model_key == "decision_tree":
        result = run_decision_tree(X, y, params)
        plots  = plot_decision_tree(result, feature_names)
    elif model_key == "random_forest":
        result = run_random_forest(X, y, params)
        plots  = plot_random_forest(result, feature_names)
    elif model_key == "knn":
        result = run_knn(X, y, params)
        plots  = plot_knn(result, feature_names)
    elif model_key == "gradient_boosting":
        result = run_gradient_boosting(X, y, params)
        plots  = plot_gradient_boosting(result, feature_names)
    elif model_key == "naive_bayes":
        result = run_naive_bayes(X, y, params)
        plots  = plot_naive_bayes(result)
    elif model_key == "dbscan":
        result = run_dbscan(X, params)
        plots  = plot_dbscan(result, feature_names)
    elif model_key == "hierarchical":
        result = run_hierarchical(X, params)
        plots  = plot_hierarchical(result, feature_names)
    else:
        result, plots = {}, {}

    return result, plots


def render():
    navbar(active=4)

    model_key = st.session_state.model_key
    info = MODELS_KNOWLEDGE.get(model_key, {})
    X    = st.session_state.X
    y    = st.session_state.y
    params = st.session_state.params

    if X is None:
        st.error("No hay datos cargados. Vuelve al paso anterior.")
        return

    # ── Entrenar (con cache en sesión) ────────────────────────────────────────
    if st.session_state.result is None:
        with st.spinner("Entrenando modelo..."):
            try:
                result, plots = _run_model(model_key, X, y, params)
                st.session_state.result = result
                st.session_state.plots  = plots
            except Exception as e:
                msg = str(e)
                st.markdown(f"""
                <div style="background:#fff0f0;border:2px solid #c0392b;border-radius:14px;
                            padding:24px 28px;margin-top:16px">
                    <div style="font-size:16px;font-weight:700;color:#c0392b;margin-bottom:10px">
                        Error durante el entrenamiento
                    </div>
                    <div style="font-size:14px;color:#7a1a1a;margin-bottom:16px">
                        <code>{msg}</code>
                    </div>
                    <div style="font-size:13px;color:#5a1a1a;line-height:1.7">
                        <strong>Posibles causas y soluciones:</strong><br>
                        • La columna <b>target (y)</b> contiene texto, fechas o valores no numericos
                          → vuelve a <b>Datos</b> y elige otra columna o aplica una transformacion.<br>
                        • Alguna columna de <b>X</b> no es numerica
                          → aplica <b>Codificacion Label</b> o elimina esa columna.<br>
                        • El dataset tiene muy pocas filas para el test_size elegido
                          → reduce el test_size en <b>Parametros</b> o usa mas datos.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_b1, col_b2, _ = st.columns([1, 1, 3])
                with col_b1:
                    if st.button("Volver a Datos", key="err_back_data"):
                        st.session_state.result = None
                        st.session_state.plots  = None
                        st.session_state.page   = "data"
                        st.rerun()
                with col_b2:
                    if st.button("Volver a Parametros", key="err_back_params"):
                        st.session_state.result = None
                        st.session_state.plots  = None
                        st.session_state.page   = "params"
                        st.rerun()
                return
    else:
        result = st.session_state.result
        plots  = st.session_state.plots

    # ── Botón re-entrenar ─────────────────────────────────────────────────────
    col_re, _ = st.columns([1, 5])
    with col_re:
        if st.button("Volver a entrenar", key="retrain"):
            st.session_state.result = None
            st.session_state.plots  = None
            st.rerun()

    st.markdown(f"""
    <div class="ml-section-header">Resultados — {info.get('name', model_key)}</div>
    <div class="ml-section-sub">{info.get('one_liner', '')}</div>
    """, unsafe_allow_html=True)

    # ── Métricas principales ──────────────────────────────────────────────────
    metrics = result.get("metrics", {})
    if metrics:
        st.markdown("<div class='ml-card-title' style='margin-bottom:16px'>Métricas de desempeño</div>",
                    unsafe_allow_html=True)
        # Máximo 4 por fila para no solaparse
        items = list(metrics.items())
        chunk_size = 4
        for chunk_start in range(0, len(items), chunk_size):
            chunk = items[chunk_start:chunk_start + chunk_size]
            metric_cols = st.columns(len(chunk))
            for col, (name, val) in zip(metric_cols, chunk):
                with col:
                    if isinstance(val, float):
                        display = f"{val:.4f}"
                    elif isinstance(val, int):
                        display = f"{val:,}"
                    else:
                        display = str(val)
                    metric_info = info.get("metrics", {}).get(name, "")
                    st.markdown(f"""
                    <div class="ml-metric" style="margin-bottom:8px">
                        <div class="ml-metric-value" style="font-size:26px">{display}</div>
                        <div class="ml-metric-label" style="margin-top:6px">{name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if metric_info:
                        st.caption(metric_info[:100] + ("…" if len(metric_info) > 100 else ""))

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Tabs: Visualización Manim · Gráficos · Código ──────────────────────────
    tab_anim, tab_plots, tab_code, tab_learn = st.tabs([
        "Animación matemática",
        "Dashboard de desempeño",
        "Código generado",
        "Aprende más",
    ])

    # ── Tab 1: Animación Manim ────────────────────────────────────────────────
    with tab_anim:
        scene_name = ANIM_SCENE_MAP.get(model_key)
        anim_path  = _get_animation_path(scene_name) if scene_name else None

        col_v, col_text = st.columns([2, 1])
        with col_v:
            if anim_path and anim_path.exists():
                st.video(str(anim_path), autoplay=True, loop=True)
            else:
                # Intento de render en background con estado
                if scene_name:
                    with st.spinner("Renderizando animación Manim... (puede tomar 30–90 segundos la primera vez)"):
                        ok = _render_animation(scene_name)
                    if ok:
                        st.video(str(anim_path), autoplay=True, loop=True)
                    else:
                        st.info(
                            "La animación Manim requiere que Manim esté instalado.\n\n"
                            "Para instalar: `pip install manim`\n\n"
                            f"Para renderizar manualmente:\n"
                            f"```\nmanim -pqm animations/manim_scenes.py {scene_name}\n```"
                        )

        with col_text:
            math_concept = info.get("math_concept", "")
            st.markdown(f"""
            <div class="ml-card">
                <div class="ml-card-title">Matemática del modelo</div>
                <div style="font-family:monospace;font-size:13px;color:#1e3a5f;
                            background:#f0f5fb;padding:16px;border-radius:10px;
                            line-height:1.8;white-space:pre-wrap">{math_concept}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 2: Dashboard Streamlit ────────────────────────────────────────────
    with tab_plots:
        if not plots:
            st.info("No hay gráficos disponibles.")
        else:
            # Gráficos según modelo
            if model_key == "linear_regression":
                c1, c2 = st.columns(2)
                with c1:
                    if "pred" in plots:     st.plotly_chart(plots["pred"],     use_container_width=True)
                    if "coef" in plots:     st.plotly_chart(plots["coef"],     use_container_width=True)
                with c2:
                    if "residuals" in plots: st.plotly_chart(plots["residuals"], use_container_width=True)
                    if "learning" in plots:  st.plotly_chart(plots["learning"],  use_container_width=True)

                # Coeficientes tabla
                if "coef_df" in result:
                    st.markdown("<div class='ml-card-title' style='margin-top:8px'>Coeficientes del modelo</div>",
                                unsafe_allow_html=True)
                    feat_names = st.session_state.feature_cols
                    coef_df = result["coef_df"].copy()
                    if feat_names and len(feat_names) == len(coef_df):
                        coef_df["Variable"] = feat_names
                    st.dataframe(coef_df[["Variable","Coeficiente"]].sort_values("Coeficiente", key=abs, ascending=False),
                                 use_container_width=True)

            elif model_key in ("logistic_regression", "svm"):
                if "cm" in plots:      st.plotly_chart(plots["cm"],      use_container_width=True)
                if "report" in plots:  st.plotly_chart(plots["report"],  use_container_width=True)
                if "roc" in plots:
                    c1, c2 = st.columns(2)
                    with c1: st.plotly_chart(plots["roc"], use_container_width=True)
                    with c2:
                        if "pr" in plots: st.plotly_chart(plots["pr"], use_container_width=True)

                # Classification report tabla
                if "report" in result:
                    st.markdown("<div class='ml-card-title' style='margin-top:8px'>Reporte de clasificación</div>",
                                unsafe_allow_html=True)
                    import pandas as pd
                    rpt = pd.DataFrame(result["report"]).T.round(3)
                    st.dataframe(rpt, use_container_width=True)

            elif model_key == "kmeans":
                # Métricas de clustering con explicación
                st.markdown("""
                <div class="ml-info-box" style="margin-bottom:16px">
                <strong>Cómo leer estas métricas:</strong>
                Silhouette cercano a 1 = clusters bien separados.
                Davies-Bouldin cercano a 0 = clusters compactos y separados.
                Calinski-Harabasz más alto = mejor.
                </div>
                """, unsafe_allow_html=True)
                if "scatter" in plots: st.plotly_chart(plots["scatter"], use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    if "elbow" in plots:  st.plotly_chart(plots["elbow"],  use_container_width=True)
                with c2:
                    if "sizes" in plots:  st.plotly_chart(plots["sizes"],  use_container_width=True)

            elif model_key == "pca":
                if "scree" in plots:   st.plotly_chart(plots["scree"],   use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    if "2d" in plots:      st.plotly_chart(plots["2d"],      use_container_width=True)
                with c2:
                    if "loadings" in plots: st.plotly_chart(plots["loadings"], use_container_width=True)

            elif model_key == "decision_tree":
                if "importances" in plots: st.plotly_chart(plots["importances"], use_container_width=True)
                if "cm" in plots or "report" in plots:
                    c1, c2 = st.columns(2)
                    with c1:
                        if "cm" in plots: st.plotly_chart(plots["cm"], use_container_width=True)
                    with c2:
                        if "report" in plots: st.plotly_chart(plots["report"], use_container_width=True)
                if "residuals" in plots: st.plotly_chart(plots["residuals"], use_container_width=True)

            elif model_key == "random_forest":
                if "importances" in plots: st.plotly_chart(plots["importances"], use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    if "cm" in plots: st.plotly_chart(plots["cm"], use_container_width=True)
                with c2:
                    if "roc" in plots: st.plotly_chart(plots["roc"], use_container_width=True)
                if "report" in plots: st.plotly_chart(plots["report"], use_container_width=True)

            elif model_key == "knn":
                if "k_curve" in plots: st.plotly_chart(plots["k_curve"], use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    if "cm" in plots: st.plotly_chart(plots["cm"], use_container_width=True)
                with c2:
                    if "roc" in plots: st.plotly_chart(plots["roc"], use_container_width=True)

            elif model_key == "gradient_boosting":
                c1, c2 = st.columns(2)
                with c1:
                    if "importances" in plots: st.plotly_chart(plots["importances"], use_container_width=True)
                with c2:
                    if "loss" in plots: st.plotly_chart(plots["loss"], use_container_width=True)
                if "cm" in plots or "roc" in plots:
                    c1, c2 = st.columns(2)
                    with c1:
                        if "cm" in plots: st.plotly_chart(plots["cm"], use_container_width=True)
                    with c2:
                        if "roc" in plots: st.plotly_chart(plots["roc"], use_container_width=True)

            elif model_key == "naive_bayes":
                if "cm" in plots: st.plotly_chart(plots["cm"], use_container_width=True)
                if "report" in plots: st.plotly_chart(plots["report"], use_container_width=True)
                if "roc" in plots:
                    c1, c2 = st.columns(2)
                    with c1: st.plotly_chart(plots["roc"], use_container_width=True)
                    with c2:
                        if "pr" in plots: st.plotly_chart(plots["pr"], use_container_width=True)

            elif model_key == "dbscan":
                st.markdown("""
                <div class="ml-info-box" style="margin-bottom:16px">
                <strong>Puntos etiquetados como -1 (rojo) son outliers</strong> — no pertenecen a ningún cluster.
                Ajusta <b>eps</b> y <b>min_samples</b> para controlar cuántos outliers detectas.
                </div>
                """, unsafe_allow_html=True)
                if "scatter" in plots: st.plotly_chart(plots["scatter"], use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    if "eps" in plots: st.plotly_chart(plots["eps"], use_container_width=True)
                with c2:
                    if "sizes" in plots: st.plotly_chart(plots["sizes"], use_container_width=True)

            elif model_key == "hierarchical":
                st.markdown("""
                <div class="ml-info-box" style="margin-bottom:16px">
                <strong>Dendrograma:</strong> cada fusión representa la unión de dos clusters.
                Corta el árbol donde el salto vertical es más grande para encontrar el K natural.
                </div>
                """, unsafe_allow_html=True)
                if "dendrogram" in plots: st.plotly_chart(plots["dendrogram"], use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    if "scatter" in plots: st.plotly_chart(plots["scatter"], use_container_width=True)
                with c2:
                    if "silhouette" in plots: st.plotly_chart(plots["silhouette"], use_container_width=True)
                if "sizes" in plots: st.plotly_chart(plots["sizes"], use_container_width=True)

    # ── Tab 3: Código generado ────────────────────────────────────────────────
    with tab_code:
        code_tpl = info.get("code_template", "")
        if code_tpl:
            code_filled = code_tpl
            for k, v in params.items():
                code_filled = code_filled.replace(f"{{{k}}}", str(v))

            # Encabezado con imports
            full_code = f"""# ═══════════════════════════════════════════════════════════
# MLearning — Código generado para {info.get('name', model_key)}
# Modelo: {model_key}  |  Tipo: {info.get('type','')}  |  Tarea: {info.get('task','')}
# ═══════════════════════════════════════════════════════════

import pandas as pd
import numpy as np

# Carga tus datos aquí
# df = pd.read_csv("tu_archivo.csv")
# X = df[{st.session_state.feature_cols}].values
# y = df["{st.session_state.target_col}"].values  # si aplica

{code_filled}"""

            st.code(full_code, language="python")
            st.download_button(
                label="Descargar código .py",
                data=full_code,
                file_name=f"mlearning_{model_key}.py",
                mime="text/x-python",
            )

    # ── Tab 4: Aprende más ────────────────────────────────────────────────────
    with tab_learn:
        st.markdown(f"""
        <div class="ml-card-title">¿Cuándo usar {info.get('name', model_key)}?</div>
        """, unsafe_allow_html=True)

        col_yes, col_no = st.columns(2)
        with col_yes:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#2e7d5a;margin-bottom:8px'>USAR cuando:</div>",
                        unsafe_allow_html=True)
            for item in info.get("when_to_use", []):
                st.markdown(f"""
                <div class="ml-info-box" style="margin-bottom:8px">• {item}</div>
                """, unsafe_allow_html=True)

        with col_no:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#c0392b;margin-bottom:8px'>NO USAR cuando:</div>",
                        unsafe_allow_html=True)
            for item in info.get("when_not_to_use", []):
                st.markdown(f"""
                <div class="ml-warn-box" style="margin-bottom:8px">• {item}</div>
                """, unsafe_allow_html=True)

        # Params resumen
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='ml-card-title'>Resumen de parámetros configurados</div>",
                    unsafe_allow_html=True)

        params_cfg = info.get("params", {})
        for pk, pv in params.items():
            cfg = params_cfg.get(pk, {})
            is_hyper = cfg.get("is_hyperparam", False)
            badge = "🔵 Hiperparámetro" if is_hyper else "⚪ Parámetro de proceso"
            st.markdown(f"""
            <div class="ml-info-box">
                <strong>{cfg.get('label', pk)}</strong> = <code>{pv}</code>
                <span style="float:right;font-size:11px;color:#5a7a9f">{badge}</span><br>
                <span style="font-size:12px">{cfg.get('what_it_is','')}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Volver a empezar ──────────────────────────────────────────────────────
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    if st.button("Empezar de nuevo con otro modelo", key="restart"):
        for key in ["page","algo_type","model_key","df","X","y",
                    "feature_cols","target_col","params","result","plots"]:
            st.session_state[key] = None if key != "page" else "select_type"
            if key == "feature_cols":
                st.session_state[key] = []
        st.rerun()
