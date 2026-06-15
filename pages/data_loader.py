"""
Pantalla 4 — Carga y transformación de datos (Power Query style).
"""
import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import (
    load_csv, get_column_summary, auto_detect_types,
    apply_transformations, get_numeric_columns,
    get_categorical_columns, get_example_datasets,
    ENCODINGS, SEPARATORS,
)
from assets.icons.icons import ICON_UPLOAD
from pages._navbar import navbar

def _get_kaggle_api():
    """Autentica y retorna la API de Kaggle usando secrets de Streamlit."""
    import os
    try:
        os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
        os.environ["KAGGLE_KEY"]      = st.secrets["kaggle"]["key"]
    except Exception:
        st.error("Credenciales de Kaggle no configuradas en Streamlit Secrets.")
        return None
    try:
        import kaggle
        kaggle.api.authenticate()
        return kaggle.api
    except Exception as e:
        st.error(f"Error al autenticar con Kaggle: {e}")
        return None


def _search_kaggle(query: str):
    """Busca datasets en Kaggle y retorna lista de dicts."""
    api = _get_kaggle_api()
    if api is None:
        return None
    try:
        with st.spinner(f"Buscando '{query}' en Kaggle..."):
            datasets = api.dataset_list(search=query, file_type="csv", max_size=500_000_000)
        results = []
        for ds in datasets[:12]:
            results.append({
                "ref":       f"{ds.ref}",
                "title":     ds.title,
                "owner":     ds.ownerName if hasattr(ds, "ownerName") else str(ds.ref).split("/")[0],
                "size":      ds.totalBytes if hasattr(ds, "totalBytes") else 0,
                "files":     ds.fileCount  if hasattr(ds, "fileCount")  else "?",
                "downloads": ds.downloadCount if hasattr(ds, "downloadCount") else 0,
            })
        if not results:
            st.info(f"No se encontraron datasets CSV para '{query}'.")
        return results
    except Exception as e:
        st.error(f"Error al buscar en Kaggle: {e}")
        return None


def _load_kaggle_dataset(identifier: str):
    """Descarga un dataset de Kaggle por su ref (owner/slug) y lo carga en session_state."""
    import tempfile, glob, os
    api = _get_kaggle_api()
    if api is None:
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        with st.spinner(f"Descargando `{identifier}`..."):
            try:
                api.dataset_download_files(identifier, path=tmpdir, unzip=True)
            except Exception as e:
                st.error(f"Error al descargar: {e}")
                return
        csv_files = glob.glob(os.path.join(tmpdir, "**", "*.csv"), recursive=True)
        if not csv_files:
            st.error("El dataset no contiene archivos CSV.")
            return
        # Si hay varios CSV, pedir al usuario que elija
        chosen_file = csv_files[0]
        if len(csv_files) > 1:
            names = [os.path.relpath(f, tmpdir) for f in csv_files]
            sel = st.selectbox("El dataset tiene varios archivos — elige uno:", names, key="kaggle_file_sel")
            chosen_file = csv_files[names.index(sel)]
        try:
            df = pd.read_csv(chosen_file)
            st.session_state.df = df
            st.session_state.applied_transforms = {}
            st.session_state["kaggle_results"] = []
            st.success(f"Cargado: **{df.shape[0]:,} filas × {df.shape[1]} columnas** · `{os.path.basename(chosen_file)}`")
            st.rerun()
        except Exception as e:
            st.error(f"Error al leer el CSV: {e}")


TRANSFORM_OPTIONS = [
    "— sin cambio —",
    "Eliminar columna (drop)",
    "Codificación Label (encode_label)",
    "Codificación One-Hot (encode_onehot)",
    "Rellenar nulos con media (fillna_mean)",
    "Rellenar nulos con mediana (fillna_median)",
    "Rellenar nulos con moda (fillna_mode)",
    "Convertir a numérico (to_numeric)",
    "Normalizar [0,1] (normalize)",
    "Estandarizar z-score (standardize)",
]
TRANSFORM_MAP = {
    "— sin cambio —": None,
    "Eliminar columna (drop)": "drop",
    "Codificación Label (encode_label)": "encode_label",
    "Codificación One-Hot (encode_onehot)": "encode_onehot",
    "Rellenar nulos con media (fillna_mean)": "fillna_mean",
    "Rellenar nulos con mediana (fillna_median)": "fillna_median",
    "Rellenar nulos con moda (fillna_mode)": "fillna_mode",
    "Convertir a numérico (to_numeric)": "to_numeric",
    "Normalizar [0,1] (normalize)": "normalize",
    "Estandarizar z-score (standardize)": "standardize",
}


def render():
    navbar(active=2)

    st.markdown("""
    <div class="ml-section-header">Carga y Transformación de Datos</div>
    <div class="ml-section-sub">
        Carga tu CSV o elige un dataset de ejemplo. Luego transforma las columnas y selecciona
        features (X) y variable objetivo (y).
    </div>
    """, unsafe_allow_html=True)

    tab_upload, tab_kaggle, tab_example = st.tabs(["Cargar CSV", "Kaggle", "Dataset de ejemplo"])

    # ─── TAB 1: Upload CSV ───────────────────────────────────────────────────
    with tab_upload:
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            sep_label = st.selectbox("Separador de columnas", list(SEPARATORS.keys()), index=0)
        with col_opt2:
            encoding  = st.selectbox("Codificación del archivo", ENCODINGS, index=0)

        st.markdown(f"""
        <div class="ml-info-box">
            <strong>Separador:</strong> Define el carácter que divide las columnas en el CSV.
            La mayoría de archivos de Excel usan coma (,) o punto y coma (;).
            Si exportas desde Excel en español, probablemente sea <b>punto y coma</b>.
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Arrastra tu archivo CSV aquí o haz click para seleccionarlo",
            type=["csv", "txt"],
            help="Archivos de hasta 200 MB",
        )

        if uploaded:
            sep = SEPARATORS[sep_label]
            try:
                df = load_csv(uploaded.read(), sep=sep, encoding=encoding)
                st.session_state.df = df
                st.session_state.applied_transforms = {}  # resetear al cargar nuevo archivo
                st.success(f"Archivo cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")
                st.info("Prueba cambiando el separador o la codificación.")

    # ─── TAB 2: Kaggle ───────────────────────────────────────────────────────
    with tab_kaggle:
        st.markdown("""
        <div class="ml-info-box">
            <strong>Busca datasets de Kaggle directamente.</strong><br>
            Escribe un término (ej: <code>titanic</code>, <code>housing prices</code>, <code>iris</code>)
            y selecciona el dataset que quieres cargar.
        </div>
        """, unsafe_allow_html=True)

        col_search, col_btn = st.columns([3, 1])
        with col_search:
            query = st.text_input(
                "Buscar en Kaggle",
                placeholder="ej: titanic, iris, house prices...",
                key="kaggle_search_query",
                label_visibility="collapsed",
            )
        with col_btn:
            do_search = st.button("Buscar", key="kaggle_search_btn", use_container_width=True)

        if do_search and query.strip():
            results = _search_kaggle(query.strip())
            if results is not None:
                st.session_state["kaggle_results"] = results

        # Mostrar resultados
        results = st.session_state.get("kaggle_results", [])
        if results:
            st.markdown(f"<div style='font-size:13px;color:var(--navy-300);margin:12px 0 8px'>{len(results)} datasets encontrados</div>", unsafe_allow_html=True)
            for ds in results:
                col_info, col_load = st.columns([4, 1])
                with col_info:
                    size_mb = round(ds["size"] / 1_000_000, 1) if ds["size"] else "?"
                    st.markdown(f"""
                    <div style="background:var(--navy-800);border:1px solid var(--navy-600);
                                border-radius:10px;padding:12px 16px;margin-bottom:8px">
                        <div style="font-size:14px;font-weight:700;color:var(--white)">{ds['title']}</div>
                        <div style="font-size:12px;color:var(--navy-300);margin-top:4px">
                            {ds['owner']} · {ds['files']} archivo(s) · {size_mb} MB
                            · ⬇ {ds['downloads']:,} descargas
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_load:
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    if st.button("Cargar", key=f"load_{ds['ref']}", use_container_width=True):
                        _load_kaggle_dataset(ds["ref"])

    # ─── TAB 3: Datasets de ejemplo ─────────────────────────────────────────
    with tab_example:
        datasets = get_example_datasets()
        chosen = st.selectbox("Elige un dataset de ejemplo", list(datasets.keys()))

        model_key = st.session_state.model_key
        rec_map = {
            "linear_regression":   "Regresion Lineal (sintetico)",
            "logistic_regression": "Clasificacion Binaria (sintetico)",
            "svm":                 "Clasificacion Binaria (sintetico)",
            "kmeans":              "Clustering (3 grupos)",
            "pca":                 "Iris (multiclase)",
        }
        rec = rec_map.get(model_key)
        if rec:
            st.markdown(f"""
            <div class="ml-info-box">
                <strong>Recomendado para {model_key.replace('_',' ').title()}:</strong>
                Usa "<b>{rec}</b>" para seguir el flujo completo sin preparar datos propios.
            </div>
            """, unsafe_allow_html=True)

        if st.button("Usar este dataset", key="use_example"):
            st.session_state.df = datasets[chosen]
            st.success(f"Dataset cargado: {datasets[chosen].shape[0]} filas × {datasets[chosen].shape[1]} columnas")

    # ─── Procesamiento ───────────────────────────────────────────────────────
    df = st.session_state.df
    if df is None:
        st.info("Carga un CSV o selecciona un dataset de ejemplo para continuar.")
        return

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Preview
    with st.expander("Vista previa de los datos", expanded=True):
        st.dataframe(df.head(20), use_container_width=True, height=280)
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Filas", f"{df.shape[0]:,}")
        col_b.metric("Columnas", f"{df.shape[1]}")
        col_c.metric("Nulos totales", f"{df.isna().sum().sum():,}")
        col_d.metric("Tipos únicos", f"{df.dtypes.nunique()}")

    # Resumen de columnas
    with st.expander("Resumen de columnas"):
        summary = get_column_summary(df)
        detected = auto_detect_types(df)
        summary["Tipo detectado"] = [detected.get(c, "?") for c in df.columns]
        st.dataframe(summary, use_container_width=True)

    # ─── Power Query — Transformaciones ──────────────────────────────────────
    st.markdown("""
    <div class="ml-card-title" style="margin:24px 0 8px">
        <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
          <rect width="22" height="22" rx="5" fill="#1e3a5f"/>
          <path d="M6 8h10M6 11h7M6 14h8" stroke="#f0a500" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        Transformaciones de columnas
    </div>
    <div style="font-size:13px;color:#5a7a9f;margin-bottom:16px">
        Aplica transformaciones antes de entrenar. Los cambios se aplican en orden.
    </div>
    """, unsafe_allow_html=True)

    # Mostrar transformaciones ya aplicadas en esta sesión
    if "applied_transforms" not in st.session_state:
        st.session_state.applied_transforms = {}

    if st.session_state.applied_transforms:
        applied = st.session_state.applied_transforms
        items = " · ".join(f"<code>{col}</code> → {act}" for col, act in applied.items())
        st.markdown(f"""
        <div style="background:#f0fff4;border:1.5px solid #2e7d5a;border-radius:10px;
                    padding:12px 16px;font-size:13px;color:#1a4a2a;margin-bottom:12px">
            <strong>Transformaciones aplicadas:</strong> {items}
        </div>
        """, unsafe_allow_html=True)

    transforms_cfg = {}
    with st.expander("Configurar transformaciones por columna", expanded=False):
        for col in df.columns:
            c1, c2 = st.columns([1.5, 2])
            with c1:
                dtype_detected = detected.get(col, "?")
                st.markdown(f"""
                <div style="padding:10px 0">
                    <div style="font-size:13px;font-weight:600;color:#1e3a5f">{col}</div>
                    <div style="font-size:11px;color:#5a7a9f">{df[col].dtype} · {dtype_detected}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                sel = st.selectbox(
                    f"Acción sobre '{col}'",
                    TRANSFORM_OPTIONS,
                    key=f"transform_{col}",
                    label_visibility="collapsed",
                )
                action = TRANSFORM_MAP[sel]
                if action:
                    transforms_cfg[col] = {"action": action}

        if transforms_cfg:
            if st.button("Aplicar transformaciones"):
                try:
                    df_transformed = apply_transformations(df, transforms_cfg)
                    st.session_state.df = df_transformed
                    for col, cfg in transforms_cfg.items():
                        st.session_state.applied_transforms[col] = cfg["action"]
                    resumen = ", ".join(f"{col} → {cfg['action']}" for col, cfg in transforms_cfg.items())
                    st.toast(f"Transformaciones aplicadas: {resumen}", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al transformar: {e}")

    # ─── Selección de columnas X / y ─────────────────────────────────────────
    st.markdown("""
    <div class="ml-card-title" style="margin:24px 0 8px">
        <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
          <rect width="22" height="22" rx="5" fill="#1e3a5f"/>
          <rect x="5" y="5" width="5" height="12" rx="1.5" fill="#4a7ec8"/>
          <rect x="12" y="5" width="5" height="12" rx="1.5" fill="#f0a500"/>
        </svg>
        Selección de variables
    </div>
    """, unsafe_allow_html=True)

    all_cols = df.columns.tolist()
    algo_type = st.session_state.algo_type
    is_unsup  = algo_type == "no_supervisado"

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("""
        <div class="ml-param-label">
            Variables de entrada — X (features)
            <span class="ml-param-badge not-hyper">No es hiperparámetro</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:12px;color:#5a7a9f;margin-bottom:8px">
            Selecciona las columnas que el modelo usará para aprender.
            Deben ser numéricas (o previamente transformadas a numéricas).
        </div>
        """, unsafe_allow_html=True)
        feature_cols = st.multiselect(
            "Features (X)", all_cols,
            default=get_numeric_columns(df)[:min(5, len(get_numeric_columns(df)))],
            key="feature_select",
            label_visibility="collapsed",
        )

    with col_right:
        if not is_unsup:
            st.markdown("""
            <div class="ml-param-label">
                Variable objetivo — y (target)
                <span class="ml-param-badge not-hyper">No es hiperparámetro</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:12px;color:#5a7a9f;margin-bottom:8px">
                La columna que el modelo intentará predecir. Para regresión debe ser numérica;
                para clasificación puede ser categórica o numérica binaria.
            </div>
            """, unsafe_allow_html=True)
            target_col = st.selectbox(
                "Target (y)", [c for c in all_cols if c not in feature_cols],
                key="target_select",
                label_visibility="collapsed",
            )
        else:
            target_col = None
            st.markdown("""
            <div class="ml-info-box">
                <strong>No Supervisado:</strong> No seleccionas variable objetivo.
                El modelo aprende la estructura de X directamente.
            </div>
            """, unsafe_allow_html=True)

    # ─── Validaciones en tiempo real ─────────────────────────────────────────
    model_key  = st.session_state.model_key
    is_regression = model_key == "linear_regression"
    is_classif    = model_key in ("logistic_regression", "svm")
    blocking_errors = []

    if feature_cols:
        X_df = df[feature_cols].dropna()
        numeric_feats  = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        non_numeric_x  = [c for c in feature_cols if c not in numeric_feats]

        if non_numeric_x:
            for col in non_numeric_x:
                sample = str(df[col].dropna().iloc[0]) if df[col].notna().any() else "?"
                st.markdown(f"""
                <div class="ml-warn-box">
                    <strong>X contiene columna no numerica: <code>{col}</code></strong>
                    (muestra: <code>{sample}</code>, tipo: {df[col].dtype})<br>
                    Que hacer: en "Configurar transformaciones", aplica
                    <b>Codificacion Label</b> si es categorica (ej: "rojo","azul") o
                    <b>Convertir a numerico</b> si son numeros guardados como texto.
                    Si es una fecha, usa <b>Eliminar columna</b> o transforma antes de cargar.
                </div>
                """, unsafe_allow_html=True)
            blocking_errors.append("X tiene columnas no numericas.")

        if not is_unsup and target_col:
            y_series = df[target_col]
            y_is_numeric = pd.api.types.is_numeric_dtype(y_series)
            y_nunique    = y_series.nunique()
            sample_val   = str(y_series.dropna().iloc[0]) if y_series.notna().any() else "?"

            # Fecha detectada — solo si el dtype es fecha O el string tiene formato claro de fecha
            is_date = False
            if hasattr(y_series.dtype, "kind") and y_series.dtype.kind == "M":
                is_date = True
            elif not y_is_numeric:
                sample_str = str(sample_val)
                import re
                date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}|^\d{2}/\d{2}/\d{4}")
                if date_pattern.match(sample_str):
                    is_date = True

            if is_date or (not y_is_numeric and is_regression):
                st.markdown(f"""
                <div class="ml-warn-box">
                    <strong>El target <code>{target_col}</code> no es numerico
                    (muestra: <code>{sample_val}</code>, tipo: {y_series.dtype})</strong><br>
                    {"Es una <b>fecha</b> — los modelos no pueden predecir fechas directamente." if is_date else ""}
                    <br><br>
                    <b>Que hacer segun tu modelo:</b><br>
                    {"• <b>Regresion Lineal</b> necesita un numero continuo como target (precio, temperatura, ventas). Elige otra columna numerica." if is_regression else ""}
                    {"• <b>Clasificacion</b> necesita una etiqueta de clase (0/1, 'spam'/'ham'). Aplica <b>Codificacion Label</b> si son textos." if is_classif else ""}
                    <br>Soluciones: <b>(1)</b> Elige otra columna como target.
                    <b>(2)</b> Aplica una transformacion en el paso de transformaciones.
                </div>
                """, unsafe_allow_html=True)
                blocking_errors.append(f"Target '{target_col}' no es valido para este modelo.")

            elif is_classif and y_is_numeric and y_nunique > 20:
                st.markdown(f"""
                <div class="ml-warn-box">
                    <strong>Posible problema: <code>{target_col}</code> tiene {y_nunique} valores unicos.</strong><br>
                    Para clasificacion se esperan pocas clases (2–10).
                    Si es un numero continuo, usa <b>Regresion Lineal</b> en su lugar.
                    Si realmente es multiclase, puedes continuar.
                </div>
                """, unsafe_allow_html=True)

            elif is_regression and y_is_numeric and y_nunique <= 5:
                st.markdown(f"""
                <div class="ml-warn-box">
                    <strong>Posible problema: <code>{target_col}</code> solo tiene {y_nunique} valores unicos.</strong><br>
                    Para regresion lineal se espera una variable continua.
                    Si es una categoria, considera usar <b>Regresion Logistica</b> en su lugar.
                </div>
                """, unsafe_allow_html=True)

            if not blocking_errors:
                st.markdown(f"""
                <div class="ml-info-box">
                    <strong>Configuracion lista:</strong> X = {len(numeric_feats)} features numericas
                    · {X_df.shape[0]:,} muestras · Target = <b>{target_col}</b>
                    ({y_series.dtype}) · {y_nunique} valores unicos
                </div>
                """, unsafe_allow_html=True)

        elif is_unsup and not blocking_errors:
            st.markdown(f"""
            <div class="ml-info-box">
                <strong>Configuracion lista:</strong> X = {len(numeric_feats)} features numericas
                · {X_df.shape[0]:,} muestras validas
            </div>
            """, unsafe_allow_html=True)

    # ─── Continuar ────────────────────────────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if blocking_errors:
        st.markdown(f"""
        <div style="background:#fff0f0;border:2px solid #c0392b;border-radius:12px;
                    padding:16px 20px;font-size:13px;color:#7a1a1a">
            <strong>Corrige los problemas marcados arriba antes de continuar.</strong><br>
            {"<br>".join("• " + e for e in blocking_errors)}
        </div>
        """, unsafe_allow_html=True)
        return

    if st.button("Continuar a Hiperparametros →", key="data_continue", use_container_width=False):
        if not feature_cols:
            st.error("Selecciona al menos una feature (X).")
            return
        if not is_unsup and not target_col:
            st.error("Selecciona una variable objetivo (y).")
            return

        numeric_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_feats:
            st.error("Necesitas al menos una feature numerica. Transforma las columnas categoricas primero.")
            return

        X = df[numeric_feats].dropna().values
        st.session_state.X = X
        st.session_state.feature_cols = numeric_feats

        if not is_unsup:
            y_aligned = df[target_col].loc[df[numeric_feats].dropna().index]
            st.session_state.y = y_aligned.values
            st.session_state.target_col = target_col
        else:
            st.session_state.y = None
            st.session_state.target_col = None

        st.session_state.page = "params"
        st.rerun()
