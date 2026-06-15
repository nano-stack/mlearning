"""
Entrenamiento y métricas de todos los modelos.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)

def _to_numpy(arr, force_float=True):
    """Convierte cualquier array-like a numpy, evitando el backend Arrow de pandas."""
    if hasattr(arr, "to_numpy"):
        raw = arr.to_numpy(na_value=np.nan)
    else:
        raw = np.array(arr)
    if force_float:
        try:
            return raw.astype(float)
        except (ValueError, TypeError):
            raise ValueError(
                f"La columna seleccionada contiene valores no numéricos: {raw[:3]}.\n"
                "Aplica una transformación (encode_label, to_numeric) en el paso de Datos."
            )
    return raw


NAVY      = "#1e3a5f"
NAVY_900  = "#0d1b2a"
NAVY_800  = "#1e3a5f"
NAVY_600  = "#3d6fba"
NAVY_300  = "#a8c4e0"
GOLD      = "#f0a500"
WHITE     = "#ffffff"
PALETTE   = ["#4a7ec8", "#f0a500", "#2e7d5a", "#c0392b", "#8e44ad", "#1abc9c"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def _dark_fig(ncols=1, nrows=1, figsize=None):
    """Crea figura matplotlib con tema oscuro de la app."""
    if figsize is None:
        figsize = (5.5 * ncols, 4.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(NAVY_900)
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        ax.set_facecolor(NAVY_800)
        ax.tick_params(colors=NAVY_300, labelsize=9)
        ax.xaxis.label.set_color(NAVY_300)
        ax.yaxis.label.set_color(NAVY_300)
        ax.title.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(NAVY_600)
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
#  ANIMACIONES PLOTLY CON DATOS REALES
# ══════════════════════════════════════════════════════════════════════════════
def animate_linear_regression(X, y, feature_names=None) -> "go.Figure":
    """Anima gradient descent sobre los datos reales del usuario."""
    X_np = _to_numpy(X)
    y_np = _to_numpy(y)

    # Usar solo la primera feature para visualización 2D
    x1 = X_np[:, 0]
    fname = (feature_names[0] if feature_names else "X")

    x_line = np.linspace(x1.min(), x1.max(), 100)

    # Simular gradient descent manual (learning rate fijo)
    w, b = 0.0, 0.0
    lr = 0.01
    n_frames = 40
    frames, losses = [], []

    for step in range(n_frames):
        y_hat = w * x1 + b
        err = y_hat - y_np
        loss = float(np.mean(err ** 2))
        losses.append(loss)

        y_line = w * x_line + b

        frames.append(go.Frame(
            data=[
                go.Scatter(x=x1.tolist(), y=y_np.tolist(), mode="markers",
                           marker=dict(color=PALETTE[0], size=7, opacity=0.7),
                           name="Datos reales"),
                go.Scatter(x=x_line.tolist(), y=y_line.tolist(), mode="lines",
                           line=dict(color=GOLD, width=2.5), name="Recta ajustada"),
            ],
            layout=go.Layout(
                title_text=f"Paso {step+1}/{n_frames} · MSE: {loss:.4f} · w={w:.3f} · b={b:.3f}"
            ),
            name=str(step),
        ))
        # Actualizar pesos
        dw = float(np.mean(err * x1))
        db = float(np.mean(err))
        w -= lr * dw
        b -= lr * db

    # Figura inicial
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=dict(text=f"Paso 1/{n_frames} · Ajuste de recta sobre datos reales",
                       font=dict(color=WHITE, size=14)),
            xaxis=dict(title=fname, color=NAVY_300, gridcolor=NAVY_600,
                       zeroline=False, showgrid=True),
            yaxis=dict(title="y", color=NAVY_300, gridcolor=NAVY_600,
                       zeroline=False, showgrid=True),
            paper_bgcolor=NAVY_900,
            plot_bgcolor=NAVY_800,
            font=dict(color=NAVY_300),
            legend=dict(bgcolor=NAVY_800, bordercolor=NAVY_600, font=dict(color=WHITE)),
            height=460,
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.15, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Reproducir",
                         method="animate",
                         args=[None, dict(frame=dict(duration=120, redraw=True),
                                         fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ Pausar",
                         method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")]),
                ],
                bgcolor=NAVY_800, bordercolor=GOLD, font=dict(color=WHITE),
            )],
            sliders=[dict(
                steps=[dict(method="animate", args=[[str(i)],
                            dict(mode="immediate", frame=dict(duration=120, redraw=True))],
                            label=str(i+1)) for i in range(n_frames)],
                currentvalue=dict(prefix="Paso: ", font=dict(color=WHITE)),
                bgcolor=NAVY_800, bordercolor=NAVY_600,
                font=dict(color=NAVY_300),
            )],
        )
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  REGRESIÓN LINEAL
# ══════════════════════════════════════════════════════════════════════════════
def run_linear_regression(X, y, params: dict) -> dict:
    from sklearn.linear_model import LinearRegression
    X, y = _to_numpy(X), _to_numpy(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"],
        random_state=int(params["random_state"]),
    )
    model = LinearRegression(
        fit_intercept=params["fit_intercept"],
        positive=params["positive"],
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)

    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

    coef_df = pd.DataFrame({
        "Variable": [f"X{i}" for i in range(len(model.coef_))],
        "Coeficiente": model.coef_,
        "Abs": np.abs(model.coef_),
    }).sort_values("Abs", ascending=False)

    return {
        "model": model,
        "metrics": {"MSE": mse, "RMSE": rmse, "R²": r2, "MAE": mae, "CV R² (5-fold)": cv_r2},
        "train": (X_train, y_train, y_pred_train),
        "test":  (X_test,  y_test,  y_pred),
        "coef_df": coef_df,
        "intercept": model.intercept_,
    }


def plot_linear_regression(result: dict, feature_names=None) -> dict:
    X_test, y_test, y_pred = result["test"]
    X_train, y_train, y_pred_train = result["train"]
    residuals = y_test - y_pred

    fig, axes = _dark_fig(ncols=2, nrows=2, figsize=(12, 8))
    ax_pred, ax_res, ax_coef, ax_lc = axes.flat

    # 1. Predicho vs Real
    ax_pred.scatter(y_test, y_pred, color=PALETTE[0], alpha=0.75, s=50, edgecolors="none")
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax_pred.plot([mn, mx], [mn, mx], "--", color=GOLD, linewidth=2, label="Ideal")
    ax_pred.set_title("Predicho vs Real")
    ax_pred.set_xlabel("Valor Real")
    ax_pred.set_ylabel("Predicción")
    ax_pred.legend(facecolor=NAVY_800, edgecolor=NAVY_600, labelcolor=WHITE, fontsize=8)

    # 2. Residuos
    ax_res.scatter(y_pred, residuals, color=PALETTE[0], alpha=0.7, s=50, edgecolors="none")
    ax_res.axhline(0, color=GOLD, linestyle="--", linewidth=1.5)
    ax_res.set_title("Residuos vs Predicción")
    ax_res.set_xlabel("Predicción")
    ax_res.set_ylabel("Residuo")

    # 3. Coeficientes
    coef_df = result["coef_df"].copy()
    if feature_names and len(feature_names) == len(coef_df):
        coef_df["Variable"] = feature_names
    coef_df = coef_df.head(15).sort_values("Coeficiente")
    colors = [PALETTE[3] if v < 0 else PALETTE[0] for v in coef_df["Coeficiente"]]
    ax_coef.barh(coef_df["Variable"], coef_df["Coeficiente"], color=colors)
    ax_coef.axvline(0, color=NAVY_300, linewidth=0.8)
    ax_coef.set_title("Coeficientes del modelo")
    ax_coef.set_xlabel("Valor")

    # 4. Curva de aprendizaje
    model = result["model"]
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_all, y_all, cv=5, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    ax_lc.plot(train_sizes, train_scores.mean(axis=1), "o-", color=PALETTE[0], label="Train R²")
    ax_lc.fill_between(train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.15, color=PALETTE[0])
    ax_lc.plot(train_sizes, val_scores.mean(axis=1), "o-", color=GOLD, label="Val R²")
    ax_lc.fill_between(train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.15, color=GOLD)
    ax_lc.set_title("Curva de Aprendizaje")
    ax_lc.set_xlabel("Muestras de entrenamiento")
    ax_lc.set_ylabel("R²")
    ax_lc.legend(facecolor=NAVY_800, edgecolor=NAVY_600, labelcolor=WHITE, fontsize=8)

    fig.tight_layout(pad=2.0)
    return {"fig": fig}


# ══════════════════════════════════════════════════════════════════════════════
#  REGRESIÓN LOGÍSTICA
# ══════════════════════════════════════════════════════════════════════════════
def run_logistic_regression(X, y, params: dict) -> dict:
    from sklearn.linear_model import LogisticRegression
    X, y = _to_numpy(X), _to_numpy(y, force_float=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"],
        random_state=int(params["random_state"]), stratify=y
    )
    penalty = None if params["penalty"] == "None" else params["penalty"]
    model = LogisticRegression(
        C=params["C"], penalty=penalty,
        solver=params["solver"], max_iter=int(params["max_iter"]),
        random_state=int(params["random_state"]),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    is_binary = len(model.classes_) == 2
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "CV Accuracy (5-fold)": cross_val_score(model, X, y, cv=5, scoring="accuracy").mean(),
    }
    if is_binary:
        from sklearn.metrics import roc_auc_score
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob[:, 1])

    return {
        "model": model,
        "metrics": metrics,
        "report": report,
        "train": (X_train, y_train),
        "test":  (X_test, y_test, y_pred, y_prob),
        "is_binary": is_binary,
        "classes": model.classes_,
    }


def plot_logistic_regression(result: dict) -> dict:
    X_test, y_test, y_pred, y_prob = result["test"]
    classes = result["classes"]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(
        cm, text_auto=True, aspect="auto",
        color_continuous_scale=[[0, "#e8f0f8"], [1, "#1e3a5f"]],
        title="Matriz de Confusión",
        labels=dict(x="Predicho", y="Real"),
        x=[str(c) for c in classes], y=[str(c) for c in classes],
    )
    fig_cm.update_layout(paper_bgcolor="white", font=dict(color=NAVY))

    figs = {"cm": fig_cm}

    # ROC curve (binary)
    if result["is_binary"]:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     line=dict(color=PALETTE[0], width=2.5),
                                     name=f"ROC (AUC = {roc_auc:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(color=GOLD, dash="dash"), name="Random"))
        fig_roc.update_layout(
            title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR",
            plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY),
        )

        prec, rec, _ = precision_recall_curve(y_test, y_prob[:, 1])
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                    line=dict(color=PALETTE[0], width=2.5), name="PR Curve"))
        fig_pr.update_layout(
            title="Curva Precision-Recall", xaxis_title="Recall", yaxis_title="Precision",
            plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY),
        )
        figs["roc"] = fig_roc
        figs["pr"] = fig_pr

    # Report
    df_report = pd.DataFrame(result["report"]).T.drop(
        ["accuracy", "macro avg", "weighted avg"], errors="ignore"
    ).reset_index().rename(columns={"index": "Clase"})
    fig_report = px.bar(
        df_report.melt(id_vars="Clase", value_vars=["precision", "recall", "f1-score"]),
        x="Clase", y="value", color="variable", barmode="group",
        title="Métricas por Clase",
        color_discrete_sequence=PALETTE,
    )
    fig_report.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY),
        yaxis_range=[0, 1], yaxis_title="Score",
    )
    figs["report"] = fig_report

    return figs


# ══════════════════════════════════════════════════════════════════════════════
#  SVM
# ══════════════════════════════════════════════════════════════════════════════
def run_svm(X, y, params: dict) -> dict:
    from sklearn.svm import SVC
    X, y = _to_numpy(X), _to_numpy(y, force_float=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"],
        random_state=int(params["random_state"]), stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = SVC(
        C=params["C"], kernel=params["kernel"], gamma=params["gamma"],
        probability=params["probability"], random_state=int(params["random_state"]),
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s) if params["probability"] else None

    is_binary = len(model.classes_) == 2
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "CV Accuracy (5-fold)": cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy").mean(),
        "Support Vectors": int(model.n_support_.sum()),
    }
    if is_binary and y_prob is not None:
        from sklearn.metrics import roc_auc_score
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob[:, 1])

    return {
        "model": model, "scaler": scaler, "metrics": metrics, "report": report,
        "train": (X_train_s, y_train),
        "test":  (X_test_s, y_test, y_pred, y_prob),
        "is_binary": is_binary, "classes": model.classes_,
    }

# SVM plots reusan los de logistic regression (mismas métricas de clasificación)
plot_svm = plot_logistic_regression


# ══════════════════════════════════════════════════════════════════════════════
#  K-MEANS
# ══════════════════════════════════════════════════════════════════════════════
def run_kmeans(X, params: dict) -> dict:
    from sklearn.cluster import KMeans
    X = _to_numpy(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=int(params["n_clusters"]),
        init=params["init"],
        n_init=int(params["n_init"]),
        max_iter=int(params["max_iter"]),
        random_state=int(params["random_state"]),
    )
    labels = model.fit_predict(X_scaled)

    sil  = silhouette_score(X_scaled, labels)
    db   = davies_bouldin_score(X_scaled, labels)
    ch   = calinski_harabasz_score(X_scaled, labels)

    # Elbow method
    inertias = []
    sil_scores = []
    K_range = range(2, min(11, X.shape[0]))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=int(params["random_state"]), n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        lbl = km.labels_
        if len(set(lbl)) > 1:
            sil_scores.append(silhouette_score(X_scaled, lbl))
        else:
            sil_scores.append(0)

    return {
        "model": model, "scaler": scaler, "labels": labels,
        "metrics": {"Inercia": model.inertia_, "Silhouette": sil,
                    "Davies-Bouldin": db, "Calinski-Harabasz": ch},
        "elbow": {"K": list(K_range), "inertia": inertias, "silhouette": sil_scores},
        "X_scaled": X_scaled,
    }


def plot_kmeans(result: dict, feature_names: list) -> dict:
    X_sc  = result["X_scaled"]
    labels = result["labels"]
    elbow = result["elbow"]

    # Scatter (primeras 2 dims o PCA)
    if X_sc.shape[1] >= 2:
        x_data, y_data = X_sc[:, 0], X_sc[:, 1]
        xname = feature_names[0] if feature_names else "Dim 1"
        yname = feature_names[1] if len(feature_names) > 1 else "Dim 2"
    else:
        x_data, y_data = X_sc[:, 0], np.zeros(len(X_sc))
        xname, yname = "Feature", ""

    fig_scatter = px.scatter(
        x=x_data, y=y_data, color=labels.astype(str),
        labels={"x": xname, "y": yname, "color": "Cluster"},
        title="Clusters (primeras 2 dimensiones)",
        color_discrete_sequence=PALETTE,
    )
    centroids = result["model"].cluster_centers_
    fig_scatter.add_trace(go.Scatter(
        x=centroids[:, 0], y=centroids[:, 1] if centroids.shape[1] > 1 else [0] * len(centroids),
        mode="markers",
        marker=dict(symbol="star", size=18, color=GOLD, line=dict(color="white", width=1)),
        name="Centroides",
    ))
    fig_scatter.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY))

    # Elbow
    fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
    fig_elbow.add_trace(go.Scatter(
        x=elbow["K"], y=elbow["inertia"], mode="lines+markers",
        name="Inercia", line=dict(color=PALETTE[0], width=2)
    ), secondary_y=False)
    fig_elbow.add_trace(go.Scatter(
        x=elbow["K"], y=elbow["silhouette"], mode="lines+markers",
        name="Silhouette", line=dict(color=GOLD, width=2)
    ), secondary_y=True)
    fig_elbow.update_layout(
        title="Elbow Method — Inercia y Silhouette vs K",
        plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY),
    )
    fig_elbow.update_yaxes(title_text="Inercia", secondary_y=False)
    fig_elbow.update_yaxes(title_text="Silhouette", secondary_y=True)

    # Cluster sizes
    cluster_counts = pd.Series(labels).value_counts().sort_index().reset_index()
    cluster_counts.columns = ["Cluster", "Tamaño"]
    fig_sizes = px.bar(
        cluster_counts, x="Cluster", y="Tamaño",
        title="Distribución de Clusters",
        color="Cluster", color_discrete_sequence=PALETTE,
    )
    fig_sizes.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY))

    return {"scatter": fig_scatter, "elbow": fig_elbow, "sizes": fig_sizes}


# ══════════════════════════════════════════════════════════════════════════════
#  PCA
# ══════════════════════════════════════════════════════════════════════════════
def run_pca(X, params: dict) -> dict:
    from sklearn.decomposition import PCA
    X = _to_numpy(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Análisis completo primero
    pca_full = PCA(random_state=int(params["random_state"]))
    pca_full.fit(X_scaled)

    n_comp = int(params["n_components"])
    pca = PCA(
        n_components=min(n_comp, X.shape[1]),
        svd_solver=params["svd_solver"],
        random_state=int(params["random_state"]),
    )
    X_reduced = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_reduced)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n95 = int(np.argmax(cumvar >= 0.95) + 1)

    recon_error = np.mean((X_scaled - X_reconstructed) ** 2)

    return {
        "model": pca, "scaler": scaler,
        "X_reduced": X_reduced, "X_scaled": X_scaled,
        "explained_ratio": pca.explained_variance_ratio_,
        "full_cumvar": cumvar,
        "full_ratio": pca_full.explained_variance_ratio_,
        "metrics": {
            "Varianza explicada (total)": float(np.sum(pca.explained_variance_ratio_)),
            "Componentes para 95% varianza": n95,
            "Error de reconstrucción (MSE)": recon_error,
        },
    }


def plot_pca(result: dict, feature_names: list) -> dict:
    X_red = result["X_reduced"]
    ratio = result["explained_ratio"]
    full_ratio = result["full_ratio"]
    cumvar = result["full_cumvar"]

    # Scree plot
    fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
    fig_scree.add_trace(go.Bar(
        x=[f"PC{i+1}" for i in range(len(full_ratio))],
        y=full_ratio * 100,
        name="% Varianza", marker_color=PALETTE[0],
    ), secondary_y=False)
    fig_scree.add_trace(go.Scatter(
        x=[f"PC{i+1}" for i in range(len(cumvar))],
        y=cumvar * 100, mode="lines+markers",
        name="% Acumulado", line=dict(color=GOLD, width=2)
    ), secondary_y=True)
    fig_scree.add_hline(y=95, line_color="red", line_dash="dash",
                        annotation_text="95%", secondary_y=True)
    fig_scree.update_layout(
        title="Scree Plot — Varianza Explicada por Componente",
        plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY),
    )

    figs = {"scree": fig_scree}

    # 2D scatter
    if X_red.shape[1] >= 2:
        fig_2d = px.scatter(
            x=X_red[:, 0], y=X_red[:, 1],
            labels={
                "x": f"PC1 ({ratio[0]*100:.1f}%)",
                "y": f"PC2 ({ratio[1]*100:.1f}%)",
            },
            title="Proyección 2D — PC1 vs PC2",
        )
        fig_2d.update_traces(marker=dict(color=PALETTE[0], size=7, opacity=0.7))
        fig_2d.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY))
        figs["2d"] = fig_2d

    # Loadings (biplot components)
    if feature_names and len(feature_names) == result["model"].components_.shape[1]:
        comps = result["model"].components_
        df_load = pd.DataFrame(
            comps[:3].T,
            index=feature_names,
            columns=[f"PC{i+1}" for i in range(min(3, comps.shape[0]))]
        ).reset_index().rename(columns={"index": "Variable"})

        fig_load = px.bar(
            df_load.melt(id_vars="Variable", var_name="Componente", value_name="Carga"),
            x="Variable", y="Carga", color="Componente", barmode="group",
            title="Loadings (contribución de variables a cada PC)",
            color_discrete_sequence=PALETTE,
        )
        fig_load.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY),
        )
        figs["loadings"] = fig_load

    return figs


# ══════════════════════════════════════════════════════════════════════════════
#  ÁRBOL DE DECISIÓN
# ══════════════════════════════════════════════════════════════════════════════
def run_decision_tree(X, y, params: dict) -> dict:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    X, y_raw = _to_numpy(X), _to_numpy(y, force_float=False)
    is_reg = False
    try:
        y_num = y_raw.astype(float)
        unique = np.unique(y_num)
        is_reg = len(unique) > 20
    except (ValueError, TypeError):
        y_num = y_raw

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_raw, test_size=params["test_size"], random_state=int(params["random_state"])
    )
    md = params["max_depth"]
    md = None if md >= 20 else int(md)

    if is_reg:
        model = DecisionTreeRegressor(
            max_depth=md,
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=int(params["random_state"]),
        )
        model.fit(X_train, y_train.astype(float))
        y_pred = model.predict(X_test)
        y_test_f = y_test.astype(float)
        mse = mean_squared_error(y_test_f, y_pred)
        metrics = {
            "R²": r2_score(y_test_f, y_pred),
            "RMSE": float(np.sqrt(mse)),
            "MAE": mean_absolute_error(y_test_f, y_pred),
            "Profundidad": model.get_depth(),
            "Nodos hoja": model.get_n_leaves(),
        }
        report = None
        is_classifier = False
    else:
        model = DecisionTreeClassifier(
            max_depth=md,
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            criterion=params["criterion"],
            random_state=int(params["random_state"]),
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cv = cross_val_score(model, X, y_raw, cv=5, scoring="accuracy").mean()
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "CV Accuracy (5-fold)": cv,
            "Profundidad": model.get_depth(),
            "Nodos hoja": model.get_n_leaves(),
        }
        is_classifier = True

    importances = pd.DataFrame({
        "Variable": [f"X{i}" for i in range(X.shape[1])],
        "Importancia": model.feature_importances_,
    }).sort_values("Importancia", ascending=False)

    return {
        "model": model, "metrics": metrics, "report": report,
        "train": (X_train, y_train), "test": (X_test, y_test, y_pred),
        "importances": importances, "is_classifier": is_classifier,
        "is_reg": is_reg,
    }


def plot_decision_tree(result: dict, feature_names=None) -> dict:
    X_test, y_test, y_pred = result["test"]
    importances = result["importances"]
    if feature_names and len(feature_names) == len(importances):
        importances = importances.copy()
        importances["Variable"] = feature_names

    fig_imp = px.bar(
        importances.head(15), x="Importancia", y="Variable", orientation="h",
        color="Importancia", color_continuous_scale=["#e8f0f8", "#1e3a5f"],
        title="Importancia de Variables",
    )
    fig_imp.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          font=dict(color=NAVY), coloraxis_showscale=False)

    figs = {"importances": fig_imp}

    if result["is_classifier"]:
        model = result["model"]
        cm = confusion_matrix(y_test, y_pred)
        classes = model.classes_
        fig_cm = px.imshow(
            cm, text_auto=True, aspect="auto",
            color_continuous_scale=[[0, "#e8f0f8"], [1, "#1e3a5f"]],
            title="Matriz de Confusión",
            x=[str(c) for c in classes], y=[str(c) for c in classes],
        )
        fig_cm.update_layout(paper_bgcolor="white", font=dict(color=NAVY))
        figs["cm"] = fig_cm

        report = result.get("report")
        if report:
            df_report = pd.DataFrame(report).T.drop(
                ["accuracy", "macro avg", "weighted avg"], errors="ignore"
            ).reset_index().rename(columns={"index": "Clase"})
            fig_report = px.bar(
                df_report.melt(id_vars="Clase", value_vars=["precision", "recall", "f1-score"]),
                x="Clase", y="value", color="variable", barmode="group",
                title="Métricas por Clase", color_discrete_sequence=PALETTE,
            )
            fig_report.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                     font=dict(color=NAVY), yaxis_range=[0, 1])
            figs["report"] = fig_report
    else:
        y_test_f = y_test.astype(float) if hasattr(y_test, 'astype') else np.array(y_test, dtype=float)
        y_pred_f = np.array(y_pred, dtype=float)
        residuals = y_test_f - y_pred_f
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=y_pred_f, y=residuals, mode="markers",
                                     marker=dict(color=PALETTE[0], size=7, opacity=0.7)))
        fig_res.add_hline(y=0, line_color=GOLD, line_dash="dash")
        fig_res.update_layout(title="Residuos", xaxis_title="Predicción", yaxis_title="Residuo",
                               plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY))
        figs["residuals"] = fig_res

    return figs


# ══════════════════════════════════════════════════════════════════════════════
#  RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
def run_random_forest(X, y, params: dict) -> dict:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    X, y_raw = _to_numpy(X), _to_numpy(y, force_float=False)
    try:
        y_num = y_raw.astype(float)
        is_reg = len(np.unique(y_num)) > 20
    except (ValueError, TypeError):
        is_reg = False

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_raw, test_size=params["test_size"], random_state=int(params["random_state"])
    )
    md = params["max_depth"]
    md = None if md >= 30 else int(md)
    mf = params["max_features"]
    if mf == "None":
        mf = None

    if is_reg:
        model = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=md,
            min_samples_split=int(params["min_samples_split"]),
            max_features=mf,
            random_state=int(params["random_state"]),
            oob_score=True, n_jobs=-1,
        )
        model.fit(X_train, y_train.astype(float))
        y_pred = model.predict(X_test)
        y_test_f = y_test.astype(float)
        mse = mean_squared_error(y_test_f, y_pred)
        metrics = {
            "R²": r2_score(y_test_f, y_pred),
            "RMSE": float(np.sqrt(mse)),
            "MAE": mean_absolute_error(y_test_f, y_pred),
            "OOB Score": model.oob_score_,
        }
        report = None
        is_classifier = False
        y_prob = None
    else:
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=md,
            min_samples_split=int(params["min_samples_split"]),
            max_features=mf,
            random_state=int(params["random_state"]),
            oob_score=True, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        is_binary = len(model.classes_) == 2
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "CV Accuracy (5-fold)": cross_val_score(model, X, y_raw, cv=5, scoring="accuracy").mean(),
            "OOB Score": model.oob_score_,
        }
        if is_binary:
            from sklearn.metrics import roc_auc_score
            metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob[:, 1])
        is_classifier = True

    importances = pd.DataFrame({
        "Variable": [f"X{i}" for i in range(X.shape[1])],
        "Importancia": model.feature_importances_,
    }).sort_values("Importancia", ascending=False)

    return {
        "model": model, "metrics": metrics, "report": report,
        "train": (X_train, y_train), "test": (X_test, y_test, y_pred),
        "importances": importances, "is_classifier": is_classifier,
        "is_reg": is_reg,
        "classes": model.classes_ if is_classifier else None,
        "y_prob": y_prob if is_classifier else None,
    }


def plot_random_forest(result: dict, feature_names=None) -> dict:
    importances = result["importances"].copy()
    if feature_names and len(feature_names) == len(importances):
        importances["Variable"] = feature_names

    fig_imp = px.bar(
        importances.head(15), x="Importancia", y="Variable", orientation="h",
        color="Importancia", color_continuous_scale=["#e8f0f8", "#1e3a5f"],
        title="Importancia de Variables (Feature Importance)",
    )
    fig_imp.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          font=dict(color=NAVY), coloraxis_showscale=False)
    figs = {"importances": fig_imp}

    if result["is_classifier"]:
        X_test, y_test, y_pred = result["test"]
        classes = result["classes"]
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, aspect="auto",
            color_continuous_scale=[[0, "#e8f0f8"], [1, "#1e3a5f"]],
            title="Matriz de Confusión",
            x=[str(c) for c in classes], y=[str(c) for c in classes],
        )
        fig_cm.update_layout(paper_bgcolor="white", font=dict(color=NAVY))
        figs["cm"] = fig_cm

        is_binary = len(classes) == 2
        y_prob = result.get("y_prob")
        if is_binary and y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         line=dict(color=PALETTE[0], width=2.5),
                                         name=f"ROC (AUC={roc_auc:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(color=GOLD, dash="dash"), name="Random"))
            fig_roc.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR",
                                  plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY))
            figs["roc"] = fig_roc

        if result.get("report"):
            df_r = pd.DataFrame(result["report"]).T.drop(
                ["accuracy", "macro avg", "weighted avg"], errors="ignore"
            ).reset_index().rename(columns={"index": "Clase"})
            fig_r = px.bar(
                df_r.melt(id_vars="Clase", value_vars=["precision", "recall", "f1-score"]),
                x="Clase", y="value", color="variable", barmode="group",
                title="Métricas por Clase", color_discrete_sequence=PALETTE,
            )
            fig_r.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                 font=dict(color=NAVY), yaxis_range=[0, 1])
            figs["report"] = fig_r

    return figs


# ══════════════════════════════════════════════════════════════════════════════
#  KNN
# ══════════════════════════════════════════════════════════════════════════════
def run_knn(X, y, params: dict) -> dict:
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    X, y_raw = _to_numpy(X), _to_numpy(y, force_float=False)
    try:
        y_num = y_raw.astype(float)
        is_reg = len(np.unique(y_num)) > 20
    except (ValueError, TypeError):
        is_reg = False

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_raw, test_size=params["test_size"], random_state=int(params["random_state"])
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    k = int(params["n_neighbors"])
    if is_reg:
        model = KNeighborsRegressor(
            n_neighbors=k, weights=params["weights"],
            metric=params["metric"], p=int(params["p"]),
        )
        model.fit(X_train_s, y_train.astype(float))
        y_pred = model.predict(X_test_s)
        y_test_f = y_test.astype(float)
        mse = mean_squared_error(y_test_f, y_pred)
        metrics = {
            "R²": r2_score(y_test_f, y_pred),
            "RMSE": float(np.sqrt(mse)),
            "MAE": mean_absolute_error(y_test_f, y_pred),
            "K vecinos": k,
        }
        report = None
        is_classifier = False
        y_prob = None
    else:
        model = KNeighborsClassifier(
            n_neighbors=k, weights=params["weights"],
            metric=params["metric"], p=int(params["p"]),
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)
        is_binary = len(model.classes_) == 2
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "CV Accuracy (5-fold)": cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy").mean(),
            "K vecinos": k,
        }
        if is_binary:
            from sklearn.metrics import roc_auc_score
            metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob[:, 1])
        is_classifier = True

    k_range = range(1, min(21, len(X_train_s)))
    k_scores = []
    for ki in k_range:
        if is_reg:
            m = KNeighborsRegressor(n_neighbors=ki, weights=params["weights"])
            sc = cross_val_score(m, X_train_s, y_train.astype(float), cv=3, scoring="r2").mean()
        else:
            m = KNeighborsClassifier(n_neighbors=ki, weights=params["weights"])
            sc = cross_val_score(m, X_train_s, y_train, cv=3, scoring="accuracy").mean()
        k_scores.append(sc)

    return {
        "model": model, "scaler": scaler, "metrics": metrics, "report": report,
        "train": (X_train_s, y_train), "test": (X_test_s, y_test, y_pred),
        "is_classifier": is_classifier, "is_reg": is_reg,
        "k_range": list(k_range), "k_scores": k_scores,
        "classes": model.classes_ if is_classifier else None,
        "y_prob": y_prob,
    }


def plot_knn(result: dict, feature_names=None) -> dict:
    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(
        x=result["k_range"], y=result["k_scores"], mode="lines+markers",
        line=dict(color=PALETTE[0], width=2), marker=dict(color=GOLD, size=8),
        name="Score CV",
    ))
    best_k = result["k_range"][int(np.argmax(result["k_scores"]))]
    fig_k.add_vline(x=best_k, line_color=GOLD, line_dash="dash",
                    annotation_text=f"K={best_k}")
    label = "R² CV" if result["is_reg"] else "Accuracy CV"
    fig_k.update_layout(title=f"K vs {label} (CV-3)", xaxis_title="K",
                        yaxis_title=label, plot_bgcolor="white",
                        paper_bgcolor="white", font=dict(color=NAVY))
    figs = {"k_curve": fig_k}

    if result["is_classifier"]:
        X_test, y_test, y_pred = result["test"]
        classes = result["classes"]
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, aspect="auto",
            color_continuous_scale=[[0, "#e8f0f8"], [1, "#1e3a5f"]],
            title="Matriz de Confusión",
            x=[str(c) for c in classes], y=[str(c) for c in classes],
        )
        fig_cm.update_layout(paper_bgcolor="white", font=dict(color=NAVY))
        figs["cm"] = fig_cm

        is_binary = len(classes) == 2
        y_prob = result.get("y_prob")
        if is_binary and y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         line=dict(color=PALETTE[0], width=2.5),
                                         name=f"ROC (AUC={roc_auc:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(color=GOLD, dash="dash"), name="Random"))
            fig_roc.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR",
                                  plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY))
            figs["roc"] = fig_roc

    return figs


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIENT BOOSTING
# ══════════════════════════════════════════════════════════════════════════════
def run_gradient_boosting(X, y, params: dict) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    X, y_raw = _to_numpy(X), _to_numpy(y, force_float=False)
    try:
        y_num = y_raw.astype(float)
        is_reg = len(np.unique(y_num)) > 20
    except (ValueError, TypeError):
        is_reg = False

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_raw, test_size=params["test_size"], random_state=int(params["random_state"])
    )

    if is_reg:
        model = GradientBoostingRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            random_state=int(params["random_state"]),
        )
        model.fit(X_train, y_train.astype(float))
        y_pred = model.predict(X_test)
        y_test_f = y_test.astype(float)
        mse = mean_squared_error(y_test_f, y_pred)
        metrics = {
            "R²": r2_score(y_test_f, y_pred),
            "RMSE": float(np.sqrt(mse)),
            "MAE": mean_absolute_error(y_test_f, y_pred),
        }
        report = None
        is_classifier = False
        y_prob = None
    else:
        model = GradientBoostingClassifier(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            random_state=int(params["random_state"]),
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        is_binary = len(model.classes_) == 2
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "CV Accuracy (5-fold)": cross_val_score(model, X, y_raw, cv=5, scoring="accuracy").mean(),
        }
        if is_binary:
            from sklearn.metrics import roc_auc_score
            metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob[:, 1])
        is_classifier = True

    importances = pd.DataFrame({
        "Variable": [f"X{i}" for i in range(X.shape[1])],
        "Importancia": model.feature_importances_,
    }).sort_values("Importancia", ascending=False)

    n_est = int(params["n_estimators"])
    step = max(1, n_est // 50)
    iters = list(range(0, n_est, step))
    y_tr = y_train.astype(float) if is_reg else y_train
    train_scores = [
        mean_squared_error(y_tr, list(model.staged_predict(X_train))[i])
        for i in iters
    ]

    return {
        "model": model, "metrics": metrics, "report": report,
        "train": (X_train, y_train), "test": (X_test, y_test, y_pred),
        "importances": importances, "is_classifier": is_classifier, "is_reg": is_reg,
        "classes": model.classes_ if is_classifier else None,
        "y_prob": y_prob,
        "train_scores": train_scores, "iters": iters,
    }


def plot_gradient_boosting(result: dict, feature_names=None) -> dict:
    importances = result["importances"].copy()
    if feature_names and len(feature_names) == len(importances):
        importances["Variable"] = feature_names

    fig_imp = px.bar(
        importances.head(15), x="Importancia", y="Variable", orientation="h",
        color="Importancia", color_continuous_scale=["#e8f0f8", "#1e3a5f"],
        title="Importancia de Variables",
    )
    fig_imp.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          font=dict(color=NAVY), coloraxis_showscale=False)

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=result["iters"], y=result["train_scores"],
                                  mode="lines", line=dict(color=PALETTE[0], width=2),
                                  name="Train Loss"))
    fig_loss.update_layout(title="Curva de pérdida por iteración",
                           xaxis_title="Iteración", yaxis_title="Loss",
                           plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY))

    figs = {"importances": fig_imp, "loss": fig_loss}

    if result["is_classifier"]:
        X_test, y_test, y_pred = result["test"]
        classes = result["classes"]
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, aspect="auto",
            color_continuous_scale=[[0, "#e8f0f8"], [1, "#1e3a5f"]],
            title="Matriz de Confusión",
            x=[str(c) for c in classes], y=[str(c) for c in classes],
        )
        fig_cm.update_layout(paper_bgcolor="white", font=dict(color=NAVY))
        figs["cm"] = fig_cm

        is_binary = len(classes) == 2
        y_prob = result.get("y_prob")
        if is_binary and y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         line=dict(color=PALETTE[0], width=2.5),
                                         name=f"ROC (AUC={roc_auc:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(color=GOLD, dash="dash"), name="Random"))
            fig_roc.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR",
                                  plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY))
            figs["roc"] = fig_roc

    return figs


# ══════════════════════════════════════════════════════════════════════════════
#  NAIVE BAYES
# ══════════════════════════════════════════════════════════════════════════════
def run_naive_bayes(X, y, params: dict) -> dict:
    from sklearn.naive_bayes import GaussianNB
    X, y_raw = _to_numpy(X), _to_numpy(y, force_float=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_raw, test_size=params["test_size"], random_state=int(params["random_state"])
    )
    model = GaussianNB(var_smoothing=float(params["var_smoothing"]))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    is_binary = len(model.classes_) == 2
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "CV Accuracy (5-fold)": cross_val_score(model, X, y_raw, cv=5, scoring="accuracy").mean(),
    }
    if is_binary:
        from sklearn.metrics import roc_auc_score
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob[:, 1])

    return {
        "model": model, "metrics": metrics, "report": report,
        "train": (X_train, y_train), "test": (X_test, y_test, y_pred, y_prob),
        "is_binary": is_binary, "classes": model.classes_,
    }


def plot_naive_bayes(result: dict) -> dict:
    return plot_logistic_regression(result)


# ══════════════════════════════════════════════════════════════════════════════
#  DBSCAN
# ══════════════════════════════════════════════════════════════════════════════
def run_dbscan(X, params: dict) -> dict:
    from sklearn.cluster import DBSCAN
    X = _to_numpy(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DBSCAN(
        eps=float(params["eps"]),
        min_samples=int(params["min_samples"]),
        metric=params["metric"],
    )
    labels = model.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = int(np.sum(labels == -1))

    metrics = {"Clusters encontrados": n_clusters, "Outliers detectados": n_outliers}
    mask = labels != -1
    if n_clusters > 1 and mask.sum() > 1:
        metrics["Silhouette (sin outliers)"] = silhouette_score(X_scaled[mask], labels[mask])
        metrics["Davies-Bouldin"] = davies_bouldin_score(X_scaled[mask], labels[mask])

    eps_range = np.linspace(max(0.05, float(params["eps"]) * 0.3),
                             float(params["eps"]) * 2.5, 20)
    eps_clusters = []
    eps_outliers = []
    for e in eps_range:
        lbl = DBSCAN(eps=e, min_samples=int(params["min_samples"])).fit_predict(X_scaled)
        eps_clusters.append(len(set(lbl)) - (1 if -1 in lbl else 0))
        eps_outliers.append(int(np.sum(lbl == -1)))

    return {
        "model": model, "scaler": scaler, "labels": labels,
        "metrics": metrics, "X_scaled": X_scaled,
        "eps_range": eps_range.tolist(),
        "eps_clusters": eps_clusters, "eps_outliers": eps_outliers,
    }


def plot_dbscan(result: dict, feature_names: list) -> dict:
    X_sc = result["X_scaled"]
    labels = result["labels"]

    label_names = ["Outlier" if l == -1 else f"Cluster {l}" for l in labels]

    if X_sc.shape[1] >= 2:
        x_data, y_data = X_sc[:, 0], X_sc[:, 1]
        xname = feature_names[0] if feature_names else "Dim 1"
        yname = feature_names[1] if len(feature_names) > 1 else "Dim 2"
    else:
        x_data, y_data = X_sc[:, 0], np.zeros(len(X_sc))
        xname, yname = "Feature", ""

    fig_scatter = px.scatter(
        x=x_data, y=y_data, color=label_names,
        labels={"x": xname, "y": yname, "color": "Grupo"},
        title="DBSCAN — Clusters y Outliers",
        color_discrete_sequence=PALETTE + ["#c0392b"],
    )
    fig_scatter.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY))

    fig_eps = make_subplots(specs=[[{"secondary_y": True}]])
    fig_eps.add_trace(go.Scatter(x=result["eps_range"], y=result["eps_clusters"],
                                  mode="lines+markers", name="Clusters",
                                  line=dict(color=PALETTE[0], width=2)), secondary_y=False)
    fig_eps.add_trace(go.Scatter(x=result["eps_range"], y=result["eps_outliers"],
                                  mode="lines+markers", name="Outliers",
                                  line=dict(color="#c0392b", width=2)), secondary_y=True)
    fig_eps.add_vline(x=float(result["model"].eps), line_color=GOLD, line_dash="dash",
                      annotation_text=f"eps={result['model'].eps}")
    fig_eps.update_layout(title="Sensibilidad a eps", plot_bgcolor="white",
                          paper_bgcolor="white", font=dict(color=NAVY))
    fig_eps.update_yaxes(title_text="N° Clusters", secondary_y=False)
    fig_eps.update_yaxes(title_text="N° Outliers", secondary_y=True)

    cluster_counts = pd.Series(labels).value_counts().sort_index().reset_index()
    cluster_counts.columns = ["Cluster", "Tamaño"]
    cluster_counts["Nombre"] = cluster_counts["Cluster"].apply(
        lambda x: "Outlier" if x == -1 else f"Cluster {x}"
    )
    fig_sizes = px.bar(
        cluster_counts, x="Nombre", y="Tamaño", title="Tamaño de Clusters",
        color="Nombre", color_discrete_sequence=PALETTE + ["#c0392b"],
    )
    fig_sizes.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY))

    return {"scatter": fig_scatter, "eps": fig_eps, "sizes": fig_sizes}


# ══════════════════════════════════════════════════════════════════════════════
#  HIERARCHICAL CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
def run_hierarchical(X, params: dict) -> dict:
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    X = _to_numpy(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n = int(params["n_clusters"])
    lnk = params["linkage"]
    aff = params["affinity"]
    if lnk == "ward":
        aff = "euclidean"

    model = AgglomerativeClustering(n_clusters=n, linkage=lnk, metric=aff)
    labels = model.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)

    Z = scipy_linkage(X_scaled, method=lnk)

    k_range = range(2, min(10, X_scaled.shape[0]))
    sil_scores = []
    for k in k_range:
        m = AgglomerativeClustering(n_clusters=k, linkage=lnk, metric=aff)
        lbl = m.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, lbl) if len(set(lbl)) > 1 else 0)

    return {
        "model": model, "scaler": scaler, "labels": labels,
        "metrics": {"Silhouette": sil, "Davies-Bouldin": db, "Calinski-Harabasz": ch},
        "X_scaled": X_scaled, "linkage_matrix": Z,
        "k_range": list(k_range), "sil_scores": sil_scores,
    }


def plot_hierarchical(result: dict, feature_names: list) -> dict:
    import plotly.figure_factory as ff

    X_sc = result["X_scaled"]
    labels = result["labels"]
    Z = result["linkage_matrix"]

    if X_sc.shape[1] >= 2:
        x_data, y_data = X_sc[:, 0], X_sc[:, 1]
        xname = feature_names[0] if feature_names else "Dim 1"
        yname = feature_names[1] if len(feature_names) > 1 else "Dim 2"
    else:
        x_data, y_data = X_sc[:, 0], np.zeros(len(X_sc))
        xname, yname = "Feature", ""

    fig_scatter = px.scatter(
        x=x_data, y=y_data, color=labels.astype(str),
        labels={"x": xname, "y": yname, "color": "Cluster"},
        title="Clusters jerárquicos",
        color_discrete_sequence=PALETTE,
    )
    fig_scatter.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY))

    try:
        fig_dend = ff.create_dendrogram(X_sc, linkagefun=lambda x: Z)
        fig_dend.update_layout(title="Dendrograma", xaxis_title="Muestras",
                                yaxis_title="Distancia",
                                plot_bgcolor="white", paper_bgcolor="white",
                                font=dict(color=NAVY))
    except Exception:
        fig_dend = go.Figure()
        fig_dend.add_annotation(text="Dendrograma no disponible para este dataset",
                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig_dend.update_layout(paper_bgcolor="white")

    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(x=result["k_range"], y=result["sil_scores"],
                                  mode="lines+markers", line=dict(color=PALETTE[0], width=2),
                                  marker=dict(color=GOLD, size=8), name="Silhouette"))
    best_k = result["k_range"][int(np.argmax(result["sil_scores"]))]
    fig_sil.add_vline(x=best_k, line_color=GOLD, line_dash="dash",
                      annotation_text=f"K={best_k}")
    fig_sil.update_layout(title="Silhouette vs K clusters", xaxis_title="K",
                           yaxis_title="Silhouette", plot_bgcolor="white",
                           paper_bgcolor="white", font=dict(color=NAVY))

    cluster_counts = pd.Series(labels).value_counts().sort_index().reset_index()
    cluster_counts.columns = ["Cluster", "Tamaño"]
    fig_sizes = px.bar(
        cluster_counts, x="Cluster", y="Tamaño", title="Tamaño de Clusters",
        color="Cluster", color_discrete_sequence=PALETTE,
    )
    fig_sizes.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY))

    return {"scatter": fig_scatter, "dendrogram": fig_dend, "silhouette": fig_sil, "sizes": fig_sizes}
