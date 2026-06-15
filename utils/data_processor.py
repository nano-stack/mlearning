"""
Utilidades para carga y transformación de datos (Power Query style).
"""
import pandas as pd
import numpy as np
from io import StringIO


ENCODINGS = ["utf-8", "latin-1", "iso-8859-1", "cp1252", "utf-16"]
SEPARATORS = {
    "Coma (,)":      ",",
    "Punto y coma (;)": ";",
    "Tabulación (\\t)": "\t",
    "Pipe (|)":      "|",
    "Espacio":       " ",
}


def load_csv(file_bytes: bytes, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    try:
        text = file_bytes.decode(encoding)
        df = pd.read_csv(StringIO(text), sep=sep, engine="python")
        return df
    except Exception as e:
        raise ValueError(f"No se pudo leer el archivo: {e}")


def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for col in df.columns:
        s = df[col]
        summary.append({
            "Columna":   col,
            "Tipo":      str(s.dtype),
            "No nulos":  s.notna().sum(),
            "Nulos":     s.isna().sum(),
            "% Nulos":   f"{100 * s.isna().mean():.1f}%",
            "Únicos":    s.nunique(),
            "Muestra":   str(s.dropna().iloc[0]) if s.notna().any() else "—",
        })
    return pd.DataFrame(summary)


def auto_detect_types(df: pd.DataFrame) -> dict:
    """Detecta tipos sugeridos para cada columna."""
    types = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            types[col] = "numérico"
        elif s.nunique() <= 20:
            types[col] = "categórico"
        else:
            try:
                pd.to_datetime(s, infer_datetime_format=True)
                types[col] = "fecha"
            except Exception:
                types[col] = "texto"
    return types


def apply_transformations(df: pd.DataFrame, transforms: dict) -> pd.DataFrame:
    """
    transforms = {
        "col_name": {
            "action": "encode_label" | "encode_onehot" | "drop" | "fillna_mean" |
                      "fillna_median" | "fillna_mode" | "to_numeric" | "normalize"
        }
    }
    """
    df = df.copy()
    for col, cfg in transforms.items():
        if col not in df.columns:
            continue
        action = cfg.get("action", "")

        if action == "drop":
            df = df.drop(columns=[col])

        elif action == "encode_label":
            df[col] = df[col].astype("category").cat.codes

        elif action == "encode_onehot":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        elif action == "fillna_mean":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].mean())

        elif action == "fillna_median":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

        elif action == "fillna_mode":
            df[col] = df[col].fillna(df[col].mode()[0])

        elif action == "to_numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")

        elif action == "normalize":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            mn, mx = df[col].min(), df[col].max()
            if mx != mn:
                df[col] = (df[col] - mn) / (mx - mn)

        elif action == "standardize":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            mu, sigma = df[col].mean(), df[col].std()
            if sigma > 0:
                df[col] = (df[col] - mu) / sigma

    return df


def get_numeric_columns(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list:
    return df.select_dtypes(exclude=[np.number]).columns.tolist()


# ─── Datasets de ejemplo integrados ──────────────────────────────────────────

def get_example_datasets() -> dict:
    """Datasets de ejemplo listos para usar sin carga de archivo."""
    from sklearn.datasets import (
        make_regression, make_classification, make_blobs, load_iris
    )

    # Regresión
    X_r, y_r = make_regression(n_samples=200, n_features=3, noise=15, random_state=42)
    df_reg = pd.DataFrame(X_r, columns=["feature_1", "feature_2", "feature_3"])
    df_reg["target"] = y_r

    # Clasificación binaria
    X_c, y_c = make_classification(n_samples=200, n_features=4, n_informative=2,
                                   n_redundant=1, n_repeated=0, random_state=42)
    df_cls = pd.DataFrame(X_c, columns=[f"feat_{i}" for i in range(4)])
    df_cls["label"] = y_c

    # Clustering
    X_b, y_b = make_blobs(n_samples=200, centers=3, n_features=2, cluster_std=0.9, random_state=42)
    df_blobs = pd.DataFrame(X_b, columns=["x1", "x2"])
    df_blobs["true_cluster"] = y_b

    # Iris (clasificación multiclase)
    iris = load_iris(as_frame=True)
    df_iris = iris.frame

    return {
        "Regresion Lineal (sintetico)": df_reg,
        "Clasificacion Binaria (sintetico)": df_cls,
        "Clustering (3 grupos)": df_blobs,
        "Iris (multiclase)": df_iris,
    }
