"""
Knowledge base local — sin API externa.
Toda la información didáctica de los modelos está aquí embebida.
"""

MODELS_KNOWLEDGE = {
    # ─────────────────────────────────────────────
    #  REGRESIÓN LINEAL
    # ─────────────────────────────────────────────
    "linear_regression": {
        "name": "Regresión Lineal",
        "type": "supervisado",
        "task": "regresión",
        "icon": "📈",
        "one_liner": "Predice un valor continuo ajustando una línea recta (o plano) a los datos.",
        "when_to_use": [
            "La variable de salida es continua (precio, temperatura, ventas).",
            "Esperas una relación aproximadamente lineal entre entradas y salida.",
            "Necesitas un modelo interpretable: los coeficientes dicen cuánto pesa cada variable.",
            "Cuentas con pocos datos y quieres evitar sobreajuste.",
        ],
        "when_not_to_use": [
            "La relación entre variables es muy no lineal.",
            "La variable objetivo es categórica (usa Regresión Logística o árboles).",
            "Hay muchos outliers sin tratar (son muy influyentes en la recta).",
        ],
        "math_concept": (
            "Ajusta β₀ y β₁…βₙ para minimizar el Error Cuadrático Medio (MSE):\n"
            "ŷ = β₀ + β₁x₁ + … + βₙxₙ\n"
            "Costo: MSE = (1/n) Σ(yᵢ − ŷᵢ)²"
        ),
        "code_template": '''\
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Separar datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state={random_state}
)

# 2. Crear y entrenar modelo
model = LinearRegression(
    fit_intercept={fit_intercept},
    positive={positive}
)
model.fit(X_train, y_train)

# 3. Predecir y evaluar
y_pred = model.predict(X_test)
print(f"MSE  : {{mean_squared_error(y_test, y_pred):.4f}}")
print(f"RMSE : {{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}}")
print(f"R²   : {{r2_score(y_test, y_pred):.4f}}")
print(f"Coeficientes: {{model.coef_}}")
print(f"Intercepto  : {{model.intercept_:.4f}}")
''',
        "params": {
            "test_size": {
                "label": "Tamaño del conjunto de prueba (test_size)",
                "is_hyperparam": False,
                "type": "slider",
                "min": 0.1, "max": 0.5, "default": 0.2, "step": 0.05,
                "what_it_is": (
                    "No es un hiperparámetro del modelo — es una decisión de diseño "
                    "experimental. Controla qué fracción de los datos se reserva para "
                    "evaluar el modelo después del entrenamiento."
                ),
                "effect": (
                    "Valores bajos (0.1–0.15): más datos para entrenar, pero la evaluación "
                    "puede ser poco confiable. Valores altos (0.3–0.4): evaluación más robusta, "
                    "pero menos datos para aprender."
                ),
                "rule_of_thumb": "Usa 0.2 (80/20) como punto de partida estándar.",
            },
            "random_state": {
                "label": "Semilla aleatoria (random_state)",
                "is_hyperparam": False,
                "type": "number",
                "min": 0, "max": 999, "default": 42,
                "what_it_is": (
                    "No es un hiperparámetro — es una semilla para el generador de números "
                    "aleatorios. Garantiza que el split de datos sea reproducible entre ejecuciones."
                ),
                "effect": "Cambiarla cambia qué muestras van a train/test, pero no el modelo en sí.",
                "rule_of_thumb": "Usa 42 o cualquier entero fijo para reproducibilidad.",
            },
            "fit_intercept": {
                "label": "Ajustar intercepto (fit_intercept)",
                "is_hyperparam": True,
                "type": "checkbox",
                "default": True,
                "what_it_is": (
                    "Hiperparámetro estructural. Decide si el modelo incluye β₀ "
                    "(el término independiente — donde la recta corta el eje Y)."
                ),
                "effect": (
                    "True (recomendado): la recta puede estar desplazada del origen. "
                    "False: fuerza que la predicción sea 0 cuando todas las entradas son 0. "
                    "Úsalo en False solo si tienes certeza teórica de que esa restricción es válida."
                ),
                "rule_of_thumb": "Déjalo en True a menos que tengas razón teórica para cambiarlo.",
            },
            "positive": {
                "label": "Coeficientes positivos (positive)",
                "is_hyperparam": True,
                "type": "checkbox",
                "default": False,
                "what_it_is": (
                    "Hiperparámetro de restricción. Si está activo, obliga a que todos los "
                    "coeficientes βᵢ sean ≥ 0 (non-negative least squares)."
                ),
                "effect": (
                    "Útil cuando el dominio garantiza relaciones positivas (ej: más metros → mayor precio). "
                    "Fuera de ese contexto, puede empeorar el ajuste."
                ),
                "rule_of_thumb": "Déjalo en False salvo que el problema exija coeficientes positivos.",
            },
        },
        "metrics": {
            "MSE": "Error Cuadrático Medio. Promedio de (y - ŷ)². Penaliza errores grandes. Unidad: variable².",
            "RMSE": "Raíz del MSE. Misma unidad que la variable objetivo. Más interpretable.",
            "R²": "Coeficiente de determinación. 1 = ajuste perfecto, 0 = modelo no mejor que la media, <0 = peor.",
            "MAE": "Error Absoluto Medio. Promedio de |y - ŷ|. Más robusto a outliers que MSE.",
        },
    },

    # ─────────────────────────────────────────────
    #  REGRESIÓN LOGÍSTICA
    # ─────────────────────────────────────────────
    "logistic_regression": {
        "name": "Regresión Logística",
        "type": "supervisado",
        "task": "clasificación",
        "icon": "🔀",
        "one_liner": "Clasifica observaciones en categorías estimando la probabilidad de pertenencia a cada clase.",
        "when_to_use": [
            "La variable de salida es categórica (binaria o multiclase).",
            "Necesitas probabilidades de predicción (no solo la clase).",
            "Quieres un modelo rápido, interpretable y que funcione bien con pocas muestras.",
            "Las clases son linealmente separables o casi separables.",
        ],
        "when_not_to_use": [
            "Las clases no son linealmente separables (considera SVM con kernel o árboles).",
            "Hay demasiadas características correlacionadas sin regularización.",
            "El problema requiere capturar interacciones complejas.",
        ],
        "math_concept": (
            "Aplica la función sigmoide a una combinación lineal:\n"
            "P(y=1|x) = σ(β₀ + β₁x₁ + … + βₙxₙ)\n"
            "donde σ(z) = 1 / (1 + e^(-z))\n"
            "Costo: Cross-Entropy = -Σ[yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]"
        ),
        "code_template": '''\
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)

# 1. Separar datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state={random_state}, stratify=y
)

# 2. Crear y entrenar modelo
model = LogisticRegression(
    C={C},
    penalty="{penalty}",
    solver="{solver}",
    max_iter={max_iter},
    random_state={random_state}
)
model.fit(X_train, y_train)

# 3. Predecir y evaluar
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy : {{accuracy_score(y_test, y_pred):.4f}}")
print(f"ROC-AUC  : {{roc_auc_score(y_test, y_prob):.4f}}")
print(classification_report(y_test, y_pred))
''',
        "params": {
            "test_size": {
                "label": "Tamaño del conjunto de prueba (test_size)",
                "is_hyperparam": False,
                "type": "slider",
                "min": 0.1, "max": 0.5, "default": 0.2, "step": 0.05,
                "what_it_is": "No es un hiperparámetro del modelo. Fracción de datos reservados para test.",
                "effect": "Afecta cuántos datos tiene el modelo para aprender vs. cuántos se usan para evaluar.",
                "rule_of_thumb": "80/20 es el estándar. Usa stratify=y para mantener proporciones de clase.",
            },
            "C": {
                "label": "Inverso de regularización (C)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 0.001, "max": 100.0, "default": 1.0, "step": None,
                "log_scale": True,
                "what_it_is": (
                    "Hiperparámetro clave. C = 1/λ donde λ es la fuerza de regularización. "
                    "Controla cuánto se penalizan los coeficientes grandes."
                ),
                "effect": (
                    "C pequeño (0.001–0.1): regularización fuerte → coeficientes pequeños → modelo simple, "
                    "menos sobreajuste pero puede underfit. "
                    "C grande (10–100): regularización débil → modelo más flexible, puede sobreajustar."
                ),
                "rule_of_thumb": "Empieza con C=1.0. Usa GridSearchCV para optimizar en [0.01, 0.1, 1, 10, 100].",
            },
            "penalty": {
                "label": "Tipo de regularización (penalty)",
                "is_hyperparam": True,
                "type": "select",
                "options": ["l2", "l1", "elasticnet", "None"],
                "default": "l2",
                "what_it_is": (
                    "Hiperparámetro estructural. Define la norma usada para penalizar coeficientes grandes."
                ),
                "effect": (
                    "l2 (Ridge): reduce coeficientes sin eliminarlos. Estable, funciona bien en general. "
                    "l1 (Lasso): puede forzar coeficientes a 0 — selección automática de variables. "
                    "elasticnet: combinación de l1 y l2. "
                    "None: sin regularización (solo si tienes muchos datos y poca colinealidad)."
                ),
                "rule_of_thumb": "l2 es el default seguro. Usa l1 si sospechas que muchas variables son irrelevantes.",
            },
            "solver": {
                "label": "Algoritmo de optimización (solver)",
                "is_hyperparam": True,
                "type": "select",
                "options": ["lbfgs", "liblinear", "saga", "newton-cg", "sag"],
                "default": "lbfgs",
                "what_it_is": (
                    "Hiperparámetro algorítmico. Define el optimizador que minimiza la función de costo."
                ),
                "effect": (
                    "lbfgs: rápido, bueno para multiclase, solo soporta l2. "
                    "liblinear: bueno para datasets pequeños, soporta l1. "
                    "saga: soporta l1, l2, elasticnet — ideal para datasets grandes. "
                    "newton-cg: preciso pero lento en datasets grandes."
                ),
                "rule_of_thumb": "lbfgs para la mayoría. liblinear si usas l1 y el dataset es pequeño.",
            },
            "max_iter": {
                "label": "Máximo de iteraciones (max_iter)",
                "is_hyperparam": False,
                "type": "number",
                "min": 100, "max": 5000, "default": 1000,
                "what_it_is": (
                    "Parámetro de convergencia — no hiperparámetro real del modelo. "
                    "Limita cuántas iteraciones usa el optimizador para converger."
                ),
                "effect": (
                    "Si el modelo lanza ConvergenceWarning, aumenta este valor. "
                    "No afecta la calidad del modelo si converge; solo es un límite de tiempo."
                ),
                "rule_of_thumb": "1000 es suficiente para la mayoría. Sube a 5000 si hay warning de convergencia.",
            },
            "random_state": {
                "label": "Semilla aleatoria (random_state)",
                "is_hyperparam": False,
                "type": "number",
                "min": 0, "max": 999, "default": 42,
                "what_it_is": "Semilla para reproducibilidad del split y de ciertos solvers.",
                "effect": "Asegura que los resultados sean iguales en cada ejecución.",
                "rule_of_thumb": "Usa un entero fijo (42 es convención).",
            },
        },
        "metrics": {
            "Accuracy": "Proporción de predicciones correctas. Misleading en clases desbalanceadas.",
            "Precision": "De los que predije positivos, ¿cuántos realmente lo son? TP/(TP+FP).",
            "Recall": "De los positivos reales, ¿cuántos detecté? TP/(TP+FN). Clave en medicina/fraude.",
            "F1-Score": "Media armónica de Precision y Recall. Balance entre ambos.",
            "ROC-AUC": "Área bajo la curva ROC. 0.5 = random, 1.0 = perfecto. Independiente del umbral.",
        },
    },

    # ─────────────────────────────────────────────
    #  SVM
    # ─────────────────────────────────────────────
    "svm": {
        "name": "Support Vector Machine",
        "type": "supervisado",
        "task": "clasificación / regresión",
        "icon": "🎯",
        "one_liner": "Encuentra el hiperplano que maximiza el margen entre clases. Muy efectivo en espacios de alta dimensión.",
        "when_to_use": [
            "Dataset mediano con muchas características (texto, imágenes).",
            "Las clases no son linealmente separables (usa kernel RBF o polynomial).",
            "Necesitas robustez ante outliers (el margen solo depende de support vectors).",
            "Clasificación binaria con pocas muestras y muchas features.",
        ],
        "when_not_to_use": [
            "Dataset muy grande (>100k filas): lento en entrenamiento.",
            "Necesitas probabilidades calibradas (SVM no las genera directamente).",
            "Multiclase con muchas clases (considera árboles o redes).",
        ],
        "math_concept": (
            "Maximiza el margen 2/||w|| sujeto a:\n"
            "yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ\n"
            "Con kernel: K(xᵢ,xⱼ) = φ(xᵢ)·φ(xⱼ)\n"
            "RBF: K(x,z) = exp(−γ||x−z||²)"
        ),
        "code_template": '''\
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# 1. Separar datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state={random_state}, stratify=y
)

# 2. IMPORTANTE: SVM requiere escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 3. Crear y entrenar modelo
model = SVC(
    C={C},
    kernel="{kernel}",
    gamma="{gamma}",
    probability={probability},
    random_state={random_state}
)
model.fit(X_train, y_train)

# 4. Predecir y evaluar
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Support Vectors por clase: {{model.n_support_}}")
''',
        "params": {
            "test_size": {
                "label": "Tamaño del conjunto de prueba (test_size)",
                "is_hyperparam": False,
                "type": "slider",
                "min": 0.1, "max": 0.5, "default": 0.2, "step": 0.05,
                "what_it_is": "No es hiperparámetro. Fracción de datos para test.",
                "effect": "Más test → evaluación más confiable. Menos → más datos para entrenar.",
                "rule_of_thumb": "0.2 como estándar.",
            },
            "C": {
                "label": "Parámetro de regularización (C)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 0.001, "max": 1000.0, "default": 1.0, "step": None,
                "log_scale": True,
                "what_it_is": (
                    "Hiperparámetro clave. Controla el trade-off entre margen amplio y errores de clasificación. "
                    "C grande → penaliza mucho los errores → margen más pequeño pero menos errores en train."
                ),
                "effect": (
                    "C pequeño: margen amplio, más tolerante a errores (underfitting posible). "
                    "C grande: margen estricto, puede sobreajustar."
                ),
                "rule_of_thumb": "Prueba [0.1, 1, 10, 100] con CrossValidation.",
            },
            "kernel": {
                "label": "Función kernel (kernel)",
                "is_hyperparam": True,
                "type": "select",
                "options": ["rbf", "linear", "poly", "sigmoid"],
                "default": "rbf",
                "what_it_is": (
                    "Hiperparámetro estructural fundamental. Define cómo se mide la similitud "
                    "entre puntos en un espacio de mayor dimensión."
                ),
                "effect": (
                    "linear: hiperplano recto. Funciona si los datos son separables. "
                    "rbf: frontera circular/elíptica. El más versátil. "
                    "poly: grado N de polynomial. Captura relaciones polinómicas. "
                    "sigmoid: similar a redes neuronales."
                ),
                "rule_of_thumb": "Empieza con rbf. Usa linear si tienes muchas features y pocos datos.",
            },
            "gamma": {
                "label": "Coeficiente del kernel (gamma)",
                "is_hyperparam": True,
                "type": "select",
                "options": ["scale", "auto"],
                "default": "scale",
                "what_it_is": (
                    "Hiperparámetro del kernel (solo aplica a rbf, poly, sigmoid). "
                    "Controla el radio de influencia de cada punto de soporte."
                ),
                "effect": (
                    "gamma alto: cada punto tiene poca influencia → frontera muy irregular → sobreajuste. "
                    "gamma bajo: influencia amplia → frontera suave → posible underfitting. "
                    "'scale' = 1/(n_features · Var(X)) — buena opción por defecto."
                ),
                "rule_of_thumb": "Usa 'scale'. Ajusta numéricamente solo con GridSearchCV.",
            },
            "probability": {
                "label": "Estimar probabilidades (probability)",
                "is_hyperparam": False,
                "type": "checkbox",
                "default": True,
                "what_it_is": (
                    "Parámetro de capacidad — no hiperparámetro del modelo. "
                    "Activa la estimación de probabilidades via Platt Scaling (5-fold CV interno)."
                ),
                "effect": "Habilita predict_proba(). Hace el entrenamiento más lento. Necesario para ROC-AUC.",
                "rule_of_thumb": "True si necesitas probabilidades o métricas como AUC.",
            },
            "random_state": {
                "label": "Semilla aleatoria (random_state)",
                "is_hyperparam": False,
                "type": "number",
                "min": 0, "max": 999, "default": 42,
                "what_it_is": "Semilla para reproducibilidad.",
                "effect": "Asegura resultados iguales entre ejecuciones.",
                "rule_of_thumb": "Usa 42.",
            },
        },
        "metrics": {
            "Accuracy": "Proporción de predicciones correctas.",
            "Precision": "TP/(TP+FP). De los predichos positivos, cuántos son reales.",
            "Recall": "TP/(TP+FN). De los positivos reales, cuántos detectaste.",
            "F1-Score": "Media armónica de Precision y Recall.",
            "Support Vectors": "Número de puntos en el margen. Cuanto menos, más generalizable.",
        },
    },

    # ─────────────────────────────────────────────
    #  K-MEANS
    # ─────────────────────────────────────────────
    "kmeans": {
        "name": "K-Means Clustering",
        "type": "no_supervisado",
        "task": "clustering",
        "icon": "🔵",
        "one_liner": "Agrupa las observaciones en K clusters minimizando la distancia intra-cluster al centroide.",
        "when_to_use": [
            "No tienes etiquetas y quieres descubrir grupos naturales.",
            "Segmentación de clientes, documentos, imágenes.",
            "Los clusters tienen forma aproximadamente esférica.",
            "Conoces (o puedes estimar) el número de clusters.",
        ],
        "when_not_to_use": [
            "Los clusters tienen formas irregulares (usa DBSCAN).",
            "Los datos tienen escalas muy distintas sin normalizar (distancias sesgadas).",
            "El número de clusters es completamente desconocido (considera el Elbow Method primero).",
        ],
        "math_concept": (
            "Minimiza la inercia (suma de distancias cuadradas al centroide):\n"
            "J = Σₖ Σᵢ∈Cₖ ||xᵢ − μₖ||²\n"
            "Algoritmo EM:\n"
            "  E-step: asignar cada punto al centroide más cercano\n"
            "  M-step: recalcular centroides como media de su cluster"
        ),
        "code_template": '''\
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

# 1. IMPORTANTE: K-Means es sensible a la escala
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Crear y ajustar modelo
model = KMeans(
    n_clusters={n_clusters},
    init="{init}",
    n_init={n_init},
    max_iter={max_iter},
    random_state={random_state}
)
labels = model.fit_predict(X_scaled)

# 3. Evaluar (sin etiquetas reales)
print(f"Inercia     : {{model.inertia_:.2f}}")
print(f"Silhouette  : {{silhouette_score(X_scaled, labels):.4f}}")
print(f"Davies-Bouldin: {{davies_bouldin_score(X_scaled, labels):.4f}}")
print(f"Iteraciones : {{model.n_iter_}}")

# 4. Elbow Method para elegir K
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state={random_state}, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
''',
        "params": {
            "n_clusters": {
                "label": "Número de clusters (n_clusters = K)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 2, "max": 15, "default": 3, "step": 1,
                "what_it_is": (
                    "Hiperparámetro principal. Define cuántos grupos busca el algoritmo en los datos. "
                    "Debes elegirlo tú — el modelo no lo aprende automáticamente."
                ),
                "effect": (
                    "K pequeño: clusters grandes y generales, puede perder estructura real. "
                    "K grande: clusters más específicos, pero puede fragmentar grupos reales. "
                    "Usa el Elbow Method o el Silhouette Score para guiar esta elección."
                ),
                "rule_of_thumb": "Genera el gráfico de inercia vs K (Elbow Method) y busca el 'codo'.",
            },
            "init": {
                "label": "Método de inicialización (init)",
                "is_hyperparam": True,
                "type": "select",
                "options": ["k-means++", "random"],
                "default": "k-means++",
                "what_it_is": (
                    "Hiperparámetro algorítmico. Define cómo se ubican los centroides iniciales "
                    "antes de comenzar el algoritmo EM."
                ),
                "effect": (
                    "k-means++: inicialización inteligente — ubica centroides lejos entre sí. "
                    "Converge más rápido y evita mínimos locales malos. "
                    "random: inicialización aleatoria. Más lento y menos estable."
                ),
                "rule_of_thumb": "Siempre usa k-means++ salvo que investigues el efecto de la inicialización.",
            },
            "n_init": {
                "label": "Número de inicializaciones (n_init)",
                "is_hyperparam": False,
                "type": "slider",
                "min": 1, "max": 50, "default": 10, "step": 1,
                "what_it_is": (
                    "Parámetro de robustez — no hiperparámetro del modelo. "
                    "Ejecuta el algoritmo n_init veces con distintas semillas y devuelve el mejor resultado (menor inercia)."
                ),
                "effect": "Más repeticiones = más estabilidad, pero más tiempo de cómputo.",
                "rule_of_thumb": "10 es el default estándar. Sube a 20 si la inercia varía mucho entre ejecuciones.",
            },
            "max_iter": {
                "label": "Máximo de iteraciones EM (max_iter)",
                "is_hyperparam": False,
                "type": "number",
                "min": 50, "max": 1000, "default": 300,
                "what_it_is": "Límite de pasos E-M por inicialización. Parámetro de convergencia, no hiperparámetro.",
                "effect": "Si el modelo no converge, aumenta este valor. 300 es suficiente en la mayoría de casos.",
                "rule_of_thumb": "Déjalo en 300 a menos que veas warnings de no convergencia.",
            },
            "random_state": {
                "label": "Semilla aleatoria (random_state)",
                "is_hyperparam": False,
                "type": "number",
                "min": 0, "max": 999, "default": 42,
                "what_it_is": "Semilla para reproducibilidad de la inicialización.",
                "effect": "Con k-means++ y n_init=10, la variación es pequeña. Fija igual para reproducibilidad.",
                "rule_of_thumb": "Usa 42.",
            },
        },
        "metrics": {
            "Inercia": "Suma de distancias cuadradas al centroide más cercano. Menor = clusters más compactos.",
            "Silhouette Score": "[-1, 1]. Mide cohesión vs separación. Valores cercanos a 1 = clusters bien definidos.",
            "Davies-Bouldin": "≥ 0. Menor es mejor. Ratio entre dispersión interna y separación entre clusters.",
            "Calinski-Harabasz": "Mayor es mejor. Ratio de varianza entre-clusters vs intra-cluster.",
        },
    },

    # ─────────────────────────────────────────────
    #  PCA
    # ─────────────────────────────────────────────
    "pca": {
        "name": "PCA — Análisis de Componentes Principales",
        "type": "no_supervisado",
        "task": "reducción de dimensionalidad",
        "icon": "📐",
        "one_liner": "Reduce la dimensionalidad proyectando los datos en las direcciones de máxima varianza.",
        "when_to_use": [
            "Tienes muchas variables correlacionadas y quieres reducirlas.",
            "Visualizar datos de alta dimensión en 2D o 3D.",
            "Pre-procesamiento antes de otro modelo (reduce ruido y tiempo de cómputo).",
            "Comprimir datos manteniendo la mayor varianza posible.",
        ],
        "when_not_to_use": [
            "Necesitas interpretabilidad directa de las variables originales.",
            "La relación entre variables no es lineal (considera t-SNE o UMAP).",
            "Tienes pocas variables — PCA no aporta mucho con <5 features.",
        ],
        "math_concept": (
            "Descomposición en Valores Singulares (SVD) de la matriz centrada X:\n"
            "X = U Σ Vᵀ\n"
            "Las columnas de V son los Componentes Principales (eigenvectors).\n"
            "Varianza explicada por componente i: λᵢ / Σλⱼ\n"
            "Proyección: Z = X · V[:, :k]"
        ),
        "code_template": '''\
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. SIEMPRE escalar antes de PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Analizar varianza explicada acumulada
pca_full = PCA()
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumvar >= 0.95) + 1
print(f"Componentes para 95% varianza: {{n_components_95}}")

# 3. Aplicar PCA con los componentes elegidos
pca = PCA(
    n_components={n_components},
    svd_solver="{svd_solver}",
    random_state={random_state}
)
X_reduced = pca.fit_transform(X_scaled)

print(f"Forma original  : {{X_scaled.shape}}")
print(f"Forma reducida  : {{X_reduced.shape}}")
print(f"Varianza expl.  : {{pca.explained_variance_ratio_}}")
print(f"Varianza acumul.: {{np.sum(pca.explained_variance_ratio_):.4f}}")
''',
        "params": {
            "n_components": {
                "label": "Número de componentes (n_components)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 1, "max": 20, "default": 2, "step": 1,
                "what_it_is": (
                    "Hiperparámetro principal. Define cuántas dimensiones tiene el espacio reducido. "
                    "Cada componente es una combinación lineal de las variables originales."
                ),
                "effect": (
                    "Menos componentes: más compresión, más pérdida de información. "
                    "Más componentes: más información preservada, menos reducción. "
                    "Puedes pasar un float (ej: 0.95) para retener el 95% de la varianza automáticamente."
                ),
                "rule_of_thumb": "Elige el mínimo de componentes que explique ≥95% de la varianza. Usa el scree plot.",
            },
            "svd_solver": {
                "label": "Algoritmo SVD (svd_solver)",
                "is_hyperparam": True,
                "type": "select",
                "options": ["auto", "full", "randomized", "arpack"],
                "default": "auto",
                "what_it_is": (
                    "Hiperparámetro algorítmico. Define el método numérico para calcular la SVD."
                ),
                "effect": (
                    "auto: selecciona automáticamente según el tamaño de los datos. "
                    "full: SVD exacta (más lento pero preciso). "
                    "randomized: aproximación rápida para datasets grandes (>500 filas, >10 columnas). "
                    "arpack: bueno para matrices dispersas."
                ),
                "rule_of_thumb": "Deja en 'auto'. Usa 'randomized' si el dataset es grande (>10k filas).",
            },
            "random_state": {
                "label": "Semilla aleatoria (random_state)",
                "is_hyperparam": False,
                "type": "number",
                "min": 0, "max": 999, "default": 42,
                "what_it_is": "Semilla para reproducibilidad (aplica solo con svd_solver='randomized').",
                "effect": "Asegura resultados iguales. Irrelevante con 'full' SVD.",
                "rule_of_thumb": "Fija en 42 por convención.",
            },
        },
        "metrics": {
            "Varianza explicada por comp.": "Fracción de la varianza total capturada por cada componente. Suma = 1 si usas todos.",
            "Varianza explicada acumulada": "Suma acumulada. 0.95 significa que retienes el 95% de la información.",
            "Ratio de reconstrucción": "Qué tan bien se puede reconstruir X original desde el espacio reducido.",
        },
    },

    # ─────────────────────────────────────────────
    #  ÁRBOL DE DECISIÓN
    # ─────────────────────────────────────────────
    "decision_tree": {
        "name": "Árbol de Decisión",
        "type": "supervisado",
        "task": "clasificación / regresión",
        "icon": "🌳",
        "one_liner": "Divide los datos en reglas if-else aprendidas automáticamente. El modelo más interpretable y visual.",
        "when_to_use": [
            "Necesitas explicar exactamente cómo el modelo toma decisiones.",
            "El dataset tiene variables mixtas (numéricas y categóricas).",
            "Quieres reglas de negocio claras (ej: 'si edad > 30 y salario > 50k → riesgo bajo').",
            "Tienes datos no lineales y quieres un modelo simple.",
        ],
        "when_not_to_use": [
            "El árbol crece demasiado (sobreajuste) — usa Random Forest en su lugar.",
            "Necesitas predicciones muy precisas — los árboles solos son menos estables.",
            "Los datos tienen muchas variables numéricas continuas con relaciones suaves.",
        ],
        "math_concept": (
            "Cada nodo divide los datos maximizando la ganancia de información:\n"
            "Ganancia = Impureza(padre) − Σ [|hijo|/|padre| × Impureza(hijo)]\n\n"
            "Impureza de Gini (clasificación):\n"
            "G = 1 − Σ pᵢ²\n\n"
            "Varianza (regresión):\n"
            "V = (1/n) Σ(yᵢ − ȳ)²"
        ),
        "code_template": '''\
from sklearn.tree import DecisionTreeClassifier  # o DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Separar datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state={random_state}
)

# 2. Crear y entrenar modelo
model = DecisionTreeClassifier(
    max_depth={max_depth},
    min_samples_split={min_samples_split},
    min_samples_leaf={min_samples_leaf},
    criterion="{criterion}",
    random_state={random_state}
)
model.fit(X_train, y_train)

# 3. Predecir y evaluar
y_pred = model.predict(X_test)
print(f"Accuracy      : {{accuracy_score(y_test, y_pred):.4f}}")
print(f"Profundidad   : {{model.get_depth()}}")
print(f"Nodos hoja    : {{model.get_n_leaves()}}")
print(classification_report(y_test, y_pred))
''',
        "params": {
            "test_size": {
                "label": "Tamaño del conjunto de prueba (test_size)",
                "is_hyperparam": False,
                "type": "slider",
                "min": 0.1, "max": 0.5, "default": 0.2, "step": 0.05,
                "what_it_is": "Fracción de datos reservada para evaluar el modelo.",
                "effect": "Más test = evaluación más confiable pero menos datos para entrenar.",
                "rule_of_thumb": "Usa 0.2 (80/20) como estándar.",
            },
            "max_depth": {
                "label": "Profundidad máxima del árbol (max_depth)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 1, "max": 20, "default": 5, "step": 1,
                "what_it_is": (
                    "Hiperparámetro clave. Limita cuántos niveles de divisiones puede tener el árbol. "
                    "None = árbol crece hasta que todas las hojas sean puras."
                ),
                "effect": (
                    "Profundidad baja (2–4): árbol simple, interpretable, puede underfit. "
                    "Profundidad alta (>10): árbol complejo, memoriza el training, sobreajusta. "
                    "Este es el hiperparámetro más importante para controlar el overfitting."
                ),
                "rule_of_thumb": "Empieza con 3–5. Aumenta gradualmente validando con CV.",
            },
            "min_samples_split": {
                "label": "Mínimo de muestras para dividir (min_samples_split)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 2, "max": 50, "default": 2, "step": 1,
                "what_it_is": (
                    "Hiperparámetro de regularización. Número mínimo de muestras que debe tener "
                    "un nodo para intentar dividirlo."
                ),
                "effect": (
                    "Valores bajos (2): el árbol divide casi siempre → más complejo → puede sobreajustar. "
                    "Valores altos (10–20): menos divisiones → árbol más simple → más generalizable."
                ),
                "rule_of_thumb": "Sube a 10–20 si el árbol sobreajusta.",
            },
            "min_samples_leaf": {
                "label": "Mínimo de muestras en hoja (min_samples_leaf)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 1, "max": 30, "default": 1, "step": 1,
                "what_it_is": (
                    "Hiperparámetro de regularización. Número mínimo de muestras que debe tener "
                    "cada hoja del árbol."
                ),
                "effect": (
                    "Leaf grande: hojas más 'gruesas', predicciones más suaves, menos sobreajuste. "
                    "Leaf=1: puede crear hojas con una sola muestra (memorizacion perfecta del train)."
                ),
                "rule_of_thumb": "Usa 5–10 para datos ruidosos. 1 solo si el dataset es muy limpio.",
            },
            "criterion": {
                "label": "Criterio de división (criterion)",
                "is_hyperparam": True,
                "type": "select",
                "options": ["gini", "entropy", "log_loss"],
                "default": "gini",
                "what_it_is": (
                    "Hiperparámetro que define la función de impureza usada para evaluar qué tan "
                    "buena es una división."
                ),
                "effect": (
                    "gini: índice de Gini — rápido, por defecto, funciona bien en la mayoría de casos. "
                    "entropy: ganancia de información — más costoso computacionalmente pero a veces genera "
                    "árboles más equilibrados. "
                    "En la práctica, la diferencia es mínima."
                ),
                "rule_of_thumb": "Usa gini por defecto. Prueba entropy si los resultados son pobres.",
            },
            "random_state": {
                "label": "Semilla aleatoria (random_state)",
                "is_hyperparam": False,
                "type": "number",
                "min": 0, "max": 999, "default": 42,
                "what_it_is": "Semilla para reproducibilidad del split y del árbol.",
                "effect": "Garantiza los mismos resultados en cada ejecución.",
                "rule_of_thumb": "Usa 42 por convención.",
            },
        },
        "metrics": {
            "Accuracy": "Proporción de predicciones correctas. Cuidado con clases desbalanceadas.",
            "F1-Score": "Balance entre precisión y recall. Mejor métrica si las clases están desbalanceadas.",
            "Profundidad": "Niveles del árbol. Más profundidad = modelo más complejo.",
            "Nodos hoja": "Número de nodos terminales. Más hojas = más reglas = más complejidad.",
        },
    },

    # ─────────────────────────────────────────────
    #  RANDOM FOREST
    # ─────────────────────────────────────────────
    "random_forest": {
        "name": "Random Forest",
        "type": "supervisado",
        "task": "clasificación / regresión",
        "icon": "🌲",
        "one_liner": "Conjunto de cientos de árboles de decisión que votan. Robusto, preciso y resistente al sobreajuste.",
        "when_to_use": [
            "Quieres alta precisión sin mucho ajuste manual de hiperparámetros.",
            "El dataset tiene valores faltantes o variables mixtas.",
            "Necesitas importancia de variables (feature importance).",
            "Árbol de decisión solo sobreajusta — Random Forest es el siguiente paso natural.",
        ],
        "when_not_to_use": [
            "Necesitas un modelo interpretable (Random Forest es una caja negra).",
            "El dataset es muy grande (>1M filas) y el tiempo de entrenamiento es crítico.",
            "Tienes pocos datos (<100 filas) — puede ser excesivo.",
        ],
        "math_concept": (
            "Bagging (Bootstrap Aggregation) + selección aleatoria de variables:\n\n"
            "Para cada árbol t en {1..T}:\n"
            "  1. Muestrea n puntos CON reemplazo (bootstrap sample)\n"
            "  2. En cada nodo, considera solo m = sqrt(p) variables aleatorias\n"
            "  3. Entrena un árbol profundo sin podar\n\n"
            "Predicción final:\n"
            "  Clasificación: voto mayoritario\n"
            "  Regresión: promedio de predicciones"
        ),
        "code_template": '''\
from sklearn.ensemble import RandomForestClassifier  # o RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# 1. Separar datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state={random_state}
)

# 2. Crear y entrenar modelo
model = RandomForestClassifier(
    n_estimators={n_estimators},
    max_depth={max_depth},
    min_samples_split={min_samples_split},
    max_features="{max_features}",
    random_state={random_state},
    n_jobs=-1
)
model.fit(X_train, y_train)

# 3. Predecir y evaluar
y_pred = model.predict(X_test)
print(f"Accuracy : {{accuracy_score(y_test, y_pred):.4f}}")
print(classification_report(y_test, y_pred))

# 4. Importancia de variables
importances = pd.Series(model.feature_importances_).sort_values(ascending=False)
print("Top features:", importances.head(5))
''',
        "params": {
            "test_size": {
                "label": "Tamaño del conjunto de prueba (test_size)",
                "is_hyperparam": False,
                "type": "slider",
                "min": 0.1, "max": 0.5, "default": 0.2, "step": 0.05,
                "what_it_is": "Fracción de datos reservada para evaluar el modelo.",
                "effect": "Más test = evaluación más confiable pero menos datos para entrenar.",
                "rule_of_thumb": "Usa 0.2 (80/20) como estándar.",
            },
            "n_estimators": {
                "label": "Número de árboles (n_estimators)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 10, "max": 500, "default": 100, "step": 10,
                "what_it_is": (
                    "Hiperparámetro principal. Cuántos árboles de decisión independientes "
                    "forman el bosque. Cada árbol vota y la mayoría gana."
                ),
                "effect": (
                    "Más árboles = modelo más estable y preciso, pero más lento. "
                    "El error disminuye con más árboles hasta un punto de saturación. "
                    "Después de ~200 árboles, la mejora es marginal."
                ),
                "rule_of_thumb": "100 es un buen inicio. Sube a 200–300 para más precisión.",
            },
            "max_depth": {
                "label": "Profundidad máxima de cada árbol (max_depth)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 1, "max": 30, "default": 10, "step": 1,
                "what_it_is": (
                    "Hiperparámetro de cada árbol individual. A diferencia del árbol solo, "
                    "Random Forest tolera árboles más profundos porque el ensemble promedia el ruido."
                ),
                "effect": (
                    "None (sin límite): cada árbol crece hasta pureza — genera varianza alta en cada árbol "
                    "pero el promedio del ensemble es estable. "
                    "Valores medios (10–15): balance entre profundidad y velocidad."
                ),
                "rule_of_thumb": "None o 10–15. El ensemble corrige el sobreajuste individual.",
            },
            "min_samples_split": {
                "label": "Mínimo de muestras para dividir (min_samples_split)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 2, "max": 20, "default": 2, "step": 1,
                "what_it_is": "Número mínimo de muestras para que un nodo pueda dividirse.",
                "effect": "Valores más altos evitan divisiones con pocos datos — reduce varianza.",
                "rule_of_thumb": "2 funciona bien con el ensemble. Sube a 5–10 si hay ruido.",
            },
            "max_features": {
                "label": "Variables candidatas por nodo (max_features)",
                "is_hyperparam": True,
                "type": "select",
                "options": ["sqrt", "log2", "None"],
                "default": "sqrt",
                "what_it_is": (
                    "Hiperparámetro clave de aleatoriedad. En cada nodo, el árbol solo "
                    "considera este subconjunto de variables — esto diversifica los árboles."
                ),
                "effect": (
                    "sqrt (recomendado para clasificación): raíz cuadrada del total de features. "
                    "log2: logaritmo base 2 — menos variables, más diversidad. "
                    "None: usa todas las variables (menos aleatorio, árboles más correlacionados)."
                ),
                "rule_of_thumb": "sqrt para clasificación, 1/3 del total para regresión.",
            },
            "random_state": {
                "label": "Semilla aleatoria (random_state)",
                "is_hyperparam": False,
                "type": "number",
                "min": 0, "max": 999, "default": 42,
                "what_it_is": "Semilla para reproducibilidad del bootstrap y la aleatoriedad.",
                "effect": "Garantiza los mismos resultados en cada ejecución.",
                "rule_of_thumb": "Usa 42 por convención.",
            },
        },
        "metrics": {
            "Accuracy": "Proporción de predicciones correctas sobre el conjunto de test.",
            "F1-Score": "Media armónica de Precision y Recall. Robusto ante clases desbalanceadas.",
            "ROC-AUC": "Área bajo la curva ROC. Mide la capacidad discriminativa del modelo.",
            "OOB Score": "Out-of-Bag score: validación interna usando las muestras no usadas en bootstrap.",
        },
    },

    # ─────────────────────────────────────────────
    #  KNN
    # ─────────────────────────────────────────────
    "knn": {
        "name": "K-Nearest Neighbors",
        "type": "supervisado",
        "task": "clasificación / regresión",
        "icon": "📍",
        "one_liner": "Clasifica o predice basándose en los K vecinos más cercanos. Sin entrenamiento explícito.",
        "when_to_use": [
            "El dataset es pequeño-mediano y la distancia entre puntos es significativa.",
            "Quieres un modelo sin suposiciones sobre la distribución de los datos.",
            "Las fronteras de decisión son complejas y no lineales.",
            "Necesitas un baseline rápido de implementar.",
        ],
        "when_not_to_use": [
            "El dataset es grande (>100k filas): la predicción es lenta (busca vecinos en todo el set).",
            "Hay muchas dimensiones (maldición de la dimensionalidad).",
            "Las features no están escaladas — KNN es muy sensible a escalas distintas.",
            "Necesitas un modelo interpretable con reglas explícitas.",
        ],
        "math_concept": (
            "Para predecir un punto nuevo x:\n\n"
            "1. Calcular distancia a todos los puntos de entrenamiento:\n"
            "   Euclidiana: d(x, xᵢ) = √Σ(xⱼ − xᵢⱼ)²\n\n"
            "2. Seleccionar los K puntos más cercanos\n\n"
            "3. Clasificación: clase mayoritaria entre los K vecinos\n"
            "   ŷ = moda({y₁, y₂, …, yₖ})\n\n"
            "4. Regresión: promedio de los K vecinos\n"
            "   ŷ = (1/K) Σ yᵢ"
        ),
        "code_template": '''\
from sklearn.neighbors import KNeighborsClassifier  # o KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# IMPORTANTE: siempre escalar antes de KNN
scaler = StandardScaler()

# 1. Separar datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state={random_state}
)
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 2. Crear y entrenar modelo
model = KNeighborsClassifier(
    n_neighbors={n_neighbors},
    weights="{weights}",
    metric="{metric}",
    p={p}
)
model.fit(X_train_s, y_train)

# 3. Predecir y evaluar
y_pred = model.predict(X_test_s)
print(f"Accuracy : {{accuracy_score(y_test, y_pred):.4f}}")
print(classification_report(y_test, y_pred))
''',
        "params": {
            "test_size": {
                "label": "Tamaño del conjunto de prueba (test_size)",
                "is_hyperparam": False,
                "type": "slider",
                "min": 0.1, "max": 0.5, "default": 0.2, "step": 0.05,
                "what_it_is": "Fracción de datos reservada para evaluar el modelo.",
                "effect": "Más test = evaluación más confiable pero menos datos para entrenar.",
                "rule_of_thumb": "Usa 0.2 (80/20) como estándar.",
            },
            "n_neighbors": {
                "label": "Número de vecinos (K)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 1, "max": 30, "default": 5, "step": 1,
                "what_it_is": (
                    "Hiperparámetro principal. Cuántos vecinos más cercanos considera el modelo "
                    "para hacer su predicción. Es la decisión más importante en KNN."
                ),
                "effect": (
                    "K=1: cada punto es su propio vecino — sobreajuste severo, frontera irregular. "
                    "K grande: frontera más suave, más estable pero puede perder detalles locales. "
                    "K muy grande: el modelo se vuelve demasiado simple (underfitting)."
                ),
                "rule_of_thumb": "Prueba K=5 como inicio. Usa CV con K impar para evitar empates en clasificación binaria.",
            },
            "weights": {
                "label": "Ponderación de vecinos (weights)",
                "is_hyperparam": True,
                "type": "select",
                "options": ["uniform", "distance"],
                "default": "uniform",
                "what_it_is": (
                    "Hiperparámetro de votación. Define si todos los vecinos pesan igual "
                    "o si los más cercanos pesan más."
                ),
                "effect": (
                    "uniform: cada vecino vota con el mismo peso. Simple y robusto. "
                    "distance: los vecinos más cercanos pesan más (inversamente proporcional a la distancia). "
                    "Útil cuando los datos más cercanos son más relevantes."
                ),
                "rule_of_thumb": "uniform por defecto. Prueba distance si la densidad varía mucho.",
            },
            "metric": {
                "label": "Métrica de distancia (metric)",
                "is_hyperparam": True,
                "type": "select",
                "options": ["euclidean", "manhattan", "chebyshev", "minkowski"],
                "default": "euclidean",
                "what_it_is": (
                    "Hiperparámetro que define cómo se mide la 'cercanía' entre puntos."
                ),
                "effect": (
                    "euclidean: distancia en línea recta — estándar para datos continuos. "
                    "manhattan: suma de diferencias absolutas — más robusto a outliers. "
                    "chebyshev: máxima diferencia en cualquier dimensión. "
                    "minkowski: generalización que incluye euclidean (p=2) y manhattan (p=1)."
                ),
                "rule_of_thumb": "Euclidean para datos continuos escalados. Manhattan si hay outliers.",
            },
            "p": {
                "label": "Parámetro p (Minkowski)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 1, "max": 5, "default": 2, "step": 1,
                "what_it_is": (
                    "Solo aplica si metric='minkowski'. p=1 → Manhattan, p=2 → Euclidean."
                ),
                "effect": "Controla la forma de la bola de distancia. p=2 es el estándar.",
                "rule_of_thumb": "Deja en 2 (Euclidean) a menos que uses minkowski explícitamente.",
            },
            "random_state": {
                "label": "Semilla aleatoria (random_state)",
                "is_hyperparam": False,
                "type": "number",
                "min": 0, "max": 999, "default": 42,
                "what_it_is": "Semilla para el split. KNN no tiene aleatoriedad interna.",
                "effect": "Garantiza el mismo train/test split en cada ejecución.",
                "rule_of_thumb": "Usa 42 por convención.",
            },
        },
        "metrics": {
            "Accuracy": "Proporción de predicciones correctas sobre el conjunto de test.",
            "F1-Score": "Balance entre precisión y recall. Robusto ante clases desbalanceadas.",
            "K óptimo": "El K que maximiza la métrica de validación cruzada.",
        },
    },

    # ─────────────────────────────────────────────
    #  GRADIENT BOOSTING
    # ─────────────────────────────────────────────
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "type": "supervisado",
        "task": "clasificación / regresión",
        "icon": "⚡",
        "one_liner": "Construye árboles secuencialmente, cada uno corrigiendo los errores del anterior. Alta precisión.",
        "when_to_use": [
            "Quieres el mejor rendimiento predictivo en datos tabulares.",
            "El dataset es mediano (cientos a decenas de miles de filas).",
            "Tienes variables mixtas (numéricas, categóricas) y valores faltantes.",
            "Compites en Kaggle o necesitas el mejor modelo posible.",
        ],
        "when_not_to_use": [
            "El dataset es muy pequeño (<50 filas) — puede sobreajustar fácilmente.",
            "Necesitas un modelo interpretable (Gradient Boosting es una caja negra).",
            "El tiempo de entrenamiento es crítico — es más lento que Random Forest.",
        ],
        "math_concept": (
            "Boosting secuencial: cada árbol predice los residuos del anterior:\n\n"
            "F₀(x) = constante (media de y)\n"
            "Para m = 1, 2, …, M:\n"
            "  rᵢₘ = −∂L(yᵢ, Fₘ₋₁(xᵢ))/∂Fₘ₋₁(xᵢ)  (pseudo-residuos)\n"
            "  hₘ = árbol ajustado sobre rᵢₘ\n"
            "  Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)\n\n"
            "η = learning_rate controla el tamaño de cada paso.\n"
            "Costo: L = Σ L(yᵢ, F(xᵢ)) (deviance, MSE, etc.)"
        ),
        "code_template": '''\
from sklearn.ensemble import GradientBoostingClassifier  # o GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Separar datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state={random_state}
)

# 2. Crear y entrenar modelo
model = GradientBoostingClassifier(
    n_estimators={n_estimators},
    learning_rate={learning_rate},
    max_depth={max_depth},
    subsample={subsample},
    random_state={random_state}
)
model.fit(X_train, y_train)

# 3. Predecir y evaluar
y_pred = model.predict(X_test)
print(f"Accuracy : {{accuracy_score(y_test, y_pred):.4f}}")
print(classification_report(y_test, y_pred))
''',
        "params": {
            "test_size": {
                "label": "Tamaño del conjunto de prueba (test_size)",
                "is_hyperparam": False,
                "type": "slider",
                "min": 0.1, "max": 0.5, "default": 0.2, "step": 0.05,
                "what_it_is": "Fracción de datos reservada para evaluar el modelo.",
                "effect": "Más test = evaluación más confiable pero menos datos para entrenar.",
                "rule_of_thumb": "Usa 0.2 (80/20) como estándar.",
            },
            "n_estimators": {
                "label": "Número de árboles (n_estimators)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 10, "max": 500, "default": 100, "step": 10,
                "what_it_is": (
                    "Hiperparámetro principal. Cuántos árboles se construyen secuencialmente. "
                    "A diferencia de Random Forest, más árboles puede causar sobreajuste si "
                    "el learning_rate es alto."
                ),
                "effect": (
                    "Más árboles = modelo más expresivo. Pero combinado con learning_rate alto, "
                    "puede sobreajustar. Siempre ajusta n_estimators junto con learning_rate."
                ),
                "rule_of_thumb": "100 con lr=0.1 o 500 con lr=0.01. Siempre par (n_estimators, learning_rate).",
            },
            "learning_rate": {
                "label": "Tasa de aprendizaje (learning_rate)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01,
                "what_it_is": (
                    "Hiperparámetro crítico. Controla cuánto contribuye cada árbol a la predicción final. "
                    "Escala la corrección de residuos en cada iteración."
                ),
                "effect": (
                    "lr alto (0.5–1.0): converge rápido pero puede sobreajustar, necesita menos árboles. "
                    "lr bajo (0.01–0.05): converge lento pero suele generalizar mejor, necesita más árboles. "
                    "Hay un trade-off directo: lr_bajo + n_estimadores_alto = mejor rendimiento."
                ),
                "rule_of_thumb": "0.1 con 100 árboles. Para máxima precisión: 0.01 con 1000 árboles.",
            },
            "max_depth": {
                "label": "Profundidad de cada árbol (max_depth)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 1, "max": 10, "default": 3, "step": 1,
                "what_it_is": (
                    "Hiperparámetro de cada árbol base. En Gradient Boosting, los árboles suelen "
                    "ser pequeños (stumps o profundidad 3–5) — se complementan entre sí."
                ),
                "effect": (
                    "max_depth=1 (stumps): modelo muy regularizado, aprende lentamente. "
                    "max_depth=3–5: captura interacciones de 2–3 variables — estándar. "
                    "max_depth>6: riesgo de sobreajuste individual de cada árbol."
                ),
                "rule_of_thumb": "3 es el estándar para Gradient Boosting. Rara vez necesitas más de 5.",
            },
            "subsample": {
                "label": "Fracción de muestras por árbol (subsample)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 0.3, "max": 1.0, "default": 1.0, "step": 0.05,
                "what_it_is": (
                    "Hiperparámetro de regularización estocástica. Fracción de muestras usadas "
                    "para entrenar cada árbol (sin reemplazo, como en Stochastic Gradient Boosting)."
                ),
                "effect": (
                    "1.0: usa todas las muestras (determinista). "
                    "0.5–0.8: introduce aleatoriedad → reduce varianza y mejora generalización. "
                    "Análogo al dropout en redes neuronales."
                ),
                "rule_of_thumb": "0.8 suele mejorar el resultado. Prueba 0.5–0.8 si hay sobreajuste.",
            },
            "random_state": {
                "label": "Semilla aleatoria (random_state)",
                "is_hyperparam": False,
                "type": "number",
                "min": 0, "max": 999, "default": 42,
                "what_it_is": "Semilla para reproducibilidad.",
                "effect": "Garantiza los mismos resultados en cada ejecución.",
                "rule_of_thumb": "Usa 42 por convención.",
            },
        },
        "metrics": {
            "Accuracy": "Proporción de predicciones correctas sobre el conjunto de test.",
            "F1-Score": "Balance entre precisión y recall. Muy usado en clasificación desbalanceada.",
            "ROC-AUC": "Área bajo la curva ROC. Mide la capacidad discriminativa.",
            "Log-Loss": "Función de costo. Menor es mejor. Penaliza predicciones con alta confianza incorrecta.",
        },
    },

    # ─────────────────────────────────────────────
    #  NAIVE BAYES
    # ─────────────────────────────────────────────
    "naive_bayes": {
        "name": "Naive Bayes",
        "type": "supervisado",
        "task": "clasificación",
        "icon": "📊",
        "one_liner": "Clasificador probabilístico basado en el teorema de Bayes. Rapidísimo y sorprendentemente efectivo.",
        "when_to_use": [
            "Clasificación de texto (spam, sentimientos, categorías).",
            "Dataset muy grande donde la velocidad importa.",
            "Quieres un baseline probabilístico rápido.",
            "Las variables son razonablemente independientes entre sí.",
        ],
        "when_not_to_use": [
            "Las variables están fuertemente correlacionadas (viola el supuesto 'naive').",
            "Necesitas alta precisión en datos numéricos continuos complejos.",
            "El problema requiere capturar interacciones entre variables.",
        ],
        "math_concept": (
            "Teorema de Bayes:\n"
            "P(y|X) = P(X|y) · P(y) / P(X)\n\n"
            "Supuesto 'naive' (independencia condicional):\n"
            "P(X|y) = P(x₁|y) · P(x₂|y) · … · P(xₙ|y)\n\n"
            "Predicción: ŷ = argmax_y P(y) · Π P(xᵢ|y)\n\n"
            "Variantes según distribución de P(xᵢ|y):\n"
            "• GaussianNB: xᵢ continua ~ N(μ, σ²)\n"
            "• MultinomialNB: conteos (texto)\n"
            "• BernoulliNB: binaria (presencia/ausencia)"
        ),
        "code_template": '''\
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Separar datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state={random_state}
)

# 2. Crear y entrenar modelo
model = GaussianNB(var_smoothing={var_smoothing})
model.fit(X_train, y_train)

# 3. Predecir y evaluar
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
print(f"Accuracy : {{accuracy_score(y_test, y_pred):.4f}}")
print(classification_report(y_test, y_pred))
''',
        "params": {
            "test_size": {
                "label": "Tamaño del conjunto de prueba (test_size)",
                "is_hyperparam": False,
                "type": "slider",
                "min": 0.1, "max": 0.5, "default": 0.2, "step": 0.05,
                "what_it_is": "Fracción de datos reservada para evaluar el modelo.",
                "effect": "Más test = evaluación más confiable pero menos datos para entrenar.",
                "rule_of_thumb": "Usa 0.2 (80/20) como estándar.",
            },
            "var_smoothing": {
                "label": "Suavizado de varianza (var_smoothing)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 1e-12, "max": 1e-3, "default": 1e-9, "step": None,
                "log_scale": True,
                "what_it_is": (
                    "Hiperparámetro de estabilidad numérica. Añade una pequeña fracción de la "
                    "varianza máxima observada a todas las varianzas estimadas. "
                    "Evita divisiones por cero cuando una variable tiene varianza muy pequeña."
                ),
                "effect": (
                    "Valor muy pequeño (1e-12): sin suavizado, puede ser inestable con datos constantes. "
                    "Valor más alto (1e-6 a 1e-3): más regularización, suaviza las estimaciones. "
                    "En la práctica, el impacto en precisión es mínimo para datos bien preparados."
                ),
                "rule_of_thumb": "Deja el valor por defecto (1e-9). Solo ajusta si hay errores numéricos.",
            },
            "random_state": {
                "label": "Semilla aleatoria (random_state)",
                "is_hyperparam": False,
                "type": "number",
                "min": 0, "max": 999, "default": 42,
                "what_it_is": "Semilla para el split. Naive Bayes no tiene aleatoriedad interna.",
                "effect": "Garantiza el mismo train/test split en cada ejecución.",
                "rule_of_thumb": "Usa 42 por convención.",
            },
        },
        "metrics": {
            "Accuracy": "Proporción de predicciones correctas.",
            "F1-Score": "Balance entre precisión y recall.",
            "Log-Loss": "Pérdida logarítmica. Penaliza predicciones con alta confianza incorrecta.",
            "ROC-AUC": "Capacidad discriminativa del clasificador.",
        },
    },

    # ─────────────────────────────────────────────
    #  DBSCAN
    # ─────────────────────────────────────────────
    "dbscan": {
        "name": "DBSCAN",
        "type": "no_supervisado",
        "task": "clustering",
        "icon": "🔍",
        "one_liner": "Detecta clusters de cualquier forma y marca automáticamente los outliers. No necesitas definir K.",
        "when_to_use": [
            "No sabes cuántos clusters existen en los datos.",
            "Los clusters tienen formas irregulares (no esféricas).",
            "Quieres detectar anomalías u outliers automáticamente.",
            "Los datos tienen zonas densas separadas por zonas vacías.",
        ],
        "when_not_to_use": [
            "Los clusters tienen densidades muy distintas (DBSCAN usa una densidad uniforme).",
            "El dataset tiene muchas dimensiones (la densidad pierde sentido en alta dimensión).",
            "Necesitas asignar un cluster a cada punto (DBSCAN puede dejar puntos sin cluster).",
        ],
        "math_concept": (
            "Conceptos clave:\n"
            "• eps (ε): radio de vecindad — distancia máxima entre puntos 'vecinos'\n"
            "• min_samples: mínimo de vecinos para ser 'punto núcleo'\n\n"
            "Tipos de puntos:\n"
            "• Núcleo: tiene ≥ min_samples vecinos en radio eps\n"
            "• Borde: es vecino de un núcleo pero no tiene suficientes vecinos propios\n"
            "• Outlier (ruido): no es núcleo ni borde → etiqueta = -1\n\n"
            "Algoritmo:\n"
            "1. Para cada punto no visitado, encontrar vecinos en eps\n"
            "2. Si |vecinos| ≥ min_samples → nuevo cluster, expandir\n"
            "3. Si no → marcar como ruido (provisionalmente)"
        ),
        "code_template": '''\
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

# IMPORTANTE: escalar siempre antes de DBSCAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear y ajustar modelo
model = DBSCAN(eps={eps}, min_samples={min_samples}, metric="{metric}")
labels = model.fit_predict(X_scaled)

# Analizar resultados
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers  = np.sum(labels == -1)
print(f"Clusters encontrados : {{n_clusters}}")
print(f"Outliers detectados  : {{n_outliers}}")
if n_clusters > 1:
    mask = labels != -1
    sil = silhouette_score(X_scaled[mask], labels[mask])
    print(f"Silhouette (sin outliers): {{sil:.4f}}")
''',
        "params": {
            "eps": {
                "label": "Radio de vecindad (eps)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 0.1, "max": 5.0, "default": 0.5, "step": 0.05,
                "what_it_is": (
                    "Hiperparámetro principal. Define el radio máximo dentro del cual "
                    "dos puntos son considerados 'vecinos'. Es la distancia de vecindad en el "
                    "espacio de features escaladas."
                ),
                "effect": (
                    "eps pequeño: pocos puntos son vecinos → más clusters pequeños, más outliers. "
                    "eps grande: muchos puntos son vecinos → clusters grandes que se fusionan, menos outliers. "
                    "Este es el parámetro más sensible y difícil de calibrar."
                ),
                "rule_of_thumb": "Usa la curva k-distancias: ordena distancias al K-ésimo vecino y busca el 'codo'.",
            },
            "min_samples": {
                "label": "Muestras mínimas para núcleo (min_samples)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 2, "max": 30, "default": 5, "step": 1,
                "what_it_is": (
                    "Hiperparámetro de densidad. Número mínimo de puntos (incluyendo el punto mismo) "
                    "que deben existir dentro de eps para que un punto sea considerado 'núcleo'."
                ),
                "effect": (
                    "min_samples bajo (2–3): muchos puntos son núcleo → clusters grandes, pocos outliers. "
                    "min_samples alto (10–20): solo zonas muy densas forman clusters → más outliers detectados. "
                    "Controla la sensibilidad a ruido."
                ),
                "rule_of_thumb": "Regla general: min_samples = 2 × dimensiones del dataset. Mínimo 5.",
            },
            "metric": {
                "label": "Métrica de distancia (metric)",
                "is_hyperparam": True,
                "type": "select",
                "options": ["euclidean", "manhattan", "chebyshev", "cosine"],
                "default": "euclidean",
                "what_it_is": (
                    "Define cómo se mide la distancia entre puntos para determinar vecindades."
                ),
                "effect": (
                    "euclidean: estándar para datos numéricos continuos. "
                    "manhattan: más robusto a outliers dimensionales. "
                    "cosine: útil para datos de texto o vectores de alta dimensión."
                ),
                "rule_of_thumb": "Euclidean para la mayoría de casos numéricos escalados.",
            },
        },
        "metrics": {
            "Clusters": "Número de clusters encontrados (sin contar outliers).",
            "Outliers": "Puntos marcados como ruido (etiqueta -1). Un proxy de anomalías.",
            "Silhouette": "Cohesión y separación de clusters. Solo se calcula sobre puntos no-outlier.",
        },
    },

    # ─────────────────────────────────────────────
    #  HIERARCHICAL CLUSTERING
    # ─────────────────────────────────────────────
    "hierarchical": {
        "name": "Clustering Jerárquico",
        "type": "no_supervisado",
        "task": "clustering",
        "icon": "🌿",
        "one_liner": "Construye un árbol de clusters (dendrograma) de abajo hacia arriba. Revela la estructura anidada de los datos.",
        "when_to_use": [
            "Quieres explorar la jerarquía natural de agrupaciones en los datos.",
            "No sabes cuántos clusters hay y quieres decidirlo visualmente con el dendrograma.",
            "Los clusters tienen formas no esféricas o diferentes tamaños.",
            "El dataset es pequeño-mediano (<5000 filas) y necesitas interpretabilidad.",
        ],
        "when_not_to_use": [
            "El dataset es grande (>10k filas): la complejidad O(n²) lo hace lento.",
            "Necesitas reasignar puntos a clusters nuevos (el dendrograma es estático).",
            "Los datos tienen mucho ruido — el agglomerative es sensible a outliers.",
        ],
        "math_concept": (
            "Algoritmo Agglomerative (bottom-up):\n"
            "1. Cada punto empieza como su propio cluster\n"
            "2. Calcula matriz de distancias entre todos los clusters\n"
            "3. Une los dos clusters más cercanos\n"
            "4. Repite hasta que quede un solo cluster\n\n"
            "Linkage (criterio de enlace):\n"
            "• ward: minimiza la varianza intra-cluster (más compactos)\n"
            "• complete: distancia máxima entre clusters\n"
            "• average: distancia promedio entre clusters\n"
            "• single: distancia mínima entre clusters\n\n"
            "El dendrograma muestra el historial completo de fusiones."
        ),
        "code_template": '''\
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

# IMPORTANTE: escalar antes del clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear y ajustar modelo
model = AgglomerativeClustering(
    n_clusters={n_clusters},
    linkage="{linkage}",
    metric="{affinity}"
)
labels = model.fit_predict(X_scaled)

# Evaluar
sil = silhouette_score(X_scaled, labels)
print(f"Clusters        : {{{n_clusters}}}")
print(f"Silhouette      : {{sil:.4f}}")
''',
        "params": {
            "n_clusters": {
                "label": "Número de clusters (n_clusters)",
                "is_hyperparam": True,
                "type": "slider",
                "min": 2, "max": 15, "default": 3, "step": 1,
                "what_it_is": (
                    "Hiperparámetro principal. A diferencia de K-Means, aquí 'cortas' el dendrograma "
                    "en el nivel que produce el número deseado de clusters."
                ),
                "effect": (
                    "El dendrograma te muestra visualmente cuántos clusters son naturales — "
                    "busca el salto vertical más grande. Ese salto indica dónde cortar."
                ),
                "rule_of_thumb": "Observa el dendrograma y corta donde los saltos sean más grandes.",
            },
            "linkage": {
                "label": "Criterio de enlace (linkage)",
                "is_hyperparam": True,
                "type": "select",
                "options": ["ward", "complete", "average", "single"],
                "default": "ward",
                "what_it_is": (
                    "Hiperparámetro que define cómo se mide la distancia entre dos clusters "
                    "al decidir cuáles unir en cada paso."
                ),
                "effect": (
                    "ward: minimiza varianza al unir — genera clusters más esféricos y compactos. Recomendado. "
                    "complete: distancia máxima — clusters más compactos, sensible a outliers. "
                    "average: distancia promedio — balance entre compacidad y tamaño. "
                    "single: distancia mínima — puede generar efecto 'cadena' (clusters largos)."
                ),
                "rule_of_thumb": "Usa ward para la mayoría. complete o average si los clusters son elongados.",
            },
            "affinity": {
                "label": "Métrica de distancia (affinity)",
                "is_hyperparam": True,
                "type": "select",
                "options": ["euclidean", "manhattan", "cosine"],
                "default": "euclidean",
                "what_it_is": (
                    "Métrica usada para calcular la distancia entre puntos. "
                    "Nota: ward solo funciona con euclidean."
                ),
                "effect": (
                    "euclidean: estándar, obligatorio con linkage='ward'. "
                    "manhattan: más robusto si hay outliers. "
                    "cosine: útil para datos de texto o proporciones."
                ),
                "rule_of_thumb": "Usa euclidean con ward. Manhattan con complete o average.",
            },
        },
        "metrics": {
            "Silhouette": "Cohesión y separación de clusters. Cercano a 1 = clusters bien definidos.",
            "Davies-Bouldin": "Compacidad y separación. Cercano a 0 = mejor.",
            "Calinski-Harabasz": "Razón varianza inter/intra-cluster. Mayor = mejor.",
        },
    },
}

# ─── Resumen de qué es y qué NO es un hiperparámetro ───────────────────────

HYPERPARAM_EXPLAINER = {
    "definition": (
        "Un hiperparámetro es una configuración del modelo que tú debes definir "
        "ANTES del entrenamiento. El modelo no los aprende de los datos — son decisiones "
        "de diseño que controlan cómo aprende el modelo."
    ),
    "not_hyperparam": (
        "Parámetros que NO son hiperparámetros:\n"
        "• Parámetros aprendidos: pesos (coeficientes β), centroides, vectores de soporte.\n"
        "• Parámetros de split: test_size, random_state del train_test_split.\n"
        "• Parámetros de convergencia: max_iter, tol — afectan el tiempo, no el modelo ideal.\n"
        "• Parámetros de capacidad técnica: probability en SVC, n_jobs."
    ),
    "examples": {
        "son_hiperparametros": ["C (SVM, LogReg)", "kernel (SVM)", "n_clusters (K-Means)", "n_components (PCA)", "penalty (LogReg)"],
        "no_son_hiperparametros": ["coef_ (pesos aprendidos)", "intercept_", "cluster_centers_", "test_size", "random_state", "max_iter"],
    },
    "how_to_tune": (
        "Para encontrar los mejores hiperparámetros:\n"
        "• GridSearchCV: prueba todas las combinaciones en una grilla.\n"
        "• RandomizedSearchCV: muestrea aleatoriamente — más eficiente con muchos parámetros.\n"
        "• Optuna / Hyperopt: optimización bayesiana — más inteligente que búsqueda exhaustiva.\n"
        "Siempre usa validación cruzada (CV) para evitar sobreajustar los hiperparámetros."
    ),
}

ALGO_SELECTION_GUIDE = {
    "flowchart": [
        {"question": "¿Tienes etiquetas (y)?", "yes": "supervisado", "no": "no_supervisado"},
        {"question": "¿La salida es continua?", "yes": "regresión", "no": "clasificación"},
        {"question": "¿Quieres descubrir grupos?", "yes": "kmeans", "no": "reducir_dims"},
    ],
    "cheatsheet": {
        "Regresión Lineal": "Salida continua + relación lineal + interpretabilidad",
        "Regresión Logística": "Clasificación binaria/multiclase + probabilidades + interpretabilidad",
        "SVM": "Clasificación + alta dimensión + margen máximo + datos medios",
        "K-Means": "Clustering + forma esférica + K conocido + datos escalados",
        "PCA": "Reducir dimensiones + visualización + preprocesamiento + datos lineales",
    },
}
