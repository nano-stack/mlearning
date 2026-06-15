# MLearning — La app definitiva para aprender Machine Learning

Aplicación interactiva para aprender, entrenar y visualizar algoritmos de Machine Learning.
Construida con **Streamlit + Manim + scikit-learn**.

## Instalación

```bash
cd MLearning
pip install -r requirements.txt
```

### Instalar Manim (animaciones matemáticas)
```bash
pip install manim
# macOS también requiere:
brew install cairo pango ffmpeg
```

## Uso

### Opción A: Pre-renderizar animaciones (recomendado)
```bash
python render_animations.py   # solo la primera vez
streamlit run app.py
```

### Opción B: Lanzar directamente (animaciones se renderizan al vuelo)
```bash
streamlit run app.py
```

## Modelos disponibles

| Modelo | Tipo | Tarea |
|--------|------|-------|
| Regresión Lineal | Supervisado | Regresión |
| Regresión Logística | Supervisado | Clasificación |
| SVM | Supervisado | Clasificación |
| K-Means | No supervisado | Clustering |
| PCA | No supervisado | Reducción dimensional |

## Estructura
```
MLearning/
├── app.py                    # Punto de entrada
├── pages/                    # Una pantalla por archivo
│   ├── landing.py
│   ├── select_type.py
│   ├── select_model.py
│   ├── data_loader.py
│   ├── hyperparams.py
│   └── results.py
├── animations/
│   └── manim_scenes.py       # Todas las escenas Manim
├── knowledge/
│   └── models_knowledge.py   # Base de conocimiento local
├── utils/
│   ├── data_processor.py
│   └── model_runner.py
└── assets/
    ├── styles/main.css
    ├── icons/icons.py
    └── videos/               # Animaciones pre-renderizadas
```
