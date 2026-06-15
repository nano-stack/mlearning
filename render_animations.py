"""
Script para pre-renderizar todas las animaciones Manim.
Ejecutar UNA VEZ antes de lanzar la app:
    python render_animations.py
"""
import subprocess
import shutil
import sys
from pathlib import Path

SCENES = [
    "LandingAnimation",
    "LinearRegressionScene",
    "LogisticRegressionScene",
    "KMeansScene",
    "PCAScene",
    "SVMScene",
    "DecisionTreeScene",
    "RandomForestScene",
    "KNNScene",
    "GradientBoostingScene",
    "NaiveBayesScene",
    "DBSCANScene",
    "HierarchicalScene",
]

OUTPUT_DIR = Path("assets/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def render_scene(scene_name: str, quality: str = "m") -> bool:
    """quality: l=low, m=medium, h=high, k=4K"""
    out_path = OUTPUT_DIR / f"{scene_name}.mp4"
    if out_path.exists():
        print(f"  [SKIP] {scene_name} ya existe.")
        return True

    print(f"  [RENDER] {scene_name}...")
    result = subprocess.run(
        [sys.executable, "-m", "manim", f"-q{quality}", "--media_dir", "assets",
         "animations/manim_scenes.py", scene_name],
        capture_output=True, text=True, timeout=300
    )

    # Buscar el MP4 generado
    candidates = list(Path("assets").rglob(f"{scene_name}*.mp4"))
    if candidates:
        shutil.copy(candidates[0], out_path)
        print(f"  [OK] {scene_name} → {out_path}")
        return True

    print(f"  [ERROR] {scene_name}")
    if result.stderr:
        print(result.stderr[-500:])
    return False


if __name__ == "__main__":
    print("MLearning — Pre-renderizado de animaciones Manim")
    print("=" * 50)
    ok = 0
    for scene in SCENES:
        if render_scene(scene):
            ok += 1
    print(f"\nCompletado: {ok}/{len(SCENES)} animaciones renderizadas.")
    print("Ahora puedes lanzar la app con: streamlit run app.py")
