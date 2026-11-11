"""Verifica que el repositorio cumpla la estructura exigida en el entregable."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

ARCHIVOS_PIPELINE = ["Cargar_datos.ipynb", "comprension_eda.ipynb", "ft_engineering.py",
                     "model_training_evaluation.py", "model_deploy.py", "model_monitoring.py"]
ARCHIVOS_RAIZ = ["Base_de_datos.csv", "requirements.txt", ".gitignore", "setup.bat", "readme.md"]


def test_archivos_de_mlops_pipeline():
    for nombre in ARCHIVOS_PIPELINE:
        ruta = RAIZ / "mlops_pipeline" / nombre
        assert ruta.is_file() and ruta.stat().st_size > 0, f"falta o está vacío: {nombre}"


def test_archivos_de_la_raiz():
    for nombre in ARCHIVOS_RAIZ:
        ruta = RAIZ / nombre
        assert ruta.is_file() and ruta.stat().st_size > 0, f"falta o está vacío: {nombre}"
