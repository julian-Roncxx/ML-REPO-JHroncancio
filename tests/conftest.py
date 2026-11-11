"""Configuración común de las pruebas: deja mlops_pipeline importable y entrena un modelo ligero."""
import os
import sys
from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "mlops_pipeline"))


@pytest.fixture(scope="session")
def datasets():
    from ft_engineering import get_datasets
    return get_datasets()


@pytest.fixture(scope="session")
def modelo_ruta(tmp_path_factory, datasets):
    """Entrena una regresión logística y la guarda en una carpeta temporal."""
    X_train, _, y_train, _, preprocessor = datasets
    pipe = Pipeline([("preprocessor", preprocessor),
                     ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipe.fit(X_train, y_train)
    ruta = tmp_path_factory.mktemp("modelo") / "mejor_modelo.joblib"
    joblib.dump(pipe, ruta)
    return ruta


@pytest.fixture(scope="session")
def cliente(modelo_ruta):
    os.environ["MODEL_PATH"] = str(modelo_ruta)
    import importlib
    import model_deploy
    importlib.reload(model_deploy)
    from fastapi.testclient import TestClient
    return TestClient(model_deploy.app)
