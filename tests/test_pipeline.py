"""Pruebas del módulo ft_engineering: datos, variables derivadas y división."""
import numpy as np

from ft_engineering import (COLS_CATEGORICAS, COLS_ORDINALES, DERIVADAS, TARGET,
                            cargar_datos, crear_variables_derivadas, limpiar_datos)


def test_dataset_tiene_forma_esperada():
    df = cargar_datos()
    assert df.shape == (10763, 23)
    assert set(df[TARGET].unique()) == {0, 1}


def test_limpieza_deja_solo_categorias_validas_de_tendencia():
    df = limpiar_datos(cargar_datos())
    validas = {"Decreciente", "Estable", "Creciente"}
    assert set(df["tendencia_ingresos"].dropna().unique()) <= validas


def test_puntaje_datacredito_no_tiene_valores_imposibles():
    df = limpiar_datos(cargar_datos())
    assert (df["puntaje_datacredito"].dropna() > 0).all()


def test_variables_derivadas_se_crean_sin_infinitos():
    df = crear_variables_derivadas(limpiar_datos(cargar_datos()))
    for col in DERIVADAS:
        assert col in df.columns
    assert not np.isinf(df[DERIVADAS].to_numpy(dtype=float)).any()


def test_split_es_estratificado_y_tiene_tamano_esperado(datasets):
    X_train, X_test, y_train, y_test, _ = datasets
    assert len(X_train) == 8610 and len(X_test) == 2153
    assert abs(y_train.mean() - y_test.mean()) < 0.005


def test_puntaje_no_entra_al_modelo_por_defecto(datasets):
    X_train, *_ = datasets
    assert "puntaje" not in X_train.columns
    assert TARGET not in X_train.columns


def test_preprocesador_produce_matriz_sin_nulos(datasets):
    from sklearn.base import clone
    X_train, _, _, _, preprocessor = datasets
    Xt = clone(preprocessor).fit_transform(X_train)
    assert not np.isnan(np.asarray(Xt, dtype=float)).any()
    assert COLS_CATEGORICAS and COLS_ORDINALES
