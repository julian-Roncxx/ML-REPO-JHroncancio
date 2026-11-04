"""
ft_engineering.py
-----------------
Primer componente del flujo de creación de modelos.

Responsabilidades:
  1. Cargar y limpiar los datos (decisiones tomadas en comprension_eda.ipynb).
  2. Crear variables derivadas.
  3. Dividir en conjuntos de entrenamiento y evaluación (estratificado).
  4. Construir el ColumnTransformer (numéricas, categóricas, categóricas ordinales).

Uso desde otros módulos:
    from ft_engineering import get_datasets
    X_train, X_test, y_train, y_test, preprocessor = get_datasets()
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
RUTA_CSV = Path(__file__).resolve().parent.parent / "Base_de_datos.csv"
TARGET = "Pago_atiempo"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Hallazgo del EDA (sección 4.4): 'puntaje' correlaciona ~0.92 con el objetivo y
# probablemente se calcula DESPUÉS del comportamiento de pago (fuga de datos).
# Por defecto se excluye. Poner True solo para comparar como referencia.
INCLUIR_PUNTAJE = False

CATEGORIAS_TENDENCIA = ["Decreciente", "Estable", "Creciente"]  # ordinal
CODIGOS_TIPO_CREDITO_FRECUENTES = [4, 9]                      # el resto se agrupa en "Otros"

# Columnas por tipo (se definen tras la ingeniería de variables)
COLS_NUMERICAS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
    "saldo_principal", "saldo_mora_codeudor", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    # derivadas
    "cuota_sobre_salario", "capital_sobre_salario", "deuda_sobre_salario",
    "creditos_total_sectores", "tiene_mora", "tiene_mora_codeudor", "sin_ingresos_dc",
]
COLS_CATEGORICAS = ["tipo_credito", "tipo_laboral"]
COLS_ORDINALES = ["tendencia_ingresos"]


# ----------------------------------------------------------------------------
# Limpieza (reglas halladas en el EDA)
# ----------------------------------------------------------------------------
def cargar_datos(ruta: Path = RUTA_CSV) -> pd.DataFrame:
    """Lee el CSV de la raíz del repositorio."""
    return pd.read_csv(ruta)


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica nulos, corrige valores inválidos y fija tipos de datos."""
    df = df.copy()

    # 1) tendencia_ingresos: solo 3 categorías válidas; lo demás (números sueltos) -> NaN
    t = df["tendencia_ingresos"].astype(object)
    df["tendencia_ingresos"] = t.where(t.isin(CATEGORIAS_TENDENCIA), np.nan)

    # 2) puntaje_datacredito <= 0 es imposible (rango real ~150-950) -> NaN
    df.loc[df["puntaje_datacredito"] <= 0, "puntaje_datacredito"] = np.nan

    # 3) tipo_credito es un código nominal; se agrupan los códigos raros en "Otros"
    df["tipo_credito"] = np.where(
        df["tipo_credito"].isin(CODIGOS_TIPO_CREDITO_FRECUENTES),
        df["tipo_credito"].astype(str), "Otros",
    )

    # 4) tipo_laboral como texto (object) con NaN estándar
    df["tipo_laboral"] = df["tipo_laboral"].astype(object)

    # 5) Fecha y objetivo
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])
    df[TARGET] = df[TARGET].astype("int8")
    return df


def crear_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Variables calculadas propuestas en el EDA (sección 7). No aprenden de los datos."""
    df = df.copy()
    salario = df["salario_cliente"].replace(0, np.nan)

    df["cuota_sobre_salario"] = df["cuota_pactada"] / salario
    df["capital_sobre_salario"] = df["capital_prestado"] / salario
    df["deuda_sobre_salario"] = df["total_otros_prestamos"] / salario
    df["creditos_total_sectores"] = df[
        ["creditos_sectorFinanciero", "creditos_sectorCooperativo", "creditos_sectorReal"]
    ].sum(axis=1)
    df["tiene_mora"] = (df["saldo_mora"].fillna(0) > 0).astype(int)
    df["tiene_mora_codeudor"] = (df["saldo_mora_codeudor"].fillna(0) > 0).astype(int)
    df["sin_ingresos_dc"] = df["promedio_ingresos_datacredito"].isna().astype(int)

    return df.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------
# Transformadores y pipelines
# ----------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Recorta cada columna a los percentiles [low, high] aprendidos en entrenamiento.

    Controla los valores extremos del EDA (salarios de miles de millones, etc.)
    sin eliminar filas y sin filtrar información del conjunto de prueba.
    """

    def __init__(self, low: float = 0.01, high: float = 0.99):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_ = X.quantile(self.low).to_numpy()
        self.upper_ = X.quantile(self.high).to_numpy()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return np.clip(X.to_numpy(dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(columnas_numericas=None, columnas_categoricas=None,
                       columnas_ordinales=None) -> ColumnTransformer:
    """ColumnTransformer con tres ramas: numeric, categoric y categoric ordinales."""
    columnas_numericas = columnas_numericas or COLS_NUMERICAS
    columnas_categoricas = columnas_categoricas or COLS_CATEGORICAS
    columnas_ordinales = columnas_ordinales or COLS_ORDINALES

    numeric = Pipeline([
        ("winsor", Winsorizer(0.01, 0.99)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categoric = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    # NaN = "SIN_DATO" (primera categoría); luego Decreciente < Estable < Creciente
    ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="SIN_DATO")),
        ("ordinal", OrdinalEncoder(
            categories=[["SIN_DATO"] + CATEGORIAS_TENDENCIA],
            handle_unknown="use_encoded_value", unknown_value=-1,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, columnas_numericas),
            ("categoric", categoric, columnas_categoricas),
            ("categoric ordinales", ordinal, columnas_ordinales),
        ],
        remainder="drop",
    )


# ----------------------------------------------------------------------------
# Función principal del módulo
# ----------------------------------------------------------------------------
def preparar_dataframe() -> pd.DataFrame:
    """Carga, limpia y crea variables derivadas."""
    return crear_variables_derivadas(limpiar_datos(cargar_datos()))


def get_datasets(incluir_puntaje: bool = INCLUIR_PUNTAJE, test_size: float = TEST_SIZE,
                 random_state: int = RANDOM_STATE):
    """Devuelve X_train, X_test, y_train, y_test y el preprocesador SIN ajustar.

    El preprocesador se ajusta dentro del pipeline de cada modelo, solo con los
    datos de entrenamiento (así no hay fuga de información hacia la evaluación).
    """
    df = preparar_dataframe()

    numericas = COLS_NUMERICAS + (["puntaje"] if incluir_puntaje else [])
    columnas = numericas + COLS_CATEGORICAS + COLS_ORDINALES
    X, y = df[columnas], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    preprocessor = build_preprocessor(columnas_numericas=numericas)
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, pre = get_datasets()
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"% no pago (clase 0)  train: {(y_train == 0).mean():.2%} | test: {(y_test == 0).mean():.2%}")
    Xt = pre.fit_transform(X_train)
    print(f"Matriz transformada (train): {Xt.shape}")
    print(f"NaN tras transformar: {int(np.isnan(np.asarray(Xt)).sum())}")
