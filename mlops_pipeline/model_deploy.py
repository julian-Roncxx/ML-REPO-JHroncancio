"""
model_deploy.py
---------------
Despliegue del mejor modelo como servicio (FastAPI).

Endpoints:
    GET  /health        -> estado del servicio y del modelo
    POST /predict       -> predicción de uno o varios créditos (JSON, por lotes)
    POST /predict/csv   -> predicción por lotes desde un archivo CSV

Ejecución local:
    uvicorn model_deploy:app --app-dir mlops_pipeline --host 0.0.0.0 --port 8000
    Documentación interactiva: http://localhost:8000/docs

El modelo se lee de MODEL_PATH (variable de entorno) o de artifacts/mejor_modelo.joblib.
"""
import io
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import preparar_entrada  # noqa: E402  (misma limpieza que en el entrenamiento)

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO = Path(os.getenv("MODEL_PATH", RAIZ / "artifacts" / "mejor_modelo.joblib"))
MAX_REGISTROS = 10_000   # tope por solicitud para proteger el servicio
UMBRAL_ALTO_RIESGO = 0.5


# ----------------------------------------------------------------------------
# Esquemas de entrada y salida (reglas de validación tomadas del EDA)
# ----------------------------------------------------------------------------
class Credito(BaseModel):
    """Un crédito con las variables crudas. Los campos opcionales se imputan en el pipeline."""
    model_config = ConfigDict(extra="ignore")   # ignora columnas que el modelo no usa

    tipo_credito: int
    capital_prestado: float = Field(gt=0, le=100_000_000)
    plazo_meses: int = Field(ge=1, le=120)
    edad_cliente: int = Field(ge=18, le=90)
    tipo_laboral: str
    salario_cliente: float = Field(ge=0, le=100_000_000)
    total_otros_prestamos: float = Field(ge=0, le=1_000_000_000)
    cuota_pactada: float = Field(gt=0)
    cant_creditosvigentes: int = Field(ge=0)
    huella_consulta: int = Field(ge=0)
    puntaje_datacredito: Optional[float] = None
    saldo_mora: Optional[float] = Field(default=None, ge=0)
    saldo_total: Optional[float] = Field(default=None, ge=0)
    saldo_principal: Optional[float] = Field(default=None, ge=0)
    saldo_mora_codeudor: Optional[float] = Field(default=None, ge=0)
    creditos_sectorFinanciero: Optional[int] = Field(default=None, ge=0)
    creditos_sectorCooperativo: Optional[int] = Field(default=None, ge=0)
    creditos_sectorReal: Optional[int] = Field(default=None, ge=0)
    promedio_ingresos_datacredito: Optional[float] = Field(default=None, ge=0)
    tendencia_ingresos: Optional[str] = None


class SolicitudLote(BaseModel):
    registros: List[Credito] = Field(min_length=1, max_length=MAX_REGISTROS)


class Prediccion(BaseModel):
    prediccion: int = Field(description="1 = pagaría a tiempo, 0 = riesgo de no pago")
    prob_pago_atiempo: float
    prob_no_pago: float
    nivel_riesgo: str


class RespuestaLote(BaseModel):
    n_registros: int
    predicciones: List[Prediccion]


# ----------------------------------------------------------------------------
# Lógica de predicción
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def cargar_modelo(ruta: str = str(RUTA_MODELO)):
    """Carga el modelo una sola vez (queda en caché)."""
    if not Path(ruta).exists():
        raise FileNotFoundError(f"No se encontró el modelo en {ruta}")
    return joblib.load(ruta)


def predecir(df_crudo: pd.DataFrame) -> List[Prediccion]:
    """Aplica las transformaciones necesarias y devuelve las predicciones."""
    modelo = cargar_modelo()
    X = preparar_entrada(df_crudo)
    proba_pago = modelo.predict_proba(X)[:, 1]
    clases = modelo.predict(X)
    salida = []
    for clase, p in zip(clases, proba_pago):
        p_no = 1 - float(p)
        nivel = "alto" if p_no >= 0.5 else "medio" if p_no >= 0.3 else "bajo"
        salida.append(Prediccion(prediccion=int(clase), prob_pago_atiempo=round(float(p), 4),
                                 prob_no_pago=round(p_no, 4), nivel_riesgo=nivel))
    return salida


# ----------------------------------------------------------------------------
# Aplicación
# ----------------------------------------------------------------------------
app = FastAPI(title="API de riesgo de crédito",
              description="Predice si un crédito se pagará a tiempo.", version="1.0.0")


@app.get("/health")
def health():
    try:
        cargar_modelo()
        return {"estado": "ok", "modelo": "cargado"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=RespuestaLote)
def predict(solicitud: SolicitudLote):
    """Recibe uno o varios créditos en JSON y devuelve una predicción por registro."""
    df = pd.DataFrame([r.model_dump() for r in solicitud.registros])
    try:
        preds = predecir(df)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return RespuestaLote(n_registros=len(preds), predicciones=preds)


@app.post("/predict/csv", response_model=RespuestaLote)
async def predict_csv(archivo: UploadFile = File(...)):
    """Recibe un CSV con créditos (mismas columnas que Base_de_datos.csv) y predice por lote."""
    if not (archivo.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un .csv")
    contenido = await archivo.read()
    try:
        df = pd.read_csv(io.BytesIO(contenido))
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer el CSV")
    if df.empty or len(df) > MAX_REGISTROS:
        raise HTTPException(status_code=400, detail=f"El CSV debe tener entre 1 y {MAX_REGISTROS} filas")
    try:
        preds = predecir(df)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Falta una columna requerida: {e}")
    return RespuestaLote(n_registros=len(preds), predicciones=preds)