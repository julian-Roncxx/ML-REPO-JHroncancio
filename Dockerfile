# Imagen de la API de riesgo de crédito (FastAPI + Uvicorn)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/mlops_pipeline \
    MODEL_PATH=/app/artifacts/mejor_modelo.joblib

WORKDIR /app

# Dependencias (capa cacheable)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código y datos
COPY mlops_pipeline ./mlops_pipeline
COPY Base_de_datos.csv .

# Se entrena el modelo dentro de la imagen: así coincide exactamente con las
# versiones de las librerías instaladas (un .joblib exige la misma versión de scikit-learn).
RUN python mlops_pipeline/model_training_evaluation.py

# Usuario sin privilegios
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
