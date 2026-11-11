# Predicción de pago a tiempo de créditos — Proyecto final de Machine Learning

Proyecto de MLOps de punta a punta: exploración de datos, ingeniería de características, entrenamiento y selección de modelos, monitoreo de *data drift* y despliegue como API.

**Autor:** Julián Roncancio · **Docente:** Juan Sebastián Parra Sánchez

---

## 1. Caso de negocio

Una entidad financiera otorga créditos y necesita saber, **antes de desembolsar**, qué créditos tienen más probabilidad de no pagarse a tiempo. Con esa señal puede ajustar la aprobación, el monto, el plazo o la tasa y reducir las pérdidas por cartera vencida.

- **Problema:** clasificación binaria supervisada.
- **Variable objetivo:** `Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó a tiempo).
- **Base de datos:** 10,763 créditos y 23 columnas (nov-2024 a abr-2026) con datos del préstamo, del cliente y de su historial en centrales de riesgo.
- **Costo asimétrico:** aprobar a quien no paga cuesta más que rechazar a quien sí pagaría. Por eso la clase de interés es la **clase 0 (no pago)** y no se usa la *accuracy* como métrica principal.

## 2. Principales hallazgos del EDA (`comprension_eda.ipynb`)

1. **Objetivo muy desbalanceado:** ~95% paga a tiempo y ~5% no. Un modelo que siempre prediga "paga" acierta 95% sin servir de nada.
2. **Posible fuga de datos en `puntaje`:** correlaciona ≈ 0.92 con el objetivo, y todos los clientes que no pagaron tienen `puntaje` ≤ 62.7. Probablemente se calcula *después* del comportamiento de pago, así que **se excluye del modelo**. Al incluirla, todos los modelos alcanzan ROC-AUC = 1.00, lo cual no es creíble.
3. **Calidad de datos:** `tendencia_ingresos` traía números sueltos en lugar de categorías (se pasaron a nulo), `puntaje_datacredito` tenía valores imposibles (≤ 0), y había edades de hasta 123 años y salarios de miles de millones.
4. **Nulos con significado:** ~27% de los clientes no tiene ingresos reportados por la central, y ese vacío ya aporta información sobre el riesgo.
5. **Señal débil por variable:** ninguna variable (salvo `puntaje`) supera ~0.1 de correlación con el objetivo. El riesgo depende de combinaciones de variables, no de una sola.
6. **Temporalidad:** el volumen de desembolsos cae con el tiempo y los créditos recientes pueden tener etiquetas aún incompletas.

## 3. Proceso

| Etapa | Archivo | Qué hace |
|---|---|---|
| Carga | `Cargar_datos.ipynb` | Lee `Base_de_datos.csv` y muestra una vista inicial. |
| EDA | `comprension_eda.ipynb` | Exploración, tipos, nulos, análisis uni/bi/multivariable, reglas de validación y variables derivadas. |
| Ingeniería de características | `ft_engineering.py` | Limpieza, 7 variables derivadas, partición 80/20 estratificada y `ColumnTransformer` (numéricas, categóricas y ordinales). |
| Entrenamiento y evaluación | `model_training_evaluation.py` | Compara 5 modelos con `build_model` y `summarize_classification`, y selecciona el mejor. |
| Monitoreo | `model_monitoring.py` | Métricas de data drift y aplicación en Streamlit. |
| Despliegue | `model_deploy.py` | API con FastAPI y predicción por lotes. |

**Preprocesamiento:** las numéricas se recortan a los percentiles 1–99 (control de extremos), se imputan con la mediana y se estandarizan; las categóricas se imputan y se codifican *one-hot*; `tendencia_ingresos` se codifica como ordinal (SIN_DATO < Decreciente < Estable < Creciente). El preprocesador se ajusta solo con los datos de entrenamiento para evitar fugas.

## 4. Resultados de los modelos

Clase de interés: **0 (no pagó a tiempo)**. Evaluación sobre 2,153 créditos de prueba.

| Modelo | ROC-AUC | PR-AUC (clase 0) | Recall (clase 0) | Precision (clase 0) | F1 (clase 0) | Desv. ROC-AUC (CV) | Tiempo de entrenamiento (s) |
|---|---|---|---|---|---|---|---|
| **Regresión Logística** ✅ | **0.696** | 0.121 | 0.657 | 0.083 | 0.148 | **0.007** | **0.07** |
| Random Forest | 0.665 | 0.146 | 0.118 | 0.250 | 0.160 | 0.012 | 3.04 |
| Gradient Boosting | 0.639 | 0.147 | 0.314 | 0.093 | 0.143 | 0.034 | 1.01 |
| Árbol de Decisión | 0.625 | 0.097 | 0.667 | 0.065 | 0.119 | 0.027 | 0.08 |
| Baseline (Dummy) | 0.500 | 0.047 | 0.000 | 0.000 | 0.000 | — | 0.05 |

> Los valores exactos pueden variar levemente según la versión de las librerías. Se regeneran ejecutando `model_training_evaluation.py`.

**Selección:** el mejor modelo se elige con tres criterios — *performance* (promedio de ROC-AUC y PR-AUC), *consistency* (baja variación en validación cruzada de 5 folds) y *scalability* (tiempos de entrenamiento y predicción), con pesos 60 / 20 / 20.

**Lectura honesta:** el mejor modelo es modesto (ROC-AUC ≈ 0.70), coherente con la señal débil hallada en el EDA. La Regresión Logística detecta cerca de 2 de cada 3 clientes que no pagan, pero con baja precisión (muchas falsas alarmas), así que sirve como filtro de riesgo y no como decisión automática. La *accuracy* engaña: Random Forest tiene 94% de accuracy pero detecta pocos incumplimientos.

### Gráficos comparativos

<img width="1430" height="650" alt="comparacion_metricas" src="https://github.com/user-attachments/assets/2b4e7839-eaf4-41f3-bd34-dda54d548fd6" />

<img width="1690" height="650" alt="curvas_roc_pr" src="https://github.com/user-attachments/assets/a2d87af0-c8ae-48cb-944f-a7f11e5a5d98" />

<img width="1690" height="585" alt="consistencia_escalabilidad" src="https://github.com/user-attachments/assets/f2326c55-b9e4-4bd0-bbd0-abcb2483e6c1" />

## 5. Monitoreo y data drift

`model_monitoring.py` une los datos con las predicciones del modelo, toma como **referencia** los créditos anteriores al 1-abr-2025 y compara la población **actual** (desde esa fecha) por periodo con:

- **KS** (Kolmogorov-Smirnov) para variables numéricas.
- **PSI** (Population Stability Index) para numéricas y categóricas. Semáforo: 🟢 < 0.10 · 🟡 0.10–0.25 · 🔴 ≥ 0.25.
- **Jensen-Shannon** para numéricas y categóricas.
- **Chi-cuadrado** para categóricas.

**Resultado actual vs. referencia:** `promedio_ingresos_datacredito` está en **rojo** (PSI 0.37) y `plazo_meses` en **amarillo** (PSI 0.24); las demás variables y la distribución de las predicciones están estables. Con muestras grandes, KS y chi-cuadrado marcan casi todo como "significativo", por eso el semáforo se basa en el PSI, que mide la *magnitud* del cambio.

La aplicación en Streamlit muestra el semáforo, la tabla de métricas por variable, la comparación histórica vs. actual, la evolución temporal del PSI con detección de tendencias o cambios abruptos, el desempeño del modelo por periodo, y mensajes automáticos con recomendaciones de revisión o reentrenamiento.

## 6. Despliegue (API)

`model_deploy.py` expone el modelo con FastAPI:

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado del servicio y del modelo. |
| `POST /predict` | Predicción de uno o varios créditos en JSON (por lotes). |
| `POST /predict/csv` | Predicción por lotes desde un archivo CSV. |

Las entradas se validan con las reglas del EDA (por ejemplo, edad entre 18 y 90). Ejemplo:

```json
POST /predict
{"registros": [{"tipo_credito": 4, "capital_prestado": 1852560, "plazo_meses": 12,
  "edad_cliente": 32, "tipo_laboral": "Empleado", "salario_cliente": 3500000,
  "total_otros_prestamos": 1000000, "cuota_pactada": 128650,
  "cant_creditosvigentes": 0, "huella_consulta": 2}]}
```

La imagen Docker (`Dockerfile`, `.dockerignore`) instala las dependencias, entrena el modelo dentro de la imagen y arranca Uvicorn en el puerto 8000.

## 7. Cómo ejecutar

```powershell
# 1) Entorno e instalación
setup.bat

# 2) Entrenar y evaluar (genera el modelo y los gráficos en artifacts/)
venv\Scripts\python.exe mlops_pipeline\model_training_evaluation.py

# 3) Monitoreo: aplicación Streamlit
venv\Scripts\streamlit run mlops_pipeline\model_monitoring.py

# 4) API
venv\Scripts\uvicorn model_deploy:app --app-dir mlops_pipeline --port 8000
#    Documentación interactiva: http://localhost:8000/docs

# 5) Docker
docker build -t riesgo-credito .
docker run -p 8000:8000 riesgo-credito
```

## 8. Estructura del repositorio

```
├── mlops_pipeline/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_deploy.py
│   └── model_monitoring.py
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
├── setup.bat
├── Dockerfile
├── .dockerignore
└── readme.md
```

Ramas: `developer` (desarrollo), `certification` (validación) y `master` (producción).
