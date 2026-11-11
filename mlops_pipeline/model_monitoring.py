"""
model_monitoring.py
-------------------
Monitoreo del modelo y detección de data drift.

Une los datos con los pronósticos del modelo, toma una población de REFERENCIA (histórica)
y la compara contra la población ACTUAL, periodo a periodo, con cuatro medidas:
  - Kolmogorov-Smirnov (KS)          -> variables numéricas
  - PSI (Population Stability Index) -> numéricas y categóricas
  - Jensen-Shannon                   -> numéricas y categóricas
  - Chi-cuadrado                     -> variables categóricas

El mismo archivo contiene la aplicación de Streamlit (así no se agregan archivos a la
estructura exigida por el pipeline de Jenkins):

    streamlit run mlops_pipeline/model_monitoring.py      # aplicación
    python mlops_pipeline/model_monitoring.py             # reporte en consola + CSV
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import (COLS_CATEGORICAS, COLS_NUMERICAS, COLS_ORDINALES, TARGET,  # noqa: E402
                            columnas_modelo, preparar_dataframe)

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO = Path(os.getenv("MODEL_PATH", RAIZ / "artifacts" / "mejor_modelo.joblib"))
DIR_SALIDAS = RAIZ / "artifacts"

FECHA_CORTE_REFERENCIA = "2025-04-01"   # antes = referencia; desde aquí = población actual
PERIODICIDAD = "M"                      # "M" mensual, "Q" trimestral
MIN_REGISTROS_PERIODO = 30              # menos que esto -> muestra insuficiente

# Umbrales estándar de la industria para PSI
PSI_AMARILLO = 0.10
PSI_ROJO = 0.25
PROPORCION_ROJAS_CRITICA = 0.20         # si >20% de las variables están en rojo -> alerta crítica

VARIABLES_NUMERICAS = [c for c in COLS_NUMERICAS if c not in
                       ("tiene_mora", "tiene_mora_codeudor", "sin_ingresos_dc")]
VARIABLES_CATEGORICAS = COLS_CATEGORICAS + COLS_ORDINALES + ["tiene_mora", "tiene_mora_codeudor", "sin_ingresos_dc"]
VARIABLES_MONITOREADAS = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS + ["prob_no_pago"]


# ----------------------------------------------------------------------------
# Datos + pronósticos
# ----------------------------------------------------------------------------
def cargar_modelo(ruta: Path = RUTA_MODELO):
    """Carga el mejor modelo guardado por model_training_evaluation.py."""
    if not Path(ruta).exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero: python mlops_pipeline/model_training_evaluation.py")
    return joblib.load(ruta)


def construir_tabla_monitoreo(modelo=None) -> pd.DataFrame:
    """Tabla con los datos, la fecha, la probabilidad de no pago y la predicción del modelo."""
    modelo = modelo or cargar_modelo()
    df = preparar_dataframe()
    X = df[columnas_modelo()]
    df["prob_no_pago"] = 1 - modelo.predict_proba(X)[:, 1]
    df["prediccion"] = modelo.predict(X)
    df["periodo"] = df["fecha_prestamo"].dt.to_period(PERIODICIDAD)
    return df.sort_values("fecha_prestamo").reset_index(drop=True)


def dividir_referencia_actual(df: pd.DataFrame, fecha_corte=FECHA_CORTE_REFERENCIA):
    """Referencia = créditos anteriores a la fecha de corte; actual = los posteriores."""
    corte = pd.Timestamp(fecha_corte)
    return df[df["fecha_prestamo"] < corte], df[df["fecha_prestamo"] >= corte]


# ----------------------------------------------------------------------------
# Métricas de drift
# ----------------------------------------------------------------------------
def _distribuciones_numericas(ref: pd.Series, act: pd.Series, bins: int = 10):
    """Probabilidades por intervalo, con cortes por cuantiles de la referencia."""
    ref, act = ref.dropna(), act.dropna()
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(cortes) < 3:                       # variable casi constante
        cortes = np.array([-np.inf, ref.median(), np.inf])
    cortes[0], cortes[-1] = -np.inf, np.inf
    p = np.histogram(ref, cortes)[0] / max(len(ref), 1)
    q = np.histogram(act, cortes)[0] / max(len(act), 1)
    return p, q


def _distribuciones_categoricas(ref: pd.Series, act: pd.Series):
    """Probabilidades por categoría (los NaN cuentan como 'SIN_DATO')."""
    ref = ref.astype(object).where(ref.notna(), "SIN_DATO")
    act = act.astype(object).where(act.notna(), "SIN_DATO")
    cats = sorted(set(ref.unique()) | set(act.unique()), key=str)
    p = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    q = act.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return p, q, cats


def calcular_psi(p: np.ndarray, q: np.ndarray, eps: float = 1e-4) -> float:
    """Population Stability Index entre dos distribuciones discretas."""
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def calcular_js(p: np.ndarray, q: np.ndarray) -> float:
    """Distancia de Jensen-Shannon (base 2: va de 0 a 1)."""
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    return float(jensenshannon(p, q, base=2))


def clasificar_psi(psi: float) -> str:
    if np.isnan(psi):
        return "sin dato"
    if psi >= PSI_ROJO:
        return "rojo"
    if psi >= PSI_AMARILLO:
        return "amarillo"
    return "verde"


def drift_variable(ref: pd.Series, act: pd.Series, tipo: str) -> dict:
    """Todas las métricas de drift de una variable."""
    if tipo == "numerica":
        p, q = _distribuciones_numericas(ref, act)
        ks = stats.ks_2samp(ref.dropna(), act.dropna())
        ks_stat, ks_p, chi2_p = float(ks.statistic), float(ks.pvalue), np.nan
    else:
        p, q, cats = _distribuciones_categoricas(ref, act)
        ks_stat = ks_p = np.nan
        conteos = np.vstack([p * len(ref), q * len(act)])
        conteos = conteos[:, conteos.sum(axis=0) > 0]
        chi2_p = float(stats.chi2_contingency(conteos)[1]) if conteos.shape[1] > 1 else np.nan
    psi = calcular_psi(p, q)
    return {"ks_stat": ks_stat, "ks_pvalue": ks_p, "psi": psi,
            "js": calcular_js(p, q), "chi2_pvalue": chi2_p, "estado": clasificar_psi(psi)}


def calcular_drift(ref: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    """Tabla de drift por variable (una fila por variable)."""
    filas = []
    for var in VARIABLES_MONITOREADAS:
        if var not in ref.columns:
            continue
        tipo = "categorica" if var in VARIABLES_CATEGORICAS else "numerica"
        filas.append({"variable": var, "tipo": tipo, **drift_variable(ref[var], act[var], tipo)})
    return pd.DataFrame(filas).sort_values("psi", ascending=False).reset_index(drop=True)


def calcular_drift_temporal(df: pd.DataFrame, fecha_corte=FECHA_CORTE_REFERENCIA,
                            min_registros: int = MIN_REGISTROS_PERIODO) -> pd.DataFrame:
    """Drift de cada periodo (mes o trimestre) frente a la referencia."""
    ref, act = dividir_referencia_actual(df, fecha_corte)
    resultados = []
    for periodo, grupo in act.groupby("periodo"):
        if len(grupo) < min_registros:
            continue                           # muestra insuficiente: se omite
        t = calcular_drift(ref, grupo)
        t.insert(0, "periodo", str(periodo))
        t.insert(1, "n_registros", len(grupo))
        resultados.append(t)
    return pd.concat(resultados, ignore_index=True) if resultados else pd.DataFrame()


# ----------------------------------------------------------------------------
# Desempeño del modelo (si ya se conoce el resultado real)
# ----------------------------------------------------------------------------
def desempeno_por_periodo(df: pd.DataFrame, min_registros: int = MIN_REGISTROS_PERIODO) -> pd.DataFrame:
    """ROC-AUC y recall de la clase 0 por periodo. Ojo: los créditos recientes pueden tener
    etiquetas incompletas (aún no cumplen su plazo), por lo que se interpretan con cautela."""
    from sklearn.metrics import recall_score, roc_auc_score
    filas = []
    for periodo, g in df.groupby("periodo"):
        y_riesgo = (g[TARGET] == 0).astype(int)
        if len(g) < min_registros or y_riesgo.nunique() < 2:
            continue
        filas.append({"periodo": str(periodo), "n_registros": len(g),
                      "tasa_no_pago_real": y_riesgo.mean(),
                      "tasa_no_pago_predicha": (g["prediccion"] == 0).mean(),
                      "roc_auc": roc_auc_score(y_riesgo, g["prob_no_pago"]),
                      "recall_0": recall_score(g[TARGET], g["prediccion"], pos_label=0, zero_division=0)})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# Alertas y recomendaciones
# ----------------------------------------------------------------------------
def detectar_tendencia(serie: pd.Series, umbral_salto: float = 0.10) -> str:
    """Describe la evolución de una serie (p. ej. PSI por periodo)."""
    s = serie.dropna().to_numpy()
    if len(s) < 3:
        return "insuficiente"
    pendiente = np.polyfit(np.arange(len(s)), s, 1)[0]
    if np.max(np.diff(s)) >= umbral_salto:
        return "cambio abrupto"
    if pendiente > 0.01:
        return "tendencia creciente"
    return "estable"


def generar_alertas(tabla_drift: pd.DataFrame) -> list:
    """Mensajes automáticos según los umbrales. Devuelve (nivel, mensaje)."""
    alertas = []
    if tabla_drift.empty:
        return [("info", "No hay periodos con suficientes registros para evaluar drift.")]
    rojas = tabla_drift[tabla_drift["estado"] == "rojo"]
    amarillas = tabla_drift[tabla_drift["estado"] == "amarillo"]
    proporcion = len(rojas) / len(tabla_drift)

    if not rojas.empty:
        nombres = ", ".join(rojas["variable"].head(5))
        alertas.append(("rojo", f"Drift crítico (PSI ≥ {PSI_ROJO}) en {len(rojas)} variable(s): {nombres}."))
    if not amarillas.empty:
        alertas.append(("amarillo", f"Drift moderado (PSI entre {PSI_AMARILLO} y {PSI_ROJO}) en "
                                    f"{len(amarillas)} variable(s): {', '.join(amarillas['variable'].head(5))}."))
    pred = tabla_drift[tabla_drift["variable"] == "prob_no_pago"]
    if not pred.empty and pred["estado"].iloc[0] in ("rojo", "amarillo"):
        alertas.append((pred["estado"].iloc[0], "La distribución de las predicciones del modelo cambió: "
                        "el desempeño puede estar degradándose."))
    if proporcion >= PROPORCION_ROJAS_CRITICA:
        alertas.append(("rojo", f"{proporcion:.0%} de las variables están en rojo: se recomienda "
                                "REENTRENAR el modelo con datos recientes."))
    elif not rojas.empty:
        alertas.append(("amarillo", "Revisar las variables en rojo (cambios en la captura de datos o en el "
                                    "perfil de clientes) antes de decidir un reentrenamiento."))
    if not alertas:
        alertas.append(("verde", "Sin desviaciones significativas: el modelo puede seguir en operación."))
    return alertas


# ----------------------------------------------------------------------------
# Reporte por consola
# ----------------------------------------------------------------------------
def ejecutar_monitoreo(guardar: bool = True) -> pd.DataFrame:
    df = construir_tabla_monitoreo()
    ref, act = dividir_referencia_actual(df)
    print(f"Referencia: {len(ref):,} créditos | Actual: {len(act):,} créditos")
    tabla = calcular_drift(ref, act)
    print("\n=== DRIFT: población actual vs referencia ===")
    print(tabla.round(3).to_string(index=False))
    print("\n=== ALERTAS ===")
    for nivel, msg in generar_alertas(tabla):
        print(f"[{nivel.upper()}] {msg}")
    temporal = calcular_drift_temporal(df)
    if guardar:
        DIR_SALIDAS.mkdir(exist_ok=True)
        tabla.to_csv(DIR_SALIDAS / "drift_actual_vs_referencia.csv", index=False)
        temporal.to_csv(DIR_SALIDAS / "drift_temporal.csv", index=False)
        print(f"\nResultados guardados en {DIR_SALIDAS}")
    return tabla


# ----------------------------------------------------------------------------
# Aplicación Streamlit
# ----------------------------------------------------------------------------
COLORES = {"verde": "#2e9e4f", "amarillo": "#e8b923", "rojo": "#d03b3b", "sin dato": "#999999"}
ICONOS = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴", "sin dato": "⚪", "info": "ℹ️"}


def _grafico_comparacion(ref: pd.DataFrame, act: pd.DataFrame, variable: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.4))
    if variable in VARIABLES_CATEGORICAS:
        p, q, cats = _distribuciones_categoricas(ref[variable], act[variable])
        x = np.arange(len(cats))
        ax.bar(x - 0.2, p, 0.4, label="Histórica")
        ax.bar(x + 0.2, q, 0.4, label="Actual")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cats], rotation=20)
        ax.set_ylabel("proporción")
    else:
        a, b = ref[variable].dropna(), act[variable].dropna()
        lo, hi = np.quantile(pd.concat([a, b]), [0.01, 0.99])
        bins = np.linspace(lo, hi, 30) if hi > lo else 10
        ax.hist(a.clip(lo, hi), bins=bins, density=True, alpha=0.55, label="Histórica")
        ax.hist(b.clip(lo, hi), bins=bins, density=True, alpha=0.55, label="Actual")
        ax.set_ylabel("densidad")
    ax.set_title(f"{variable}: histórica vs actual")
    ax.legend()
    fig.tight_layout()
    return fig


def _colorear_estado(valor):
    return f"background-color: {COLORES.get(valor, 'white')}; color: white"


def app():
    import matplotlib.pyplot as plt
    import streamlit as st

    st.set_page_config(page_title="Monitoreo del modelo de crédito", layout="wide")
    st.title("📊 Monitoreo del modelo — Pago a tiempo de créditos")
    st.caption("Detección de data drift: población actual frente a la histórica de referencia.")

    # --- Parámetros ---
    st.sidebar.header("Parámetros")
    corte = st.sidebar.date_input("Fecha de corte (referencia | actual)", pd.Timestamp(FECHA_CORTE_REFERENCIA))
    periodicidad = st.sidebar.radio("Periodicidad del muestreo", ["Mensual", "Trimestral"])
    min_reg = st.sidebar.number_input("Mínimo de registros por periodo", 10, 500, MIN_REGISTROS_PERIODO)

    @st.cache_data(show_spinner="Calculando predicciones...")
    def _datos(per):
        d = construir_tabla_monitoreo()
        d["periodo"] = d["fecha_prestamo"].dt.to_period("M" if per == "Mensual" else "Q")
        return d

    try:
        df = _datos(periodicidad)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    ref, act = dividir_referencia_actual(df, corte)
    if len(act) < min_reg or len(ref) < min_reg:
        st.warning("La fecha de corte deja muy pocos registros en la referencia o en la población actual.")
        st.stop()

    tabla = calcular_drift(ref, act)
    temporal = calcular_drift_temporal(df, corte, int(min_reg))

    # --- Indicadores ---
    n_rojo, n_amar = int((tabla["estado"] == "rojo").sum()), int((tabla["estado"] == "amarillo").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros de referencia", f"{len(ref):,}")
    c2.metric("Registros actuales", f"{len(act):,}")
    c3.metric("🔴 Variables en rojo", n_rojo)
    c4.metric("🟡 Variables en amarillo", n_amar)

    # --- Alertas y recomendaciones ---
    st.subheader("🚨 Alertas y recomendaciones")
    for nivel, msg in generar_alertas(tabla):
        {"rojo": st.error, "amarillo": st.warning, "verde": st.success, "info": st.info}[nivel](msg)

    tab1, tab2, tab3, tab4 = st.tabs(["Métricas por variable", "Histórica vs actual",
                                      "Evolución temporal", "Desempeño del modelo"])

    with tab1:
        st.markdown("**Semáforo según PSI:** 🟢 < 0.10 · 🟡 0.10–0.25 · 🔴 ≥ 0.25")
        st.dataframe(tabla.style.map(_colorear_estado, subset=["estado"]).format(
            {"ks_stat": "{:.3f}", "ks_pvalue": "{:.4f}", "psi": "{:.3f}", "js": "{:.3f}", "chi2_pvalue": "{:.4f}"},
            na_rep="—"), width="stretch", hide_index=True)
        st.markdown("**Barras de riesgo (PSI por variable)**")
        graf = tabla.set_index("variable")["psi"].sort_values()
        fig, ax = plt.subplots(figsize=(8, max(3, len(graf) * 0.28)))
        ax.barh(graf.index, graf.values, color=[COLORES[clasificar_psi(v)] for v in graf.values])
        ax.axvline(PSI_AMARILLO, ls="--", color="gray", lw=.8)
        ax.axvline(PSI_ROJO, ls="--", color="black", lw=.8)
        ax.set_xlabel("PSI")
        fig.tight_layout()
        st.pyplot(fig)

    with tab2:
        variable = st.selectbox("Variable", tabla["variable"].tolist())
        fila = tabla[tabla["variable"] == variable].iloc[0]
        st.markdown(f"Estado: {ICONOS[fila['estado']]} **{fila['estado']}** · PSI = {fila['psi']:.3f} · "
                    f"Jensen-Shannon = {fila['js']:.3f}")
        st.pyplot(_grafico_comparacion(ref, act, variable))

    with tab3:
        if temporal.empty:
            st.info("No hay periodos con suficientes registros.")
        else:
            st.markdown("**PSI por periodo (columnas) y variable (filas)**")
            pivote = temporal.pivot(index="variable", columns="periodo", values="psi")
            st.dataframe(pivote.style.background_gradient(cmap="RdYlGn_r", vmin=0, vmax=0.4).format("{:.2f}"),
                         width="stretch")
            top = pivote.max(axis=1).sort_values(ascending=False).head(5).index.tolist()
            elegidas = st.multiselect("Variables a graficar", pivote.index.tolist(), default=top)
            if elegidas:
                st.line_chart(pivote.loc[elegidas].T)
            st.markdown("**Tendencia detectada**")
            tend = pd.DataFrame({"variable": pivote.index,
                                 "tendencia": [detectar_tendencia(pivote.loc[v]) for v in pivote.index],
                                 "PSI máximo": pivote.max(axis=1).values}).sort_values("PSI máximo", ascending=False)
            st.dataframe(tend, hide_index=True, width="stretch")
            n_periodo = temporal.groupby("periodo")["n_registros"].first()
            st.caption("Registros por periodo: " + ", ".join(f"{p}: {n}" for p, n in n_periodo.items()))

    with tab4:
        des = desempeno_por_periodo(df, int(min_reg))
        st.caption("Los créditos más recientes pueden tener etiquetas incompletas: interpretar con cautela.")
        if des.empty:
            st.info("No hay periodos con suficientes casos de ambas clases.")
        else:
            st.dataframe(des.round(3), hide_index=True, width="stretch")
            st.line_chart(des.set_index("periodo")[["roc_auc", "recall_0"]])


def _en_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


if _en_streamlit():
    app()
elif __name__ == "__main__":
    ejecutar_monitoreo()