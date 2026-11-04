"""
model_training_evaluation.py
----------------------------
Entrena y evalúa varios modelos supervisados y selecciona el mejor según tres
criterios: performance, consistency (estabilidad en validación cruzada) y
scalability (tiempo de entrenamiento y de predicción).

Variable objetivo: Pago_atiempo (1 = pagó a tiempo, 0 = no pagó).
El interés del negocio está en detectar a quienes NO pagan, por eso las métricas
principales se calculan con la clase 0 como clase de interés ("positiva").

Ejecución:  python mlops_pipeline/model_training_evaluation.py
            python mlops_pipeline/model_training_evaluation.py --con-puntaje   (solo referencia)
"""
import argparse
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # permite ejecutar sin pantalla
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, make_scorer, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ft_engineering import RANDOM_STATE, get_datasets

warnings.filterwarnings("ignore")

# Las salidas se guardan fuera de la estructura versionada (carpeta ignorada en .gitignore)
DIR_SALIDAS = Path(__file__).resolve().parent.parent / "artifacts"
CLASE_INTERES = 0  # no pagó a tiempo


# ----------------------------------------------------------------------------
# Funciones reutilizables
# ----------------------------------------------------------------------------
def summarize_classification(y_true, y_pred, y_proba_pago) -> dict:
    """Resume el desempeño de un clasificador binario.

    y_proba_pago: probabilidad de la clase 1 (pagó a tiempo).
    Las métricas *_0 se calculan para la clase 0 (no pagó), la de mayor riesgo.
    """
    p_no_pago = 1 - np.asarray(y_proba_pago)
    y_true_riesgo = (np.asarray(y_true) == CLASE_INTERES).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_riesgo, (np.asarray(y_pred) == CLASE_INTERES).astype(int)).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_0": precision_score(y_true, y_pred, pos_label=CLASE_INTERES, zero_division=0),
        "recall_0": recall_score(y_true, y_pred, pos_label=CLASE_INTERES, zero_division=0),
        "f1_0": f1_score(y_true, y_pred, pos_label=CLASE_INTERES, zero_division=0),
        "roc_auc": roc_auc_score(y_true_riesgo, p_no_pago),
        "pr_auc_0": average_precision_score(y_true_riesgo, p_no_pago),
        "tp_0": int(tp), "fp_0": int(fp), "fn_0": int(fn), "tn_0": int(tn),
    }


def build_model(nombre, estimador, preprocessor, X_train, y_train, X_test, y_test, cv=5) -> dict:
    """Arma el pipeline (preprocesador + modelo), lo entrena, lo evalúa y mide
    consistencia (validación cruzada) y escalabilidad (tiempos)."""
    from sklearn.base import clone

    pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", estimador)])

    # Performance en el conjunto de evaluación
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    pred_time = time.perf_counter() - t0

    resultado = {"modelo": nombre, **summarize_classification(y_test, y_pred, y_proba)}

    # Consistencia: variación de las métricas entre folds (solo con datos de entrenamiento)
    scoring = {
        "roc_auc": "roc_auc",
        "recall_0": make_scorer(recall_score, pos_label=CLASE_INTERES, zero_division=0),
    }
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    cvres = cross_validate(clone(pipe), X_train, y_train, cv=skf, scoring=scoring, n_jobs=1)
    resultado.update({
        "cv_roc_auc_mean": cvres["test_roc_auc"].mean(),
        "cv_roc_auc_std": cvres["test_roc_auc"].std(),
        "cv_recall_0_mean": cvres["test_recall_0"].mean(),
        "cv_recall_0_std": cvres["test_recall_0"].std(),
        "fit_time_s": fit_time,
        "pred_time_s": pred_time,
    })
    return {"resumen": resultado, "pipeline": pipe, "y_pred": y_pred, "y_proba": y_proba}


def get_models() -> dict:
    """Modelos a comparar. class_weight='balanced' compensa el desbalance 95/5."""
    return {
        "Baseline (Dummy)": DummyClassifier(strategy="prior"),
        "Regresión Logística": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=5, min_samples_leaf=50,
                                                    class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                                            class_weight="balanced", random_state=RANDOM_STATE),
    }


def seleccionar_mejor(tabla: pd.DataFrame, pesos=(0.6, 0.2, 0.2)) -> pd.DataFrame:
    """Puntaje compuesto de tres criterios, cada uno normalizado entre 0 y 1 entre los candidatos:
      - performance : promedio de ROC-AUC y PR-AUC de la clase 0 (en el conjunto de evaluación)
      - consistency : baja desviación del ROC-AUC en validación cruzada
      - scalability : menor tiempo total de entrenamiento + predicción
    El baseline sirve de referencia y no compite por ser el mejor modelo."""
    def minmax(s, mayor_es_mejor=True):
        rng = s.max() - s.min()
        n = (s - s.min()) / rng if rng > 0 else pd.Series(1.0, index=s.index)
        return n if mayor_es_mejor else 1 - n

    t = tabla.copy()
    es_cand = ~t["modelo"].str.contains("Baseline")
    c = t[es_cand]
    t.loc[es_cand, "score_performance"] = 0.5 * minmax(c["roc_auc"]) + 0.5 * minmax(c["pr_auc_0"])
    t.loc[es_cand, "score_consistency"] = minmax(c["cv_roc_auc_std"], mayor_es_mejor=False)
    t.loc[es_cand, "score_scalability"] = minmax(c["fit_time_s"] + c["pred_time_s"], mayor_es_mejor=False)
    t[["score_performance", "score_consistency", "score_scalability"]] = \
        t[["score_performance", "score_consistency", "score_scalability"]].fillna(0)
    t["score_final"] = (pesos[0] * t["score_performance"] + pesos[1] * t["score_consistency"]
                        + pesos[2] * t["score_scalability"])
    mejor = t.loc[es_cand, "score_final"].idxmax()
    t["seleccionado"] = t.index == mejor
    return t.sort_values("score_final", ascending=False)


# ----------------------------------------------------------------------------
# Gráficos comparativos
# ----------------------------------------------------------------------------
def graficar_comparacion(tabla, resultados, y_test, destino: Path):
    destino.mkdir(parents=True, exist_ok=True)
    sin_base = tabla[~tabla["modelo"].str.contains("Baseline")]

    # 1) Barras de métricas principales
    metricas = ["roc_auc", "pr_auc_0", "recall_0", "precision_0", "f1_0"]
    ax = tabla.set_index("modelo")[metricas].plot(kind="bar", figsize=(11, 5), rot=20)
    ax.set_title("Comparación de modelos (clase de interés: no pagó a tiempo)")
    ax.set_ylabel("valor"); ax.legend(loc="upper right", ncol=5, fontsize=8)
    plt.tight_layout(); plt.savefig(destino / "comparacion_metricas.png", dpi=130); plt.close()

    # 2) Curvas ROC y Precision-Recall
    y_riesgo = (np.asarray(y_test) == CLASE_INTERES).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for nombre, r in resultados.items():
        if "Baseline" in nombre:
            continue  # la línea del baseline no aporta a las curvas
        p = 1 - r["y_proba"]
        fpr, tpr, _ = roc_curve(y_riesgo, p)
        pre, rec, _ = precision_recall_curve(y_riesgo, p)
        axes[0].plot(fpr, tpr, label=f"{nombre} ({r['resumen']['roc_auc']:.2f})")
        axes[1].plot(rec, pre, label=f"{nombre} ({r['resumen']['pr_auc_0']:.2f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=.8); axes[0].set_title("Curva ROC"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[1].axhline(y_riesgo.mean(), color="k", ls="--", lw=.8); axes[1].set_title("Curva Precision-Recall (clase 0)")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    for a in axes: a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(destino / "curvas_roc_pr.png", dpi=130); plt.close()

    # 3) Consistencia y escalabilidad
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(sin_base["modelo"], sin_base["cv_roc_auc_mean"], yerr=sin_base["cv_roc_auc_std"], capsize=5)
    axes[0].set_title("Consistency: ROC-AUC en validación cruzada (media ± desv.)"); axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(sin_base["modelo"], sin_base["fit_time_s"], label="entrenamiento")
    axes[1].bar(sin_base["modelo"], sin_base["pred_time_s"], bottom=sin_base["fit_time_s"], label="predicción")
    axes[1].set_title("Scalability: tiempo (s)"); axes[1].legend(); axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.savefig(destino / "consistencia_escalabilidad.png", dpi=130); plt.close()


# ----------------------------------------------------------------------------
# Flujo principal
# ----------------------------------------------------------------------------
def main(incluir_puntaje: bool = False, guardar: bool = True):
    X_train, X_test, y_train, y_test, preprocessor = get_datasets(incluir_puntaje=incluir_puntaje)
    print(f"Entrenamiento: {X_train.shape} | Evaluación: {X_test.shape} | puntaje incluido: {incluir_puntaje}\n")

    resultados = {}
    for nombre, est in get_models().items():
        print(f"Entrenando {nombre} ...")
        resultados[nombre] = build_model(nombre, est, preprocessor, X_train, y_train, X_test, y_test)

    tabla = pd.DataFrame([r["resumen"] for r in resultados.values()])
    tabla = seleccionar_mejor(tabla)

    cols = ["modelo", "roc_auc", "pr_auc_0", "recall_0", "precision_0", "f1_0", "accuracy",
            "cv_roc_auc_std", "fit_time_s", "pred_time_s", "score_final", "seleccionado"]
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("\n=== TABLA RESUMEN DE EVALUACIÓN ===")
    print(tabla[cols].round(3).to_string(index=False))

    mejor_nombre = tabla.loc[tabla["seleccionado"], "modelo"].iloc[0]
    mejor = resultados[mejor_nombre]
    print(f"\nMejor modelo: {mejor_nombre}")

    if guardar:
        sufijo = "_con_puntaje" if incluir_puntaje else ""
        DIR_SALIDAS.mkdir(exist_ok=True)
        tabla.to_csv(DIR_SALIDAS / f"resumen_modelos{sufijo}.csv", index=False)
        graficar_comparacion(tabla, resultados, y_test, DIR_SALIDAS / f"graficos{sufijo}")
        if not incluir_puntaje:
            joblib.dump(mejor["pipeline"], DIR_SALIDAS / "mejor_modelo.joblib")
            print(f"Modelo guardado en {DIR_SALIDAS / 'mejor_modelo.joblib'}")
        print(f"Tabla y gráficos guardados en {DIR_SALIDAS}")
    return mejor["pipeline"], tabla


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--con-puntaje", action="store_true", help="incluye 'puntaje' (solo para comparar; posible fuga)")
    args = ap.parse_args()
    main(incluir_puntaje=args.con_puntaje)
