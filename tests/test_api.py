"""Pruebas de la API de riesgo de crédito (FastAPI)."""

CREDITO = {
    "tipo_credito": 4, "capital_prestado": 1852560, "plazo_meses": 12, "edad_cliente": 32,
    "tipo_laboral": "Empleado", "salario_cliente": 3500000, "total_otros_prestamos": 1000000,
    "cuota_pactada": 128650, "cant_creditosvigentes": 0, "huella_consulta": 2,
    "puntaje_datacredito": 795, "saldo_mora": 0, "saldo_total": 0,
    "creditos_sectorFinanciero": 0, "creditos_sectorCooperativo": 0, "creditos_sectorReal": 0,
    "promedio_ingresos_datacredito": 916148, "tendencia_ingresos": "Creciente",
}


def test_health_responde_ok(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200 and r.json()["estado"] == "ok"


def test_predict_devuelve_probabilidad_y_riesgo(cliente):
    r = cliente.post("/predict", json={"registros": [CREDITO]})
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["n_registros"] == 1
    p = cuerpo["predicciones"][0]
    assert 0 <= p["prob_no_pago"] <= 1
    assert p["nivel_riesgo"] in {"bajo", "medio", "alto"}


def test_predict_rechaza_edad_invalida(cliente):
    malo = {**CREDITO, "edad_cliente": 5}
    assert cliente.post("/predict", json={"registros": [malo]}).status_code == 422


def test_predict_rechaza_lista_vacia(cliente):
    assert cliente.post("/predict", json={"registros": []}).status_code == 422
