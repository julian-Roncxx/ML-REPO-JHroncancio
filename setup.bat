@echo off
REM Crea el entorno virtual e instala las dependencias del proyecto
echo [1/3] Creando entorno virtual...
python -m venv venv
if errorlevel 1 (
    echo No se pudo crear el entorno virtual. Verifica que Python este instalado.
    exit /b 1
)
echo [2/3] Instalando dependencias...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo Error instalando dependencias.
    exit /b 1
)
echo [3/3] Listo. Activa el entorno con:  venv\Scripts\activate
echo Entrenar modelos:   venv\Scripts\python.exe mlops_pipeline\model_training_evaluation.py
echo Monitoreo (app):    venv\Scripts\streamlit run mlops_pipeline\model_monitoring.py
echo API:               venv\Scripts\uvicorn model_deploy:app --app-dir mlops_pipeline --port 8000
