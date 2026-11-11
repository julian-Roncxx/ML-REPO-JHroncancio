// Pipeline de integración continua del proyecto de scoring de crédito.
// Se dispara al hacer merge/push a developer, certification o master.
// Funciona en agentes Windows (bat) y Linux (sh).

def cmd(String comando) {
    if (isUnix()) { sh comando } else { bat comando }
}

pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    parameters {
        booleanParam(name: 'CONSTRUIR_DOCKER', defaultValue: false, description: 'Construir la imagen Docker (requiere Docker en el agente)')
        booleanParam(name: 'EJECUTAR_SONAR', defaultValue: false, description: 'Analizar con SonarCloud (requiere credencial sonar-token y sonar-scanner)')
    }

    environment {
        VENV = 'venv_ci'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                echo "Rama: ${env.BRANCH_NAME ?: 'desconocida'}"
            }
        }

        stage('Instalar dependencias') {
            steps {
                script {
                    if (isUnix()) {
                        sh 'python3 -m venv $VENV && $VENV/bin/pip install --upgrade pip && $VENV/bin/pip install -r requirements.txt -r requirements-dev.txt'
                    } else {
                        bat 'python -m venv %VENV% && %VENV%\\Scripts\\python -m pip install --upgrade pip && %VENV%\\Scripts\\pip install -r requirements.txt -r requirements-dev.txt'
                    }
                }
            }
        }

        stage('Validar estructura') {
            steps {
                script {
                    cmd(isUnix() ? '$VENV/bin/python -m pytest tests/test_estructura.py -q'
                                 : '%VENV%\\Scripts\\python -m pytest tests/test_estructura.py -q')
                }
            }
        }

        stage('Pruebas y cobertura') {
            steps {
                script {
                    cmd(isUnix() ? '$VENV/bin/python -m pytest tests -q --junitxml=resultados.xml --cov=mlops_pipeline --cov-report=xml'
                                 : '%VENV%\\Scripts\\python -m pytest tests -q --junitxml=resultados.xml --cov=mlops_pipeline --cov-report=xml')
                }
            }
            post {
                always { junit allowEmptyResults: true, testResults: 'resultados.xml' }
            }
        }

        stage('SonarCloud') {
            when { expression { return params.EJECUTAR_SONAR } }
            steps {
                withCredentials([string(credentialsId: 'sonar-token', variable: 'SONAR_TOKEN')]) {
                    script {
                        cmd(isUnix() ? 'sonar-scanner -Dsonar.token=$SONAR_TOKEN'
                                     : 'sonar-scanner -Dsonar.token=%SONAR_TOKEN%')
                    }
                }
            }
        }

        stage('Entrenar y evaluar') {
            steps {
                script {
                    cmd(isUnix() ? '$VENV/bin/python mlops_pipeline/model_training_evaluation.py'
                                 : '%VENV%\\Scripts\\python mlops_pipeline\\model_training_evaluation.py')
                }
            }
            post {
                success { archiveArtifacts artifacts: 'artifacts/**', allowEmptyArchive: true }
            }
        }

        stage('Construir imagen Docker') {
            when { expression { return params.CONSTRUIR_DOCKER } }
            steps {
                script { cmd('docker build -t riesgo-credito:latest .') }
            }
        }
    }

    post {
        success { echo 'Pipeline exitoso: estructura válida, pruebas aprobadas y modelo entrenado.' }
        failure { echo 'El pipeline falló: revisa la etapa marcada en rojo.' }
        cleanup { cleanWs(deleteDirs: true, patterns: [[pattern: 'venv_ci/**', type: 'INCLUDE']]) }
    }
}
