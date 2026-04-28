from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo entrenado y el escalador
print("Cargando modelo y escalador...")
model = joblib.load('modelo_entrenado.pkl')
scaler = joblib.load('scaler.pkl')
print("¡Modelo cargado exitosamente!")

# Ruta principal - muestra el formulario
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para predicción individual (recibe datos del formulario)
@app.route('/predict_individual', methods=['POST'])
def predict_individual():
    try:
        # Obtener los datos del formulario HTML
        antiguedad = float(request.form['antiguedad_meses'])
        asistencia = float(request.form['asistencia_entrenamiento'])
        servicios = float(request.form['servicios_mes'])
        edad = float(request.form['edad'])
        estudiante = float(request.form['es_estudiante'])
        sociales = float(request.form['actividades_sociales'])
        
        # Crear un array con los datos
        features = np.array([[antiguedad, asistencia, servicios, edad, estudiante, sociales]])
        
        # Escalar los datos (igual que se hizo en el entrenamiento)
        features_scaled = scaler.transform(features)
        
        # Hacer la predicción
        probabilidad = model.predict_proba(features_scaled)[0][1]  # Probabilidad de renuncia
        prediccion = model.predict(features_scaled)[0]  # 0 = No renuncia, 1 = Renuncia
        
        # Determinar el nivel de riesgo
        if probabilidad >= 0.7:
            riesgo = "ALTO"
            color = "red"
            recomendacion = "Se recomienda entrevista personal inmediata y plan de retención."
        elif probabilidad >= 0.4:
            riesgo = "MEDIO"
            color = "orange"
            recomendacion = "Realizar seguimiento mensual y aumentar actividades de integración."
        else:
            riesgo = "BAJO"
            color = "green"
            recomendacion = "Voluntario comprometido. Mantener seguimiento normal."
        
        # Mostrar el resultado
        return render_template('resultado.html', 
                             probabilidad=round(probabilidad * 100, 2),
                             riesgo=riesgo,
                             color=color,
                             recomendacion=recomendacion,
                             prediccion=prediccion)
    
    except Exception as e:
        return f"Error en la predicción: {str(e)}"

# Ruta para predicción masiva (subir archivo CSV)
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        # Obtener el archivo subido
        file = request.files['file']
        
        # Leer el CSV
        df = pd.read_csv(file)
        
        # Verificar que tenga las columnas necesarias
        columnas_requeridas = ['antiguedad_meses', 'asistencia_entrenamiento', 
                               'servicios_mes', 'edad', 'es_estudiante', 'actividades_sociales']
        
        for col in columnas_requeridas:
            if col not in df.columns:
                return f"Error: El archivo debe tener la columna '{col}'"
        
        # Escalar los datos
        X = scaler.transform(df[columnas_requeridas])
        
        # Hacer predicciones
        df['prob_renuncia'] = model.predict_proba(X)[:, 1]
        df['prediccion'] = model.predict(X)
        df['riesgo'] = df['prob_renuncia'].apply(lambda x: 'Alto' if x > 0.6 else 'Medio' if x > 0.3 else 'Bajo')
        
        # Crear un resumen
        total = len(df)
        alto_riesgo = len(df[df['riesgo'] == 'Alto'])
        medio_riesgo = len(df[df['riesgo'] == 'Medio'])
        bajo_riesgo = len(df[df['riesgo'] == 'Bajo'])
        
        # Convertir a HTML para mostrar
        tabla_html = df.to_html(classes='table table-striped')
        
        return render_template('batch_resultado.html',
                             total=total,
                             alto_riesgo=alto_riesgo,
                             medio_riesgo=medio_riesgo,
                             bajo_riesgo=bajo_riesgo,
                             tabla=tabla_html)
    
    except Exception as e:
        return f"Error en el procesamiento masivo: {str(e)}"

# Ruta para API REST (para integración con otros sistemas)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Obtener datos JSON
        data = request.get_json()
        
        # Extraer características
        features = np.array([[data['antiguedad_meses'], 
                              data['asistencia_entrenamiento'],
                              data['servicios_mes'], 
                              data['edad'],
                              data['es_estudiante'], 
                              data['actividades_sociales']]])
        
        features_scaled = scaler.transform(features)
        probabilidad = float(model.predict_proba(features_scaled)[0][1])
        
        # Devolver resultado en JSON
        return jsonify({
            'probabilidad_renuncia': round(probabilidad * 100, 2),
            'riesgo': 'Alto' if probabilidad > 0.6 else 'Medio' if probabilidad > 0.3 else 'Bajo',
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

# Ejecutar la aplicación
if __name__ == '__main__':
    print("=" * 50)
    print("Iniciando aplicación web de predicción - Bomberos")
    print("=" * 50)
    print("La aplicación estará disponible en: http://127.0.0.1:5000")
    print("Presiona Ctrl+C para detener el servidor")
    print("=" * 50)
    app.run(debug=True)