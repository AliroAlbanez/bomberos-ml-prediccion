import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler

# Mostrar mensaje de inicio
print("=" * 50)
print("Entrenando modelo de predicción de renuncia - Bomberos")
print("=" * 50)

# 1. Cargar los datos
print("\n1. Cargando datos...")
df = pd.read_csv('data/voluntarios.csv')
print(f"   Total de registros: {len(df)}")
print(f"   Columnas: {list(df.columns)}")

# 2. Separar características (X) y variable objetivo (y)
print("\n2. Separando características y variable objetivo...")
X = df.drop('renuncia', axis=1)  # Características: antiguedad, asistencia, etc.
y = df['renuncia']                # Variable objetivo: renuncia (0 o 1)
print(f"   Características (X): {X.shape[1]} columnas")
print(f"   Variable objetivo (y): {y.value_counts().to_dict()}")

# 3. Escalar los datos (normalizar para mejorar el modelo)
print("\n3. Escalando datos...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   Escalado completado")

# 4. Dividir en entrenamiento (80%) y prueba (20%)
print("\n4. Dividiendo datos en entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"   Entrenamiento: {len(X_train)} registros")
print(f"   Prueba: {len(X_test)} registros")

# 5. Entrenar el modelo Random Forest
print("\n5. Entrenando modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("   Entrenamiento completado")

# 6. Hacer predicciones sobre los datos de prueba
print("\n6. Evaluando modelo...")
y_pred = model.predict(X_test)

# 7. Calcular métricas de rendimiento
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# Calcular matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Extraer valores
VP = cm[1, 1]  # Verdaderos Positivos
VN = cm[0, 0]  # Verdaderos Negativos
FP = cm[0, 1]  # Falsos Positivos
FN = cm[1, 0]  # Falsos Negativos

print("\n" + "=" * 50)
print("RESULTADOS DE EVALUACIÓN DEL MODELO")
print("=" * 50)
print(f"✓ Accuracy (Precisión global): {accuracy * 100:.2f}%")
print(f"✓ Precision (Precisión positiva): {precision * 100:.2f}%")
print(f"✓ Recall (Sensibilidad): {recall * 100:.2f}%")
print(f"✓ F1-Score (Balance): {f1 * 100:.2f}%")
print("=" * 50)
print("\n" + "=" * 50)
print("MATRIZ DE CONFUSIÓN")
print("=" * 50)
print("                 Predicho")
print("              No renuncia  Renuncia")
print(f"Real No renuncia     {VN}         {FP}")
print(f"Real Renuncia        {FN}         {VP}")
print("")
print(f"✓ Verdaderos Positivos (aciertos en renuncia): {VP}")
print(f"✓ Verdaderos Negativos (aciertos en no renuncia): {VN}")
print(f"✗ Falsos Positivos (falsas alarmas): {FP}")
print(f"✗ Falsos Negativos (errores críticos): {FN}")
print("\n" + "=" * 50)
# 8. Guardar el modelo y el escalador para usarlos en la web
print("\n8. Guardando modelo y escalador...")
joblib.dump(model, 'modelo_entrenado.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("   ✓ modelo_entrenado.pkl guardado")
print("   ✓ scaler.pkl guardado")

print("\n" + "=" * 50)
print("¡ENTRENAMIENTO COMPLETADO CON ÉXITO!")
print("=" * 50)