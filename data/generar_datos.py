import pandas as pd
import numpy as np

# Fijar una semilla para que los resultados sean reproducibles
np.random.seed(42)

# Número de voluntarios a generar
n = 1000

# Generar datos simulados
datos = {
    'antiguedad_meses': np.random.randint(1, 120, n),           # entre 1 mes y 10 años
    'asistencia_entrenamiento': np.random.uniform(0.3, 1.0, n), # entre 30% y 100%
    'servicios_mes': np.random.poisson(3, n),                   # promedio 3 servicios por mes
    'edad': np.random.randint(18, 55, n),                       # entre 18 y 55 años
    'es_estudiante': np.random.choice([0, 1], n, p=[0.6, 0.4]), # 40% son estudiantes
    'actividades_sociales': np.random.poisson(1.5, n),          # promedio 1.5 act sociales/mes
}

# Crear DataFrame
df = pd.DataFrame(datos)

# Regla lógica para determinar quién renuncia (simulado)
# Un voluntario renuncia si cumple AL MENOS 2 de estas 3 condiciones:
# - Asistencia a entrenamientos menor al 60%
# - Menos de 2 servicios por mes
# - Antigüedad menor a 24 meses (2 años)
condicion_baja_asistencia = df['asistencia_entrenamiento'] < 0.6
condicion_bajo_servicios = df['servicios_mes'] < 2
condicion_baja_antiguedad = df['antiguedad_meses'] < 24

# Sumar cuántas condiciones cumple (True = 1, False = 0)
df['renuncia'] = (condicion_baja_asistencia.astype(int) + 
                  condicion_bajo_servicios.astype(int) + 
                  condicion_baja_antiguedad.astype(int)) >= 2

# Convertir renuncia a entero (0 o 1)
df['renuncia'] = df['renuncia'].astype(int)

# Guardar el archivo CSV
df.to_csv('data/voluntarios.csv', index=False)

print("¡Datos generados exitosamente!")
print(f"Total de voluntarios generados: {len(df)}")
print(f"Voluntarios que renunciaron (simulado): {df['renuncia'].sum()}")
print(f"Tasa de renuncia simulada: {df['renuncia'].mean() * 100:.1f}%")
print("\nLas primeras 5 filas del archivo generado:")
print(df.head())