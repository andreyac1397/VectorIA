# VectorIA: datos base

> **Aviso:** los casos y el clima de los archivos `*_real.csv` y de `vectoria_acciones.csv` son reales (OPS/PAHO y NASA POWER). `vectoria_dengue_semanal.csv` es **sintético** (versión opcional por provincia) y **no son cifras oficiales**.

## Archivos

| Archivo | Uso | Filas |
|---|---|---|
| `vectoria_dengue_nacional_real.csv` | **Real.** Módulo 2 (LSTM): casos + clima, serie nacional | 574 semanas (2014–2024), sin faltantes |
| `vectoria_casos_cantonal_anual_real.csv` | **Real.** Mapa de calor del dashboard | 422 filas (71 cantones, 2013–2018) |
| `vectoria_dengue_semanal.csv` | Sintético. Módulo 2, versión por provincia | 4,382 (7 provincias × 626 semanas, 2014–2025) |
| `vectoria_acciones.csv` | Módulo 3 (ANN): base real, etiquetas por reglas | 2,060 (412 semanas × 5 escenarios, 2017–2024) |
| `generar_acciones_real.py` | Regenera `vectoria_acciones.csv` (semilla 42) | — |
| `generar_sintetico.py` | Regenera el CSV sintético provincial (semilla 42) | — |
| `construir_dataset_real.py` | Casos reales de OpenDengue + clima real de NASA POWER | — |

## Datos reales (OPS/PAHO vía OpenDengue)

- **Origen:** extracto de OpenDengue filtrado a la fuente PAHO, procesado con `construir_dataset_real.py --solo-casos`.
- **`vectoria_dengue_nacional_real.csv`:** casos semanales nacionales más clima semanal, con semanas que inician en domingo.
  - Los totales anuales coinciden con los registros anuales de la OPS; por ejemplo, 23,319 en 2016 y 31,259 en 2024.
  - Años de brote: 2016, 2023 y 2024. El pico máximo es de 1,749 casos en la semana 46 de 2023.
  - **Clima:** NASA POWER diario (lluvia corregida, temperatura máxima y mínima a 2 m, humedad relativa) en las 7 cabeceras de provincia, promediado y ponderado por población.
  - Las semanas de lluvia extrema coinciden con eventos conocidos: el huracán Nate (oct. 2017), Eta (nov. 2020) y la tormenta Sara (nov. 2024). Sirven para hablar de outliers en el EDA.
  - **Hallazgo para el EDA:** la temperatura mínima con unas 6 semanas de rezago es el mejor predictor (r ≈ 0.71 con log(casos)). La lluvia aporta menos (r ≈ 0.34 con 0–2 semanas de rezago).
  - Para regenerarlo: `python construir_dataset_real.py --opendengue filtered_data_PAHO.csv`
- **`vectoria_casos_cantonal_anual_real.csv`:** casos anuales por cantón.
  - **Ojo:** la suma de los cantones es mayor que el total nacional (en 2013, 81,942 frente a 49,993). Probablemente se deba a una diferencia de definición de caso en la fuente.
  - Úselo para comparar cantones entre sí (proporciones y ranking en el mapa de calor), no como cifra absoluta.
  - Discutir esta inconsistencia es un buen ejercicio de calidad de datos para el informe técnico.

## `vectoria_dengue_semanal.csv`

| Columna | Descripción |
|---|---|
| `fecha_inicio_semana` | Domingo en que inicia la semana (AAAA-MM-DD) |
| `anio`, `semana_epi` | Año y semana epidemiológica |
| `provincia`, `region_climatica` | Provincia y región (Valle Central, Pacífico Norte, Pacífico Central/Sur, Caribe) |
| `poblacion` | Población aproximada de la provincia |
| `casos_dengue` | Casos reportados en la semana (**variable objetivo**) |
| `tasa_100k` | Casos por cada 100,000 habitantes |
| `lluvia_mm` | Lluvia acumulada de la semana |
| `temp_max_c`, `temp_min_c` | Promedio semanal de las temperaturas máxima y mínima diarias |
| `humedad_rel_pct` | Humedad relativa promedio |

**Qué deben descubrir los estudiantes en el EDA:**

- La lluvia influye en los casos con un rezago de unas 4 semanas. La correlación es máxima en ese punto y desaparece hacia las 12 semanas, así que hay que crear features con lags.
- El pico estacional de casos ocurre en julio y agosto, después del inicio de la época lluviosa en mayo.
- Hay años epidémicos (2019, 2023, 2024) y años bajos (2017, 2021).
- Limón tiene un régimen de lluvia caribeño, distinto al del resto del país.
- Cartago casi no tiene casos por su altitud.
- Cerca del 0.7 % de los valores de lluvia, humedad y temperatura máxima faltan a propósito, para practicar imputación.

**Tarea sugerida:** predecir `casos_dengue` de las próximas 4 semanas por provincia, con ventanas de entrada de 12 a 30 semanas. Comparar contra el baseline estacional (la misma semana del año anterior).

## `vectoria_acciones.csv`

Features: `casos_pred_4sem`, `casos_umbral_endemico`, `lluvia_14d_mm`, `temp_max`, `temp_min`, `humedad`, `aegypti_detectado`, `semana_epi`.
Etiqueta: `recomendacion`.

**Qué es real y qué es simulado:**

- **Real:** casos, lluvia, temperaturas y humedad (de `vectoria_dengue_nacional_real.csv`), y `casos_umbral_endemico`, que es el percentil 75 de los casos de esas mismas 4 semanas en los 3 a 5 años previos (canal endémico simplificado).
- **Simulado:**
  - `casos_pred_4sem`: los casos reales de las próximas 4 semanas con un error de alrededor del 20 %, como si vinieran del LSTM.
  - `aegypti_detectado`: simula la salida de la CNN; es más probable con calor y lluvia.
  - La etiqueta, generada con reglas.
- **Escenarios:** la serie nacional da solo 412 semanas útiles, así que cada semana aparece en 5 escenarios (`escenario` 1–5) con distinto error de predicción y distinta detección de Aedes.

**Reglas de etiquetado**, aplicadas en orden de prioridad:

1. `casos_pred_4sem` > 2 × umbral → `alerta_sanitaria`
2. `casos_pred_4sem` > umbral y `aegypti_detectado` = 1 → `fumigacion_focalizada`
3. `lluvia_14d_mm` > 80 y `temp_max` > 27 → `eliminacion_criaderos`
4. En otro caso → `monitoreo_normal`

Un 3 % de las etiquetas tiene ruido aleatorio, para que el modelo no alcance un 100 % trivial.

**Distribución de clases:** monitoreo 1,213 · alerta 379 · eliminación 321 · fumigación 147. Está desbalanceada, así que conviene usar class weights y reportar precision y recall por clase.

**Cómo separar train y test (importante):**

- **No usar un split aleatorio por fila.** Los 5 escenarios de una misma semana son casi idénticos, así que se filtraría información entre train y test.
- **Recomendado:** agrupar por semana con `GroupShuffleSplit` o `GroupKFold` de scikit-learn, usando `groups=fecha_inicio_semana`.
- **Alternativa realista:** split temporal, por ejemplo train 2017–2022 y test 2023–2024. Ojo: las alertas se concentran en las epidemias reales de 2023–2024, así que el test tiene una distribución distinta. Es un buen tema para discutir en el informe técnico.

