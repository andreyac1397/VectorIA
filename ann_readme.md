# Modelo de Red Neuronal Artificial (ANN) - Clasificación de Acciones Preventivas

Este módulo contiene la implementación de una Red Neuronal Artificial (ANN) construida con TensorFlow y Keras para la clasificación multiclase de acciones de prevención frente al dengue.

El modelo toma como entrada variables epidemiológicas y ambientales para predecir la categoría de intervención necesaria y facilitar la toma de decisiones en la interfaz de usuario.

---

## 1. Arquitectura y Métricas del Modelo

La red fue entrenada como un clasificador multiclase. El modelo final se guardó exitosamente en la ruta:
```text
models/ann_model.keras

Categoría,Precisión,Recall,F1-Score,Soporte (Muestras)
alerta_sanitaria,0.97,0.94,0.96,81
eliminacion_criaderos,0.93,0.90,0.92,60
fumigacion_focalizada,0.91,0.77,0.83,26
monitoreo_normal,0.95,0.99,0.97,245
Accuracy,—,—,—,412
Macro Avg,0.94,0.90,0.92,412
Weighted Avg,0.95,0.95,0.95,412

Aquí tienes el texto ordenado en formato **Markdown limpio** (con su tabla de métricas y listas estructuradas) para que solo tengas que copiarlo y pegarlo directamente en tu archivo `ANN_README.md` sin que se deforme:

```markdown
# Modelo de Red Neuronal Artificial (ANN) - Clasificación de Acciones Preventivas

Este módulo contiene la implementación de una Red Neuronal Artificial (ANN) construida con TensorFlow y Keras para la clasificación multiclase de acciones de prevención frente al dengue.

El modelo toma como entrada variables epidemiológicas y ambientales para predecir la categoría de intervención necesaria y facilitar la toma de decisiones en la interfaz de usuario.

---

## 1. Arquitectura y Métricas del Modelo

La red fue entrenada como un clasificador multiclase. El modelo final se guardó exitosamente en la ruta:
```text
models/ann_model.keras

```

### Resultados de Evaluación

El modelo alcanzó una precisión general (**Accuracy**) del **95%**.

| Categoría | Precisión | Recall | F1-Score | Soporte (Muestras) |
| --- | --- | --- | --- | --- |
| **alerta_sanitaria** | 0.97 | 0.94 | 0.96 | 81 |
| **eliminacion_criaderos** | 0.93 | 0.90 | 0.92 | 60 |
| **fumigacion_focalizada** | 0.91 | 0.77 | 0.83 | 26 |
| **monitoreo_normal** | 0.95 | 0.99 | 0.97 | 245 |
| **Accuracy** | — | — | — | 412 |
| **Macro Avg** | 0.94 | 0.90 | 0.92 | 412 |
| **Weighted Avg** | 0.95 | 0.95 | 0.95 | 412 |

---

## 2. Recomendaciones Preventivas por Categoría de Salida

A continuación se detallan las acciones institucionales y comunitarias asociadas a cada etiqueta devuelta por la red neuronal, diseñadas para desplegarse en el módulo de alertas de la aplicación.

### A. Alerta Sanitaria

* **Nivel de Riesgo:** Alto / Emergencia local.
* **Acciones Inmediatas:**
* Despliegue de personal técnico de salud pública para inspección en terreno.
* Emisión de boletines informativos y alertas tempranas a la comunidad afectada.
* Coordinación con centros de atención primaria para el monitoreo de pacientes febriles.



### B. Eliminación de Criaderos

* **Nivel de Riesgo:** Medio-Alto / Proliferación de vectores.
* **Acciones Inmediatas:**
* Organización de jornadas comunales de recolección de recipientes y desechos sólidos (llantas, botellas, latas).
* Aplicación de larvicidas biológicos en depósitos de agua no eliminables (tanques, estañones).
* Inspección física de predios baldíos, espacios públicos y centros educativos.



### C. Fumigación Focalizada

* **Nivel de Riesgo:** Moderado-Alto / Presencia confirmada de mosquito adulto en zonas delimitadas.
* **Acciones Inmediatas:**
* Programación de ciclos de nebulización o fumigación espacial en el sector crítico identificado.
* Notificación previa a los vecinos para el resguardo de animales domésticos y cierre de accesos durante la aplicación.
* Erradicación concurrente de criaderos cercanos para evitar reinfestación.



### D. Monitoreo Normal

* **Nivel de Riesgo:** Bajo / Estado de control.
* **Acciones Inmediatas:**
* Continuidad de las labores rutinarias de vigilancia entomológica y epidemiológica.
* Difusión continua de campañas educativas sobre prevención de vectores en la comunidad.
* Mantenimiento del registro periódico de datos ambientales e incidencias.



```

```