
# 🦟 VectorIA

Proyecto de inteligencia artificial para la identificación de mosquitos, predicción de casos de dengue y generación de recomendaciones preventivas.

VectorIA integra tres modelos de inteligencia artificial:

- **CNN:** identificación de especies de mosquitos mediante imágenes.
- **LSTM:** predicción de casos de dengue para las próximas cuatro semanas.
- **ANN:** generación de recomendaciones preventivas a partir de predicciones y variables epidemiológicas.

La plataforma cuenta con una interfaz desarrollada en **Streamlit** y una API construida con **FastAPI**.

---

## 📂 Datasets CNN — Identificación de mosquitos

| # | Dataset | Fuente | Descarga |
|---|---|---|---|
| 01 | Image Dataset of Aedes and Culex Mosquito Species | IEEE DataPort | [Descargar](https://ieee-dataport.org/open-access/image-dataset-aedes-and-culex-mosquito-species) |
| 02 | Dataset of Vector Mosquito Images | Mendeley Data | [Descargar](https://data.mendeley.com/datasets/88s6fvgg2p/4) |
| 03 | AI Mosquito Alert Challenge Dataset 2023 | Zenodo | [Descargar](https://zenodo.org/records/15063886) |
| 04 | Aedes Mosquitos | Kaggle | [Descargar](https://www.kaggle.com/datasets/pradeepisawasan/aedes-mosquitos) |

### 🔬 Dataset adicional — Bonus

| Dataset | Fuente | Descarga |
|---|---|---|
| MLMI-2024 — Mosquito Larvae Microscopic Images | Mendeley Data | [Descargar](https://data.mendeley.com/datasets/pgby6jmtp4/1) |

---

## 📊 Datasets LSTM, ANN y visualización

**Fuente:** CUC Virtual — Material proporcionado por el profesor.

| Módulo | Dataset |
|---|---|
| LSTM | [vectoria_dengue_nacional_real.csv](https://www.cucvirtual.ac.cr/pluginfile.php/318201/mod_folder/content/0/VectorIA-datos/vectoria_dengue_nacional_real.csv?forcedownload=1) |
| ANN | [vectoria_acciones.csv](https://www.cucvirtual.ac.cr/pluginfile.php/318201/mod_folder/content/0/VectorIA-datos/vectoria_acciones.csv?forcedownload=1) |
| Mapa de calor | [vectoria_casos_cantonal_anual_real.csv](https://www.cucvirtual.ac.cr/pluginfile.php/318201/mod_folder/content/0/VectorIA-datos/vectoria_casos_cantonal_anual_real.csv?forcedownload=1) |
| Bonus | [vectoria_dengue_semanal.csv](https://www.cucvirtual.ac.cr/pluginfile.php/318201/mod_folder/content/0/VectorIA-datos/vectoria_dengue_semanal.csv?forcedownload=1) |

**Documentación:** [LEEME_datos.md](https://www.cucvirtual.ac.cr/pluginfile.php/318201/mod_folder/content/0/VectorIA-datos/LEEME_datos.md?forcedownload=1)

Los datasets deben colocarse en `data/raw/` según las necesidades de cada módulo.

---

# 🚀 Ejecutar VectorIA

Para utilizar la interfaz de Streamlit y los modelos de inteligencia artificial, es necesario ejecutar **FastAPI y Streamlit simultáneamente**.

Todos los comandos deben ejecutarse desde la carpeta principal del proyecto `VectorIA`.

## 1. Instalar dependencias

Se recomienda utilizar un entorno virtual de Python.

```bash
python -m venv .venv
```

Activar el entorno virtual en Windows:

```bat
.venv\Scripts\activate
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

## 2. Iniciar FastAPI

Abrir una terminal y ejecutar:

```bash
python -m uvicorn api.main:app --reload
```

La API estará disponible en:

http://127.0.0.1:8000

Documentación interactiva de los endpoints:

http://127.0.0.1:8000/docs

FastAPI permite ejecutar los modelos y consultar sus resultados desde Streamlit.

## 3. Iniciar Streamlit

Sin cerrar la terminal de FastAPI, abrir **una segunda terminal** desde la carpeta principal de VectorIA.

Ejecutar:

```bash
python -m streamlit run app/Home.py
```

Streamlit abrirá la aplicación en el navegador, normalmente en:

http://localhost:8501

## 4. Utilizar la aplicación

Desde el laboratorio de modelos de Streamlit se pueden consultar los módulos:

**CNN — Identificación de mosquitos**

Permite cargar una fotografía para identificar la especie mediante el modelo CNN, una vez completada su integración con FastAPI.

**LSTM — Predicción de dengue**

Permite seleccionar una fecha y obtener la predicción nacional de casos de dengue para las próximas cuatro semanas.

**ANN — Recomendaciones preventivas**

Utiliza las predicciones del LSTM y variables históricas de dengue y clima para generar recomendaciones.

El flujo de integración es:

1. El usuario selecciona una fecha en LSTM.
2. El LSTM genera las predicciones de cuatro semanas.
3. Streamlit solicita automáticamente las variables históricas al servicio de ANN.
4. La ANN recibe los casos pronosticados, las variables preparadas y la detección de Aedes.
5. El usuario genera la recomendación preventiva.

**Importante:** FastAPI debe permanecer ejecutándose mientras se utiliza Streamlit.

---

# 🧠 Módulo CNN — VectorIA

## Propósito

Clasificar fotografías de mosquitos adultos en:

- Aedes aegypti
- Aedes albopictus
- Culex

## Uso desde Python

```python
from src.predict_mosquito import predecir_mosquito

resultado = predecir_mosquito("foto.jpg")

print(resultado["especie"])
print(resultado["score"])
print(resultado["probabilidades"])
```

## Formato de entrada

Formatos recomendados:

- JPG
- JPEG
- PNG

La imagen se convierte automáticamente a RGB y se redimensiona internamente a **128 × 128 píxeles**, conservando la proporción mediante padding.

Para obtener mejores resultados, el mosquito debe ser claramente visible y ocupar una parte relevante de la fotografía.

## Formato de salida

La función retorna un diccionario con:

- `class_id`: identificador numérico de la clase.
- `especie`: especie predicha.
- `score`: valor Softmax de la clase seleccionada.
- `probabilidades`: scores de las tres clases.
- `modelo`: nombre del modelo utilizado.
- `input_size`: tamaño interno utilizado por la CNN.

## Modelos

Los modelos entrenados se encuentran en:

```text
models/
├── cnn_mosquitos.keras
├── cnn_mosquitos_config.json
└── mobilenetv2_mosquitos.keras
```

---


VectorIA no diagnostica dengue en personas ni sustituye los criterios o decisiones de las autoridades sanitarias.
