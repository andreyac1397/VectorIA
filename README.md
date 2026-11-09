# 🦟 VectorIA

### Datasets del proyecto

Repositorio de recursos y conjuntos de datos utilizados en VectorIA.

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


# Módulo CNN — VectorIA

## Propósito

Clasificar fotografías de mosquitos adultos en:

- Aedes aegypti
- Aedes albopictus
- Culex

## Uso

from src.predict\_mosquito import predecir\_mosquito

resultado = predecir\_mosquito("foto.jpg")

print(resultado["especie"])

print(resultado["score"])

print(resultado["probabilidades"])

## Formato de entrada

Formatos recomendados:

- JPG
- JPEG
- PNG

La imagen se convierte automáticamente a RGB y se redimensiona internamente a 128 × 128 píxeles conservando la proporción mediante padding.

Para obtener mejores resultados, el mosquito debe ser claramente visible y ocupar una parte relevante de la fotografía.

## Formato de salida

La función retorna un diccionario con:

- class\_id: identificador numérico de la clase.
- especie: especie predicha.
- score: valor Softmax de la clase seleccionada.
- probabilidades: scores de las tres clases.
- modelo: nombre del modelo utilizado.
- input\_size: tamaño interno utilizado por la CNN.

## Modelo

El modelo actual se encuentra en:

models/cnn\_mosquitos.keras

## Aviso

VectorIA no diagnostica dengue en personas. La clasificación automática de imágenes es una herramienta de apoyo y puede producir clasificaciones incorrectas.


