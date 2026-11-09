import pandas as pd

df = pd.read_csv("data/raw/vectoria_acciones.csv")


def regla_profesor(f):
    if f["casos_pred_4sem"] > 2 * f["casos_umbral_endemico"]:
        return "alerta_sanitaria"

    if (
        f["casos_pred_4sem"] > f["casos_umbral_endemico"]
        and f["aegypti_detectado"] == 1
    ):
        return "fumigacion_focalizada"

    if f["lluvia_14d_mm"] > 80 and f["temp_max"] > 27:
        return "eliminacion_criaderos"

    return "monitoreo_normal"


df["regla_esperada"] = df.apply(regla_profesor, axis=1)

errores = df[
    df["recomendacion"].astype(str).str.strip()
    != df["regla_esperada"]
]

print("Total de registros:", len(df))
print("Etiquetas incorrectas:", len(errores))

print("\nClases originales:")
print(df["recomendacion"].value_counts())

print("\nClases según el profesor:")
print(df["regla_esperada"].value_counts())

print("\nEjemplos de diferencias:")
print(errores.head(10).to_string(index=False))