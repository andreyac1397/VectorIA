"""Script ejecutor del pipeline completo de validación cruzada, entrenamiento, evaluación y pronóstico (VectorIA).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.train import lstm as L

def main():
    print("=== INICIANDO PIPELINE DE MODELADO RECURRENTE (VECTORIA) ===")
    L.fijar_semilla(42)

    # 1. Carga de datos
    df = L.cargar_datos()
    print(f"Dataset cargado: {len(df)} semanas (desde {df['fecha_inicio_semana'].min().date()} hasta {df['fecha_inicio_semana'].max().date()})")

    # 2. Validación del calendario
    for row in df.itertuples():
        a, s = L.anio_semana_epi(row.fecha_inicio_semana)
        assert a == row.anio and s == row.semana_epi, f"Fallo en fecha {row.fecha_inicio_semana}"
    assert L.anio_semana_epi("2024-12-22") == (2024, 52)
    assert L.anio_semana_epi("2024-12-29") == (2025, 1)
    assert L.anio_semana_epi("2025-01-05") == (2025, 2)
    assert L.anio_semana_epi("2025-12-28") == (2026, 1)
    assert L.anio_semana_epi("2026-12-27") == (2026, 53)
    print("Validación de calendario epidemiológico: CORRECTO")

    # 3. Validación Cruzada Expansiva
    print("\n--- EJECUTANDO VALIDACIÓN CRUZADA EXPANSIVA (CANDIDATOS A-G) ---")
    df_cv_det, df_cv_res = L.validacion_cruzada_expansiva(
        df,
        candidatos=["A", "B", "C", "D", "E", "F", "G"],
        folds_val=[2018, 2019, 2021, 2022],
        semillas=(42, 43),
        epochs=300,
        batch_size=16,
        patience=40,
        lr_patience=12,
        start_from_epoch=30,
        verbose=0,
    )
    L.RUTA_PROCESADOS.mkdir(parents=True, exist_ok=True)
    df_cv_det.to_csv(L.RUTA_PROCESADOS / "cv_resultados_detalle.csv", index=False)
    df_cv_res.to_csv(L.RUTA_PROCESADOS / "cv_resultados_resumen.csv", index=False)
    print("Resumen de Validación Cruzada:")
    print(df_cv_res.to_string(index=False))

    # Selección del modelo principal (entre B, D, E, F con mayor Skill_mean)
    candidatos_validos = df_cv_res[df_cv_res["Candidato"].isin(["B", "D", "E", "F"])]
    mejor_cand_row = candidatos_validos.iloc[0]
    cand_ganador_id = mejor_cand_row["Candidato"]
    cand_ganador_nombre = mejor_cand_row["Modelo"]
    print(f"\nModelo Principal Seleccionado por CV: Candidato {cand_ganador_id} ({cand_ganador_nombre}) con Skill CV = {mejor_cand_row['Skill_mean']:.4f}")

    # 4. Preparación de datos para Train/Val/Test
    datos = L.preparar_datos(df, incluir_clima=True)
    m = datos["masks"]
    n_f = datos["X"].shape[2]

    X_tr, y_tr = datos["X"][m["train"]], datos["y"][m["train"]]
    X_va, y_va = datos["X"][m["val"]], datos["y"][m["val"]]

    # 5. Entrenamiento del Modelo Base (SimpleRNN - Candidato A)
    print("\n--- ENTRENANDO SIMPLERNN BASE (CANDIDATO A) ---")
    rnn_base = L.construir_modelo_candidato("A", n_features=n_f, seed=42)
    hist_rnn = L.entrenar(rnn_base, X_tr, y_tr, X_va, y_va, epochs=300, batch_size=16, patience=40, lr_patience=12, start_from_epoch=30, seed=42)
    epocas_opt_rnn = int(np.argmin(hist_rnn.history["val_loss"])) + 1
    print(f"SimpleRNN Base — épocas entrenadas: {len(hist_rnn.history['loss'])} | mejor época: {epocas_opt_rnn}")

    # 6. Entrenamiento del Modelo Principal Seleccionado (LSTM)
    print(f"\n--- ENTRENANDO MODELO PRINCIPAL SELECCIONADO: CANDIDATO {cand_ganador_id} ---")
    lstm_opt = L.construir_modelo_candidato(cand_ganador_id, n_features=n_f, seed=42)
    hist_lstm = L.entrenar(lstm_opt, X_tr, y_tr, X_va, y_va, epochs=300, batch_size=16, patience=40, lr_patience=12, start_from_epoch=30, seed=42)
    epocas_opt_lstm = int(np.argmin(hist_lstm.history["val_loss"])) + 1
    print(f"Modelo Principal ({cand_ganador_nombre}) — épocas entrenadas: {len(hist_lstm.history['loss'])} | mejor época: {epocas_opt_lstm}")

    # Curvas de aprendizaje
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    ejes[0].plot(hist_rnn.history["loss"], label="Entrenamiento")
    ejes[0].plot(hist_rnn.history["val_loss"], label="Validación (2022)")
    ejes[0].set(title="Curva de aprendizaje — SimpleRNN (Base)", xlabel="Época", ylabel="MSE (escala normalizada)")
    ejes[0].legend()

    ejes[1].plot(hist_lstm.history["loss"], label="Entrenamiento")
    ejes[1].plot(hist_lstm.history["val_loss"], label="Validación (2022)")
    ejes[1].set(title=f"Curva de aprendizaje — {cand_ganador_nombre} (Principal)", xlabel="Época")
    ejes[1].legend()
    plt.tight_layout()
    plt.savefig(L.RUTA_FIGURAS / "curvas_aprendizaje.png", dpi=150)
    plt.close()

    # 7. Evaluación en conjunto de prueba (2023–2024)
    print("\n--- EVALUACIÓN FINAL SOBRE EL CONJUNTO DE PRUEBA (2023–2024) ---")
    res_rnn = L.evaluar_en(datos, rnn_base, "test")
    res_lstm = L.evaluar_en(datos, lstm_opt, "test")

    real_test = L.casos_reales_ventanas(datos, m["test"])
    base_est_test, ok_est = L.baseline_estacional(datos, m["test"])
    base_per_test = L.baseline_persistencia(datos, m["test"])

    rmse_est = L.rmse(real_test[ok_est], base_est_test[ok_est])
    mae_est = L.mae(real_test[ok_est], base_est_test[ok_est])

    rmse_per = L.rmse(real_test, base_per_test)
    mae_per = L.mae(real_test, base_per_test)

    rmse_rnn = res_rnn["metricas"]["RMSE"]
    mae_rnn = res_rnn["metricas"]["MAE"]
    skill_rnn = 1.0 - (rmse_rnn / rmse_per)

    rmse_lstm = res_lstm["metricas"]["RMSE"]
    mae_lstm = res_lstm["metricas"]["MAE"]
    skill_lstm = 1.0 - (rmse_lstm / rmse_per)

    tabla_metricas_test = pd.DataFrame([
        {"Modelo": "Baseline Estacional (t-52)", "RMSE": rmse_est, "MAE": mae_est, "Skill vs Persistencia": 1.0 - (rmse_est / rmse_per)},
        {"Modelo": "Baseline Persistencia (t)", "RMSE": rmse_per, "MAE": mae_per, "Skill vs Persistencia": 0.0},
        {"Modelo": "SimpleRNN Base (A)", "RMSE": rmse_rnn, "MAE": mae_rnn, "Skill vs Persistencia": skill_rnn},
        {"Modelo": f"LSTM Principal ({cand_ganador_id}: {cand_ganador_nombre})", "RMSE": rmse_lstm, "MAE": mae_lstm, "Skill vs Persistencia": skill_lstm},
    ])
    print("Métricas en Test (2023-2024):")
    print(tabla_metricas_test.to_string(index=False))

    tabla_metricas_test.to_csv(L.RUTA_PROCESADOS / "metricas_lstm.csv", index=False)

    # Gráfico de predicción vs real
    idx_test = datos["idx_obj"][m["test"]]
    fechas_test = datos["df_feat"]["fecha_inicio_semana"].to_numpy()
    fig, ejes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for eje, h in zip(ejes, [0, 3]):
        f = fechas_test[idx_test + h]
        eje.plot(f, res_lstm["real"][:, h], "k", lw=2, label="Casos reales")
        eje.plot(f, res_lstm["pred"][:, h], color="tab:blue", label=f"LSTM ({cand_ganador_nombre})")
        eje.plot(f, res_rnn["pred"][:, h], color="tab:orange", ls="--", label="SimpleRNN Base", alpha=0.8)
        eje.plot(f[ok_est], base_est_test[ok_est, h], color="gray", ls=":", label="Baseline estacional", alpha=0.7)
        eje.set(title=f"Pronóstico a +{h + 1} semana(s)", ylabel="Casos semanales")
        eje.legend()
    ejes[1].set_xlabel("Semana pronosticada")
    plt.tight_layout()
    plt.savefig(L.RUTA_FIGURAS / "prueba_real_vs_predicho.png", dpi=150)
    plt.close()

    # 8. Guardado de modelos evaluados
    L.guardar_artefactos(rnn_base, datos["esc_x"], datos["esc_y"], "simplernn_dengue")
    L.guardar_artefactos(lstm_opt, datos["esc_x"], datos["esc_y"], "lstm_dengue")
    print("Artefactos de prueba guardados en:", L.RUTA_MODELOS)

    # 9. Reentrenamiento con toda la serie histórica y modelos finales
    print("\n--- REENTRENAMIENTO FINAL CON TODA LA SERIE (2014–2024) ---")
    datos_f = L.preparar_datos(df, ajustar_con_todo=True, incluir_clima=True)
    mf = datos_f["masks"]["train"]
    n_f_full = datos_f["X"].shape[2]

    epocas_final_rnn = max(epocas_opt_rnn, 20)
    epocas_final_lstm = max(epocas_opt_lstm, 20)

    rnn_final = L.construir_modelo_candidato("A", n_features=n_f_full, seed=42)
    L.entrenar(rnn_final, datos_f["X"][mf], datos_f["y"][mf], epochs=epocas_final_rnn, batch_size=16, seed=42)

    lstm_final = L.construir_modelo_candidato(cand_ganador_id, n_features=n_f_full, seed=42)
    L.entrenar(lstm_final, datos_f["X"][mf], datos_f["y"][mf], epochs=epocas_final_lstm, batch_size=16, seed=42)

    L.guardar_artefactos(rnn_final, datos_f["esc_x"], datos_f["esc_y"], "simplernn_dengue_final")
    L.guardar_artefactos(lstm_final, datos_f["esc_x"], datos_f["esc_y"], "lstm_dengue_final")
    print("Modelos finales guardados en:", L.RUTA_MODELOS)

    # 10. Pronóstico Recursivo 2025–2026
    print("\n--- GENERANDO PRONÓSTICOS EPIDEMIOLÓGICOS 2025–2026 ---")
    fut_rnn = L.pronosticar_futuro(rnn_final, df, datos_f["esc_x"], datos_f["esc_y"], hasta_anio=2026)
    fut_lstm = L.pronosticar_futuro(lstm_final, df, datos_f["esc_x"], datos_f["esc_y"], hasta_anio=2026)

    assert len(fut_lstm) == 105, f"Esperadas 105 filas de pronóstico, obtenidas {len(fut_lstm)}"
    assert (fut_lstm["anio"] == 2024).sum() == 0, "No debe haber filas con año 2024 en pronóstico"
    assert (fut_lstm["anio"] == 2025).sum() == 52, "Deben haber 52 semanas en 2025"
    assert (fut_lstm["anio"] == 2026).sum() == 53, "Deben haber 53 semanas en 2026"

    futuro = pd.DataFrame({
        "fecha_inicio_semana": fut_lstm["fecha_inicio_semana"],
        "anio": fut_lstm["anio"],
        "semana_epi": fut_lstm["semana_epi"],
        "casos_reales": np.nan,  # NaN estricto en pronóstico
        "pred_SimpleRNN": np.round(fut_rnn["casos_pred"].to_numpy(), 1),
        "pred_LSTM": np.round(fut_lstm["casos_pred"].to_numpy(), 1),
        "periodo": "pronostico_" + fut_lstm["anio"].astype(str),
        "clima_usado": "climatologia_historica",
        "modelo_principal": "LSTM",
    })
    # Alerta acumulada a 4 semanas
    futuro["casos_acum_4sem_LSTM"] = np.round(futuro["pred_LSTM"].rolling(4).sum().shift(-3), 1)

    evaluacion = L.tabla_evaluacion(
        datos,
        {"SimpleRNN": res_rnn["pred"], "LSTM": res_lstm["pred"]},
        modelo_principal_nombre="LSTM",
    )

    salida = pd.concat([evaluacion, futuro], ignore_index=True)
    salida.to_csv(L.RUTA_PROCESADOS / "predicciones_lstm_2025_2026.csv", index=False)

    print(f"Predicciones consolidadas guardadas: {len(salida)} registros (Evaluación: {len(evaluacion)}, Pronóstico: {len(futuro)})")

    # Asserts de integridad
    assert len(salida) == 206, f"Se esperaban 206 filas, obtenidas {len(salida)}"
    assert (salida["periodo"] == "evaluacion_test").sum() == 101
    assert (salida["periodo"] == "pronostico_2025").sum() == 52
    assert (salida["periodo"] == "pronostico_2026").sum() == 53
    assert salida["pred_SimpleRNN"].isna().sum() == 0, "No deben haber NaNs en predicciones SimpleRNN"
    assert salida["pred_LSTM"].isna().sum() == 0, "No deben haber NaNs en predicciones LSTM"
    assert salida.loc[salida["periodo"] == "evaluacion_test", "casos_reales"].isna().sum() == 0, "Evaluación no debe tener NaNs en casos_reales"
    assert salida.loc[salida["periodo"].str.startswith("pronostico"), "casos_reales"].isna().all(), "Pronóstico DEBE tener NaN en casos_reales"
    print("Todos los asserts de integridad del CSV de salida pasaron exitosamente.")

    # Gráfico oficial de pronóstico 2025-2026
    plt.figure(figsize=(15, 5))
    rec = df[df["anio"] >= 2021]
    plt.plot(rec["fecha_inicio_semana"], rec["casos_dengue"], "k", lw=2, label="Casos reales (observados, hasta 2024)")
    plt.plot(evaluacion["fecha_inicio_semana"], evaluacion["pred_LSTM"], color="tab:blue", alpha=0.7, label="LSTM (evaluación test 2023–2024)")
    plt.plot(futuro["fecha_inicio_semana"], futuro["pred_LSTM"], color="tab:red", lw=1.8, label="LSTM — pronóstico 2025–2026")
    plt.plot(futuro["fecha_inicio_semana"], futuro["pred_SimpleRNN"], color="tab:orange", ls="--", label="SimpleRNN — pronóstico 2025–2026")
    plt.axvline(df["fecha_inicio_semana"].max(), color="gray", ls="--")
    plt.axvspan(df["fecha_inicio_semana"].max(), futuro["fecha_inicio_semana"].max(), color="gray", alpha=0.08, label="Zona de pronóstico (sin datos reales)")
    plt.title("Dengue en Costa Rica: observado, evaluación y pronóstico 2025–2026")
    plt.xlabel("Fecha")
    plt.ylabel("Casos semanales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(L.RUTA_FIGURAS / "pronostico_2025_2026.png", dpi=150)
    plt.close()

    print("=== PIPELINE COMPLETADO EXITOSAMENTE ===")

if __name__ == "__main__":
    main()
