# Proyecto 2 — Cancelación de Artefactos de Movimiento en PPG mediante RLS

**Materia:** Procesamiento Adaptativo de Señales (22.46) — ITBA  
**Profesor:** Marc S. Ressl  
**Autores:** Álvaro Martín (69507), Elisa Fernández Morales (69616), Thomas Michael Spitzner (69453)

---

## Contenido de la entrega

```text
Proyecto_2_submission/
├── informe.pdf                                        # Informe final en formato IEEE (9 páginas)
├── README.md                                          # Esta guía de estructura y ejecución
└── implementation/
    ├── __init__.py                                    # Interfaz del paquete
    ├── dataset_preparation.py                         # Carga de .ts y GroupKFold por sujeto
    ├── preprocessing.py                               # Pasabanda Butterworth [0.5, 4.0] Hz y normalización
    ├── reference_construction.py                      # Matriz de regresores MISO (con dilatación opcional D)
    ├── LMS_algorithms.py                              # RLS re-arquitectado + LMS, NLMS, VS-LMS y bucket dual
    ├── fusion.py                                      # Fusión tardía ponderada por SNR cardíaca
    ├── bpm_detection.py                               # Periodograma con zero-padding y función objetivo
    ├── pipeline.py                                    # Pipeline de extremo a extremo y HeartRateEstimatorModel
    ├── optimization.py                                # Validación cruzada (5 folds), TPE (M en [6, 50]) y exportación JSON
    ├── optimized_models_config.json                   # Hiperparámetros óptimos y métricas CV/Test exportados
    ├── pipeline_optimization_and_evaluation.ipynb     # Notebook 1: Optimización CV (RLS/NLMS/VS-LMS), test y exporta config
    ├── comparacion_LMS_vs_RLS_y_SQI.ipynb             # Notebook 2: Comparación LMS vs. RLS en test y análisis SQI
    ├── analisis_filtros_adaptativos_test.ipynb        # Notebook 3: Modos de R_hat, aprendizaje en frío, estabilidad de w y ablación en test
    └── code_structure_summary.txt                     # Documentación técnica de la arquitectura
```

---

## Requisitos de datos (archivos `.ts`)

Por límite de tamaño de entrega (< 50 MB), esta carpeta **no incluye** los archivos binarios del dataset (`IEEEPPG_TRAIN.ts` ≈ 78 MB e `IEEEPPG_TEST.ts` ≈ 57 MB).

Para ejecutar cualquiera de los dos notebooks, copie `IEEEPPG_TRAIN.ts` e `IEEEPPG_TEST.ts` desde la entrega del Proyecto 1 o descárguelos de Zenodo (`https://zenodo.org/records/3902710`) y colóquelos dentro de `Proyecto_2_submission/` o de `Proyecto_2_submission/implementation/` (ambos notebooks detectan automáticamente los archivos en la raíz del proyecto, en `implementation/` o en `/content/` para Google Colab).

---

## Guía de ejecución rápida

1. **Dependencias Python:** `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`.
2. **Optimización y evaluación principal:** abrir y ejecutar `implementation/pipeline_optimization_and_evaluation.ipynb`. Optimiza desde cero **RLS**, **NLMS** y **VS-LMS** sobre el rango ampliado $M \in [6, 50]$ mediante validación cruzada de 5 *folds* por participante, evalúa sobre las 1328 ventanas de prueba y guarda automáticamente `implementation/optimized_models_config.json`.
3. **Comparación estadística, ablación y SQI:** abrir y ejecutar `implementation/comparacion_LMS_vs_RLS_y_SQI.ipynb`, que carga `optimized_models_config.json` y ejecuta el análisis completo en 9 secciones (incluyendo el compromiso de dilatación a apertura fija $D \in \{1,2,3,4\}$):
   1. Carga de datos y configuración de modelos equiparados (`RLS`, `LMS`, `NLMS`, `VS-LMS`).
   2. Evaluación global y bajo detector común (Hann 8192 y Hamming 4096) frente a la línea base del Proyecto 1.
   3. Significancia estadística mediante *bootstrap* por participante (2000 réplicas), CDF del error y comparación pareada.
   4. Dispersión BPM estimado vs. real y AAE por sujeto.
   5. Casos de estudio en el Sujeto 4 (ventanas #481, #484 y #477).
   6. Curvas de convergencia en arranque en frío y costo computacional por ventana.
   7. Ablación de componentes, análisis de condicionamiento de $\hat{\mathbf{R}}$ y sensibilidad de RLS ($N_{\text{eff}}$ y $\delta$).
   8. Índice de calidad de señal (SQI dual AND) con control aleatorio, sensibilidad de umbrales y análisis por regímenes de movimiento (paradoja de Simpson).
   9. Tabla resumen cuantitativa.
4. **Diagnósticos adaptativos y ablación estricta sobre prueba (rápido, sin reoptimizar):** abrir y ejecutar `implementation/analisis_filtros_adaptativos_test.ipynb`, que carga únicamente `optimized_models_config.json` e `IEEEPPG_TEST.ts` para generar:
   - Puntos de operación teóricos ($\mu_{\text{ef}}$, $2/L$, desajuste $\mathcal{M}$), espectro de autovalores y constantes de tiempo modales $\tau_i$, curvas de aprendizaje con arranque en frío ($\text{WS}=0$) y estabilidad de $\lVert\mathbf{w}[n]\rVert_2$ (`09_modos_y_aprendizaje_test.png`).
   - Ablación de $16$ variantes, compromiso de dilatación a apertura fija ($D\in\{1,2,3,4\}$) y sensibilidad a $N_{\text{eff}}$ y $\delta$ sobre $200$ ventanas estratificadas de prueba (`08_sensibilidad_RLS_test.png`).
