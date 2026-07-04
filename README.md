# WC 2026 Match Predictor

Sistema de predicción de resultados del Mundial 2026. La idea central: en vez de predecir
directamente "quién gana", se modelan los **goles de cada lado por separado** (dos procesos
de Poisson, local y visitante), con una corrección (Dixon-Coles) para el hecho de que los
marcadores bajos (0-0, 1-0, 0-1, 1-1) ocurren algo más de lo que la independencia pura
predice. De ahí se derivan todas las demás preguntas — quién gana, empate, marcador
probable, quién avanza en un cruce eliminatorio — como consecuencia matemática de esas dos
distribuciones.

El modelo nunca ve el resultado de una ronda antes de predecirla: cada ronda se evalúa "a
ciegas" y **solo después** se incorpora su resultado real para la siguiente (reentrenamiento
walk-forward), guardando un snapshot (`models/checkpoints/`) en cada etapa del torneo.

## El pipeline en 5 notebooks

| # | Notebook | Qué hace | ¿Cuándo se ejecuta? |
|---|---|---|---|
| 1 | [`01_adquisicion_datos.ipynb`](notebooks/01_adquisicion_datos.ipynb) | Descarga los resultados históricos y el Elo de cada selección | Para traer resultados nuevos |
| 2 | [`02_limpieza_y_eda.ipynb`](notebooks/02_limpieza_y_eda.ipynb) | Limpia, cruza fuentes, construye todas las features, y valida los supuestos estadísticos antes de modelar | Después del 1 |
| 3 | [`03_eleccion_modelo.ipynb`](notebooks/03_eleccion_modelo.ipynb) | Compara familias de modelo (GLM/LightGBM/XGBoost) contra 5 Mundiales pasados, no solo contra 2026 | Rara vez — solo para reconfirmar qué familia usar |
| 4 | [`04_prediccion.ipynb`](notebooks/04_prediccion.ipynb) | Entrena, predice partido a partido con reentrenamiento walk-forward, y simula el cuadro completo por Montecarlo (día 0 o con lo jugado hasta hoy) | Cada vez que hay resultados nuevos |
| 5 | [`05_cuadro_final.ipynb`](notebooks/05_cuadro_final.ipynb) | Cuadro visual (bracket) con banderas, real donde ya se jugó y predicho en cascada el resto | Después del 4 |

Aparte del pipeline (1-5), [`06_resumen_visual_portfolio.ipynb`](notebooks/06_resumen_visual_portfolio.ipynb)
es un recorrido visual completo del proyecto — datos, limpieza, features, estandarización,
EDA, comparación de modelos, calibración, precisión real y probabilidades — pensado para
mostrar el proyecto de principio a fin (portfolio / Kaggle). Autocontenido: solo lee de
`data/` y `results/`, no depende de que se acabe de ejecutar nada; para subirlo a Kaggle basta
con adjuntar esas carpetas como dataset y ajustar `DIR_DATOS` en la primera celda.

## La receta: qué ejecutar para cada cosa

**Actualizar la base de datos** (traer resultados nuevos):
1. Si tienes un `wc2026_calendario.json` con marcadores frescos, sobrescribe
   `data/raw/wc2026_calendario.json` (opcional: sin él, el mirror público de GitHub acaba
   actualizándose solo, aunque con más retraso).
2. Ejecuta el **Notebook 1** — siempre vuelve a descargar todo desde cero (no hace falta que
   distinga qué ya tenías, no tarda). Lo único que "actualiza en el sitio" es el paso del
   calendario JSON: si un partido ya estaba, lo sobrescribe; si es nuevo, lo añade.
3. Ejecuta el **Notebook 2** para reconstruir las features con los datos frescos.

**Obtener predicciones nuevas** (con los resultados ya jugados incorporados):
1. Ejecuta el **Notebook 4** — detecta qué rondas nuevas hay jugadas, evalúa la predicción que
   ya se había hecho para ellas (sin tocarla a posteriori) y **entonces** reentrena antes de
   predecir la siguiente ronda. Esto ES el reentrenamiento — no hace falta ningún paso aparte
   para "reentrenar".
2. Ejecuta el **Notebook 5** para ver el cuadro/bracket actualizado con los resultados nuevos.

**Ver las probabilidades de cada selección, actualizadas con lo jugado hasta hoy**:
- Dentro del Notebook 4, sección "Simulación Montecarlo": pon `ETAPA_SIMULACION = "actual"` y
  ejecuta desde ahí — no hace falta repetir todo el notebook. Da la probabilidad de cada
  selección de llegar a octavos/cuartos/semis/ser campeona, con todo lo jugado hasta hoy ya
  metido dentro. Si en cambio pones `ETAPA_SIMULACION = "pre_mundial"`, ves la misma vista pero
  congelada el día antes de empezar el torneo — útil para comparar "qué se esperaba" contra lo
  que pasó de verdad.

**¿Y si cambio de modelo?**: el Notebook 3 es el que decide, con evidencia (backtesting contra
2010/2014/2018/2022), qué familia usar — está pensado para ejecutarse rara vez, solo si quieres
reconfirmar la elección. Si su conclusión cambia, el Notebook 4 hoy fuerza la familia ganadora
a mano en su sección de selección de modelo (comentado el porqué ahí mismo).

## Estructura del repositorio

```
data/
  raw/                      # results.csv, elo_historico.csv, wc2026_calendario.json (opcional)
  processed/                # partidos_features.csv -- se regenera, no va en git
results/
  predicciones_*.csv        # predicciones partido a partido (SÍ va en git -- es el resultado final)
  comparacion_modelos.csv   # comparación de familias, 5 Mundiales
  simulacion_probabilidades_*.csv  # Montecarlo, "pre_mundial" y/o "actual"
  cuadro_completo.csv, bracket.html  # el cuadro visual
models/
  checkpoints/<etapa>/      # snapshot del modelo en cada punto del torneo
  modelo_goles_*.joblib     # estado "actual" (el más reciente)
notebooks/
  01_adquisicion_datos.ipynb
  02_limpieza_y_eda.ipynb
  03_eleccion_modelo.ipynb
  04_prediccion.ipynb
  05_cuadro_final.ipynb
tests/
  test_pipeline.py          # funciones puras extraídas de los notebooks (sin reimplementar)
requirements.txt
```

`data/` y `models/` están en `.gitignore` (se regeneran ejecutando los notebooks). `results/`
**sí** va en git: es el resultado final, no un paso intermedio.

## Cómo ejecutar

```
pip install -r requirements.txt
```

Para ejecutar un notebook de principio a fin desde la terminal (sin abrir Jupyter):

```
jupyter nbconvert --to notebook --execute --inplace notebooks/01_adquisicion_datos.ipynb
```

## Tests

```
pytest tests/
```

Cubren las funciones más delicadas (extraídas en caliente de los propios notebooks, sin
reimplementar): la moda correcta de una Poisson (y su matiz con la moda del marcador
*conjunto* cuando hay correlación de Dixon-Coles), que la decisión de quién avanza en un
cruce sea siempre por probabilidad y no por marcador puntual, que el `rho` de Dixon-Coles
se quede cerca de 0 con datos independientes, y la normalización local/visitante del JSON
del Mundial.

## Estado actual (última ejecución)

- Familia de modelo en producción: revisar `models/metadata.json` (`familia`).
- Campeón previsto: `results/cuadro_completo.csv`, partido 104 (Final).
- Probabilidades por selección: `results/simulacion_probabilidades_actual.csv`.
