# Estrategia de apuestas: maximizar beneficio/riesgo con la evidencia en la mano

**TL;DR: con la evidencia medida hasta hoy, la posición que maximiza beneficio/riesgo es
stake 0.** No es una renuncia — es el resultado del mismo proceso de medición que decidió
todo lo demás en este proyecto. Este documento deja la estrategia *condicional* lista para
activarse el día que la condición de entrada se cumpla, y el procedimiento exacto para
re-evaluarla.

## 1. La evidencia (toda reproducible en `results/`)

| Test | Resultado | Archivo |
|---|---|---|
| LogLoss modelo vs mercado, 4 Mundiales (266 partidos, cuotas medias de cierre) | El mercado gana **4 de 4** (0.926 vs 0.999 agregado) | `apuestas_benchmark_mercado.csv` |
| Kelly fraccionado (25%) sobre el edge aparente, banca 100 | **ROI −80.6%**, drawdown 82% | `apuestas_historial_kelly.csv` |
| Apuesta de valor en empates (umbral P(X)≥0.29 pre-registrado, stake plano) | **ROI −36.8%** (65 apuestas), negativo en todo el barrido de umbrales | sección 16b del notebook 06 |

Lectura: el "edge" que el modelo cree ver (edge medio 0.42 en las apuestas tomadas) es un
artefacto de calibración — el modelo está *más equivocado* que la cuota justo cuando más
discrepa de ella. Apostar discrepancias así es pagar el margen de la casa más el error
propio.

## 2. La regla de activación (cuándo SÍ apostar)

Solo existe edge explotable si el modelo bate al mercado en probabilidad, no en corazonadas:

> **Condición de entrada**: LogLoss del modelo < LogLoss de las probabilidades implícitas
> del mercado (sin margen), sobre una ventana de validación reciente de ≥ 60 partidos con
> cuotas reales, evaluada ANTES de apostar.
> `python scripts/simulador_kelly.py --predicciones <csv>` la comprueba automáticamente
> y avisa si no se cumple.

Hoy no se cumple (ni de cerca). Qué podría cambiarlo: información que el mercado tarda en
incorporar (mercados menores con menos liquidez, cuotas de apertura en vez de cierre, o
una fuente de datos que el consenso no usa — el valor de plantilla histórico por fecha
sigue pendiente de backtest). Sin algo así, no se activa.

## 3. La estructura de staking (si la condición se cumple)

1. **Kelly fraccionado al 25%** (`--kelly 0.25`): Kelly completo maximiza crecimiento
   esperado pero con drawdowns que ninguna banca real tolera; 1/4 de Kelly conserva ~2/3
   del crecimiento con ~1/4 de la varianza.
2. **Umbral de edge mínimo 3%** (`--umbral-edge 0.03`): absorbe el ruido de calibración
   residual — un edge menor que el margen de la casa no es señal.
3. **Tope del 5% de la banca por apuesta** (`--tope-apuesta 0.05`): protege de los edges
   grandes, que en este dominio son casi siempre errores del modelo, no regalos del mercado.
4. **Banca aparte y fija** (el `--banca` inicial): nunca recargar a mitad de torneo — la
   recarga convierte un sistema con freno en una martingala.

## 4. Procedimiento de re-evaluación

1. Regenerar predicciones con el pipeline (notebooks 1→4).
2. Descargar cuotas actualizadas: `https://www.football-data.co.uk/WorldCup2026.xlsx`.
3. Cruzar y comparar: el script del backtest está en el historial del proyecto y los CSVs
   de entrada (`results/apuestas_wc*.csv`) muestran el formato exacto.
4. Si (y solo si) la condición de entrada se cumple, ejecutar el simulador con los
   parámetros de la sección 3 sobre partidos FUTUROS, nunca re-ajustados sobre los ya vistos.

## 5. Por qué publicar una estrategia que dice "no apuestes"

Porque la alternativa — publicar un sistema con backtest inflado por fuga temporal o
umbrales sobre-ajustados — es exactamente lo que este proyecto lleva semanas evitando en
la parte de modelado, y las apuestas no merecen menos rigor. Un lector que entienda esta
página sabe más de apuestas de valor que la mayoría de vendedores de picks: sabe medir
si el edge existe antes de pagar por descubrir que no.
