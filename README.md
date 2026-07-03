# WC 2026 Match Predictor

Sistema de predicción de resultados del Mundial 2026: modela los goles de cada
partido como dos procesos independientes (local / visitante) con la corrección de
Dixon-Coles para marcadores bajos, compara varias familias de modelos, simula el
torneo fase a fase con un pipeline de **reentrenamiento walk-forward**, y guarda un
snapshot del modelo en cada etapa del torneo — el modelo nunca ve el resultado de una
ronda antes de predecirla.

## Pipeline

6 notebooks secuenciales; cada uno consume la salida del anterior desde `data/`:

| Notebook | Objetivo | Entrada → Salida |
|---|---|---|
| [`01_adquisicion_datos.ipynb`](notebooks/01_adquisicion_datos.ipynb) | Descarga programática de datos; opcionalmente refresca 2026 con un JSON manual | → `data/raw/results.csv`, `data/raw/elo_historico.csv` |
| [`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) | Cruce de fuentes + features con ventanas móviles, sin fuga temporal | `data/raw/*` → `data/processed/partidos_features.csv` |
| [`03_eda_avanzado.ipynb`](notebooks/03_eda_avanzado.ipynb) | Valida los supuestos matemáticos antes de modelar | `data/processed/*` → `FEATURES_FINALES` |
| [`04_modelado_optuna_walkforward.ipynb`](notebooks/04_modelado_optuna_walkforward.ipynb) | Entrena, compara modelos, simula el torneo ronda a ronda, persiste checkpoints | `data/processed/*` → predicciones + `models/` |
| [`05_comparacion_modelos.ipynb`](notebooks/05_comparacion_modelos.ipynb) | Compara GLM, LightGBM, XGBoost y Random Forest sobre el mismo split | `data/processed/*` → `comparacion_modelos.csv` |
| [`06_simulacion_montecarlo.ipynb`](notebooks/06_simulacion_montecarlo.ipynb) | Simula el cuadro completo con el modelo congelado *antes* del Mundial | `data/raw/wc2026_calendario.json` → probabilidades por ronda |

### 1. Adquisición de datos

- **Histórico de partidos**: espejo público en GitHub del dataset de Kaggle
  `martj42/international-football-results` (1872 → hoy), vía `raw.githubusercontent.com`.
- **Histórico de Elo**: `eloratings.net`, un TSV por selección con el Elo posterior a
  cada partido, para las 48 clasificadas al Mundial 2026 (detectadas automáticamente a
  partir del propio histórico, sin lista fija a mano).
- **Sección 1.5 (opcional)**: si existe `data/raw/wc2026_calendario.json` (calendario
  con marcadores, formato `openfootball`), sus partidos ya jugados refrescan
  `results.csv` sin esperar a que el mirror de GitHub los tenga — útil durante el propio
  torneo. El local se determina por el país de la sede, no por el orden `team1`/`team2`
  del JSON (que no siempre coincide con quién juega en casa).
- Validación de tipos, nulos y rangos físicamente posibles antes de persistir.

### 2. Feature engineering

Formato largo (una fila = un equipo en un partido) para que las ventanas móviles sean
triviales; vuelve a ancho al final. Por partido:

- **Elo antes del partido** (nunca después): `elo.shift(1)` por selección + `merge_asof`
  hacia atrás — evita la fuga de información más sutil del pipeline.
- **Forma reciente**: goles a favor/en contra y racha de puntos a 5 y 10 partidos, con
  `shift(1)` antes del `rolling`.
- **Días de descanso** desde el partido anterior de cada selección.
- Diferencias directas (`elo_diff`, diferencias de forma) para el modelo.

Partidos programados pero no jugados se conservan pero se excluyen del entrenamiento.

### 3. EDA avanzado

- Los goles se aproximan a Poisson (con sobredispersión) → sostiene el GLM de Poisson.
- Se descartan partidos anteriores a **1990** (juego no estacionario antes de esa fecha).
- Colinealidad (Spearman + VIF): quedan `elo_diff`, `tendencia_elo_local/visitante`,
  `dif_forma_gf_5/10`, `dif_racha_5/10`, `dias_descanso_local/visitante`.
- El Elo solo ya separa razonablemente bien ganadores de perdedores → cota mínima de
  accuracy a superar.

### 4. Modelado, walk-forward y checkpoints

- **Split temporal**: entrenamiento con todo el histórico desde 1990 hasta el 10 de
  junio de 2026 (día antes del Mundial) — el modelo no ve ni un partido del torneo antes
  de predecirlo.
- **Modelos de goles**: Poisson GLM vs. LightGBM afinado con Optuna (CV temporal, nunca
  K-Fold barajado). El GLM gana en la fase de grupos — ver `models/metadata.json` para
  las métricas exactas de la última ejecución.
- **Corrección de Dixon-Coles (1997)**: un Poisson bivariante independiente infravalora
  los marcadores bajos (0-0, 1-0, 0-1, 1-1); se calibra un `rho` por máxima
  verosimilitud sobre el propio entrenamiento y se aplica a la matriz conjunta.
- **Resolución de eliminatorias**: se compara **siempre** la probabilidad acumulada de
  cada lado en la matriz de probabilidad conjunta — nunca un marcador puntual (ni
  `round(lambda)`, que ni siquiera es la moda correcta de una Poisson, ni `floor(lambda)`,
  que sí lo es pero no equivale a "quién tiene más probabilidad de ganar").
- **Predicciones partido a partido**: tanto de fase de grupos
  (`predicciones_fase_grupos.csv`, sin "avanza" — ese concepto no aplica a un partido de
  grupo) como de eliminatoria (`predicciones_eliminatoria.csv`, con avance y método de
  desempate) y de los próximos partidos aún sin jugar (`predicciones_proximos_partidos.csv`).
- **Bucle walk-forward**: las rondas se detectan por salto de fecha. Para cada ronda:
  evaluar con el modelo *antes* de conocerla → resolver cada cruce → **solo entonces**
  incorporar los resultados reales y reentrenar.
- **Checkpoints por etapa** (`models/checkpoints/<etapa>/`): `pre_mundial`,
  `post_grupos`, `post_dieciseisavos`, `post_octavos`... cada uno con su propio
  modelo + `metadata.json`, además del estado "actual" en `models/`. El Notebook 6
  reutiliza el checkpoint `pre_mundial` en vez de reentrenar.

### 5. Comparación extendida de modelos

Mismo split y métricas que el Notebook 4, añadiendo XGBoost y Random Forest (ambos con
objetivo/criterio de Poisson, afinados con Optuna) para comprobar si alguna otra familia
de árboles supera al GLM. Resultado en `data/processed/comparacion_modelos.csv`.

### 6. Simulación Monte Carlo del cuadro completo

A diferencia del walk-forward (que ya incorpora resultados reales ronda a ronda), este
notebook congela el modelo **tal como estaba el día antes del Mundial** (sin ver ni la
fase de grupos) y simula miles de veces el cuadro entero de eliminatoria —
dieciseisavos a la final — con sorteos de Poisson (más el mismo ajuste de Dixon-Coles),
para propagar la incertidumbre ronda a ronda en vez de asumir siempre "gana el
favorito". Salida: probabilidad de cada selección de alcanzar cada ronda
(`simulacion_probabilidades_equipos.csv`) y marcador previsto + probabilidad de cada
cruce de dieciseisavos (`simulacion_dieciseisavos.csv`).

## Estructura del repositorio

```
data/
  raw/                      # results.csv, elo_historico.csv, wc2026_calendario.json (opcional)
  processed/                # partidos_features.csv y todos los *.csv de predicciones/comparación
models/
  checkpoints/<etapa>/      # snapshot del modelo en cada punto del torneo
  modelo_goles_*.joblib     # estado "actual" (el más reciente)
  metadata.json
notebooks/
  01_adquisicion_datos.ipynb
  02_feature_engineering.ipynb
  03_eda_avanzado.ipynb
  04_modelado_optuna_walkforward.ipynb
  05_comparacion_modelos.ipynb
  06_simulacion_montecarlo.ipynb
tests/
  test_pipeline.py          # funciones puras extraídas de los notebooks (sin reimplementar)
requirements.txt
.env                        # FUTBOL_API_KEY (no versionado)
```

`data/` y `models/` están en `.gitignore`: todo se regenera ejecutando los notebooks en
orden desde cero, sin depender de ficheros pre-descargados.

## Cómo ejecutar

```
pip install -r requirements.txt
```

Ejecutar los notebooks en orden (01 → 04 es el camino crítico; 05 y 06 son análisis
paralelos, no hace falta reejecutarlos en cada actualización).

## Actualizar tras una nueva ronda

1. Si tienes un `wc2026_calendario.json` con los marcadores nuevos, sobrescribe
   `data/raw/wc2026_calendario.json` (si no, el mirror de GitHub acaba actualizándose solo).
2. Ejecutar Notebook 1 (ingiere los resultados nuevos).
3. Ejecutar Notebook 2 (regenera las features).
4. Ejecutar Notebook 4 (detecta la ronda nueva, la evalúa a ciegas, reentrena, guarda
   checkpoint, y predice los siguientes partidos pendientes en la sección 4.9).

Los notebooks 3, 5 y 6 no forman parte de este ciclo: 3 es un análisis puntual, 5 tiene
sentido repetirlo de vez en cuando (no por ronda), y 6 usa el modelo pre-Mundial, que no
cambia aunque avance el torneo.

## Tests

```
pytest tests/
```

Cubren las funciones más delicadas (extraídas en caliente de los propios notebooks, sin
reimplementar): la moda correcta de una Poisson, que `resolver_eliminatoria` decida
siempre por probabilidad y no por marcador puntual, que el `rho` de Dixon-Coles se
quede cerca de 0 con datos independientes, y la normalización local/visitante del JSON
del Mundial.
