"""Simulador de apuestas de valor: edge del modelo vs cuotas + gestión de banca Kelly.

Qué hace, dado un CSV de predicciones del modelo CON cuotas de mercado:
  1. Detecta apuestas de valor: edge = p_modelo x cuota - 1 > umbral (el modelo cree
     que el resultado es más probable de lo que la cuota implica).
  2. Dimensiona cada apuesta con Kelly fraccionado: f = fraccion x edge / (cuota - 1),
     con tope por apuesta. Kelly completo (fraccion=1) maximiza el crecimiento
     esperado del logaritmo de la banca PERO con varianza brutal; en la práctica se
     usa 1/4 o 1/2 de Kelly -- por eso `--kelly 0.25` es el defecto.
  3. Simula la banca en orden cronológico liquidando con los resultados reales, y
     reporta ROI, drawdown máximo y la secuencia completa.

Qué NO hace: inventar edge. Si el modelo no bate a las cuotas, el resultado sale
negativo y ese ES el resultado -- el propósito del script es medirlo, no adornarlo.
Benchmark previo obligatorio: comparar el LogLoss del modelo contra el de las
probabilidades implícitas del mercado (sin margen). Si el mercado gana en LogLoss,
no hay edge que explotar por definición y apostar es -EV.

Formato de entrada (CSV): columnas `fecha`, `equipo_local`, `equipo_visitante`,
`prob_local`, `prob_empate`, `prob_visitante` (del modelo), `cuota_local`,
`cuota_empate`, `cuota_visitante` (decimales europeos), y `resultado_1x2_real`
(LOCAL/EMPATE/VISITANTE) para liquidar. Filas sin cuotas o sin resultado se omiten.

Uso:
    python scripts/simulador_kelly.py --predicciones results/predicciones_con_cuotas.csv \
        --banca 100 --kelly 0.25 --umbral-edge 0.03 --tope-apuesta 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTADOS = ["LOCAL", "EMPATE", "VISITANTE"]
COLS_PROB = {"LOCAL": "prob_local", "EMPATE": "prob_empate", "VISITANTE": "prob_visitante"}
COLS_CUOTA = {"LOCAL": "cuota_local", "EMPATE": "cuota_empate", "VISITANTE": "cuota_visitante"}


def probabilidades_implicitas_sin_margen(fila: pd.Series) -> dict[str, float]:
    """Probabilidades implícitas del mercado quitando el margen de la casa
    (normalización proporcional: 1/cuota reescalado a que sume 1)."""
    brutas = {r: 1.0 / fila[COLS_CUOTA[r]] for r in RESULTADOS}
    total = sum(brutas.values())
    return {r: v / total for r, v in brutas.items()}


def comparar_con_mercado(df: pd.DataFrame) -> None:
    """LogLoss del modelo vs LogLoss de las probabilidades implícitas del mercado.
    Es el benchmark que decide si tiene sentido apostar: si el mercado tiene mejor
    LogLoss, el 'edge' que detecte el simulador es espejismo de calibración."""
    eps = 1e-12
    ll_modelo, ll_mercado = [], []
    for _, fila in df.iterrows():
        real = fila["resultado_1x2_real"]
        implicitas = probabilidades_implicitas_sin_margen(fila)
        ll_modelo.append(-np.log(max(fila[COLS_PROB[real]], eps)))
        ll_mercado.append(-np.log(max(implicitas[real], eps)))
    print(f"LogLoss modelo:  {np.mean(ll_modelo):.4f}")
    print(f"LogLoss mercado: {np.mean(ll_mercado):.4f} (implícitas sin margen)")
    if np.mean(ll_mercado) <= np.mean(ll_modelo):
        print("[aviso] El mercado está mejor calibrado que el modelo en esta muestra: "
              "el edge detectado abajo probablemente es ruido, interpreta el ROI con escepticismo.")


def simular(df: pd.DataFrame, banca_inicial: float, fraccion_kelly: float,
            umbral_edge: float, tope_apuesta: float) -> pd.DataFrame:
    banca = banca_inicial
    pico = banca_inicial
    drawdown_max = 0.0
    registro = []

    for _, fila in df.sort_values("fecha").iterrows():
        for r in RESULTADOS:
            p, cuota = fila[COLS_PROB[r]], fila[COLS_CUOTA[r]]
            edge = p * cuota - 1.0
            if edge <= umbral_edge:
                continue
            kelly = fraccion_kelly * edge / (cuota - 1.0)
            fraccion_apostada = min(kelly, tope_apuesta)
            importe = banca * fraccion_apostada
            if importe <= 0:
                continue
            gano = fila["resultado_1x2_real"] == r
            banca += importe * (cuota - 1.0) if gano else -importe
            pico = max(pico, banca)
            drawdown_max = max(drawdown_max, (pico - banca) / pico)
            registro.append({
                "fecha": fila["fecha"], "partido": f"{fila['equipo_local']} - {fila['equipo_visitante']}",
                "apuesta": r, "cuota": cuota, "p_modelo": round(p, 4), "edge": round(edge, 4),
                "importe": round(importe, 2), "gano": gano, "banca": round(banca, 2),
            })

    historial = pd.DataFrame(registro)
    print(f"\n=== Simulación: banca inicial {banca_inicial:.2f} ===")
    if historial.empty:
        print("Ninguna apuesta supera el umbral de edge -- no hay valor detectado (o el umbral es alto).")
        return historial
    n_ganadas = int(historial["gano"].sum())
    print(f"Apuestas: {len(historial)} ({n_ganadas} ganadas, {n_ganadas / len(historial):.1%})")
    print(f"Banca final: {banca:.2f}  |  ROI: {(banca - banca_inicial) / banca_inicial:+.1%}")
    print(f"Drawdown máximo: {drawdown_max:.1%}")
    print(f"Edge medio de las apuestas tomadas: {historial['edge'].mean():.3f}")
    return historial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predicciones", required=True, help="CSV con predicciones + cuotas (ver docstring)")
    parser.add_argument("--banca", type=float, default=100.0, help="Banca inicial (por defecto 100)")
    parser.add_argument("--kelly", type=float, default=0.25,
                        help="Fracción de Kelly (por defecto 0.25 -- Kelly completo es demasiado volátil)")
    parser.add_argument("--umbral-edge", type=float, default=0.03,
                        help="Edge mínimo para apostar (por defecto 0.03 = 3%%, absorbe ruido de calibración)")
    parser.add_argument("--tope-apuesta", type=float, default=0.05,
                        help="Fracción máxima de banca por apuesta (por defecto 0.05)")
    parser.add_argument("--salida", default=None, help="Ruta CSV donde guardar el historial de apuestas")
    args = parser.parse_args()

    df = pd.read_csv(args.predicciones, parse_dates=["fecha"])
    requeridas = set(COLS_PROB.values()) | set(COLS_CUOTA.values()) | {"resultado_1x2_real"}
    faltan = requeridas - set(df.columns)
    if faltan:
        raise SystemExit(f"Faltan columnas en el CSV: {sorted(faltan)}")

    df = df.dropna(subset=sorted(requeridas))
    df = df[df["resultado_1x2_real"].isin(RESULTADOS)]
    print(f"{len(df)} partidos con cuotas y resultado para liquidar.\n")

    comparar_con_mercado(df)
    historial = simular(df, args.banca, args.kelly, args.umbral_edge, args.tope_apuesta)

    if args.salida and not historial.empty:
        historial.to_csv(args.salida, index=False)
        print(f"Historial guardado: {args.salida}")


if __name__ == "__main__":
    main()
