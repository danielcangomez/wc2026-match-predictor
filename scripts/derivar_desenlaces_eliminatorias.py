"""Deriva los parámetros del módulo de prórroga/penaltis desde los datos crudos.

De dónde sale cada número que usa `resolver_eliminatoria` (Notebook 4, sección 4.6):

  1. El caché del scraper de SofaScore (data/raw/sofascore_equipos_cache.jsonl) guarda,
     para cada partido, el marcador a los 90' (`normaltime`) además del final (`current`),
     y los goles de la tanda si la hubo -- con eso se etiqueta CÓMO se decidió cada
     cruce: en 90 minutos, en la prórroga o en los penaltis. Eso es
     data/processed/desenlaces_sofascore.csv (una fila por partido).
  2. De los cruces que pasaron de los 90' salen las dos tasas base:
     P(prórroga | empate a 90') y P(penaltis | empate a 90').
  3. Para los decididos en la prórroga se ajusta una logística sin intercepto
     (sede neutral, simétrica) de P(pasa el local) sobre su diferencia de Elo en ese
     momento (cruzando con el tablón de features por fecha+equipos).
  4. Los penaltis se dejan en 50/50 A PROPÓSITO: el favorito por Elo ganó el 45.9% de
     las tandas (n=146) y el primer lanzador el 52.5% (n=259) -- ninguno de los dos
     efectos es distinguible de una moneda con esas muestras, y el primer lanzador ni
     siquiera se conoce antes del partido.

El resultado va a results/params_eliminatorias.csv... no: results/params_eliminatorias.json
(con las n de cada estimación, para saber cuánta confianza darle a cada número). Los
valores están replicados como literales en el Notebook 4 -- si re-derivas y cambian,
actualízalos allí también (el notebook avisa de dónde vienen).

Uso:
    python scripts/derivar_desenlaces_eliminatorias.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

DIR_RAIZ = Path(__file__).resolve().parent.parent
RUTA_CACHE = DIR_RAIZ / "data" / "raw" / "sofascore_equipos_cache.jsonl"
RUTA_DESENLACES = DIR_RAIZ / "data" / "processed" / "desenlaces_sofascore.csv"
RUTA_TABLON = DIR_RAIZ / "data" / "processed" / "partidos_features.csv"
RUTA_SALIDA = DIR_RAIZ / "results" / "params_eliminatorias.json"

# SofaScore nombra algunas selecciones distinto que results.csv (mismo motivo que los
# alias del Notebook 2, sección 2.2b).
ALIAS = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina", "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic", "Côte d'Ivoire": "Ivory Coast", "Türkiye": "Turkey",
    "USA": "United States", "Ireland": "Republic of Ireland",
}


def derivar_desenlaces() -> pd.DataFrame:
    """Una fila por partido del caché, con la etiqueta de cómo se decidió."""
    eventos = {}
    with open(RUTA_CACHE, encoding="utf-8") as f:
        for linea in f:
            fila = json.loads(linea)
            for ev in fila.get("eventos", []):
                eventos[ev["id"]] = ev

    filas = []
    for ev in eventos.values():
        hs, as_ = ev.get("homeScore") or {}, ev.get("awayScore") or {}
        if "normaltime" not in hs or "normaltime" not in as_:
            continue
        descripcion = (ev.get("status") or {}).get("description", "")
        fue_penaltis = "penalties" in hs or "penalties" in as_ or descripcion == "AP"
        fue_prorroga = (not fue_penaltis) and (
            "overtime" in hs or "overtime" in as_ or descripcion == "AET"
            or hs.get("current") != hs.get("normaltime") or as_.get("current") != as_.get("normaltime")
        )
        filas.append({
            "event_id": ev["id"],
            "fecha": datetime.fromtimestamp(ev["startTimestamp"], tz=timezone.utc).date().isoformat(),
            "torneo": (ev.get("tournament") or {}).get("name"),
            "local": ev["homeTeam"]["name"], "visitante": ev["awayTeam"]["name"],
            "gl_90": hs["normaltime"], "gv_90": as_["normaltime"],
            "gl_final": hs.get("current"), "gv_final": as_.get("current"),
            "pen_local": hs.get("penalties"), "pen_visitante": as_.get("penalties"),
            "winner_code": ev.get("winnerCode"),
            "decidido_en": "penaltis" if fue_penaltis else ("prorroga" if fue_prorroga else "90min"),
        })
    return pd.DataFrame(filas)


def main() -> None:
    if RUTA_DESENLACES.exists():
        desenlaces = pd.read_csv(RUTA_DESENLACES, parse_dates=["fecha"])
        print(f"Reutilizando {RUTA_DESENLACES.name} ({len(desenlaces):,} partidos).")
    else:
        desenlaces = derivar_desenlaces()
        desenlaces.to_csv(RUTA_DESENLACES, index=False)
        print(f"Derivado {RUTA_DESENLACES.name}: {len(desenlaces):,} partidos.")
        desenlaces["fecha"] = pd.to_datetime(desenlaces["fecha"])

    mas_alla_90 = desenlaces[desenlaces["decidido_en"] != "90min"].copy()
    p_prorroga = float((mas_alla_90["decidido_en"] == "prorroga").mean())
    p_penaltis = float((mas_alla_90["decidido_en"] == "penaltis").mean())

    # Cruce con el tablón para tener la diferencia de Elo de cada cruce en su fecha
    tablon = pd.read_csv(RUTA_TABLON, parse_dates=["fecha"],
                         usecols=["fecha", "equipo_local", "equipo_visitante", "elo_diff"])
    mas_alla_90["local_n"] = mas_alla_90["local"].replace(ALIAS)
    mas_alla_90["visitante_n"] = mas_alla_90["visitante"].replace(ALIAS)
    cruzado = mas_alla_90.merge(
        tablon, left_on=["fecha", "local_n", "visitante_n"],
        right_on=["fecha", "equipo_local", "equipo_visitante"], how="inner",
    )

    prorrogas = cruzado[cruzado["decidido_en"] == "prorroga"]
    x = prorrogas["elo_diff"].to_numpy()
    y = (prorrogas["winner_code"] == 1).to_numpy().astype(float)

    def neg_log_verosimilitud(beta: float) -> float:
        p = np.clip(1 / (1 + np.exp(-beta * x)), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    beta = float(minimize_scalar(neg_log_verosimilitud, bounds=(0, 0.05), method="bounded").x)

    penaltis = cruzado[cruzado["decidido_en"] == "penaltis"]
    favorito_gana_penaltis = float(((penaltis["elo_diff"] > 0) == (penaltis["winner_code"] == 1)).mean())

    params = {
        "p_prorroga_dado_empate90": round(p_prorroga, 4),
        "p_penaltis_dado_empate90": round(p_penaltis, 4),
        "beta_elo_prorroga": beta,
        "p_penaltis_favorito": 0.5,
        "n_total_mas90": int(len(mas_alla_90)),
        "n_prorrogas_ajuste": int(len(prorrogas)),
        "n_penaltis": int(len(penaltis)),
        "favorito_gana_penaltis_observado": round(favorito_gana_penaltis, 4),
        "nota": ("penaltis fijados en 50/50: ni la ventaja del favorito ni la del primer "
                 "lanzador son estadísticamente distinguibles de una moneda con estas muestras"),
    }
    RUTA_SALIDA.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(params, indent=2, ensure_ascii=False))
    print(f"\nGuardado: {RUTA_SALIDA}")
    print("Recuerda: el Notebook 4 (4.6) lleva estos valores como literales -- si cambian, actualízalos allí.")


if __name__ == "__main__":
    main()
