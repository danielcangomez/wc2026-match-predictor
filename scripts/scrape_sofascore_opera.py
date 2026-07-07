"""Segunda instancia del scraper de SofaScore, pensada para correr EN PARALELO con
`scrape_sofascore.py` normal, conectándose a un navegador YA ABIERTO (Chrome, Opera,
cualquier Chromium) por su puerto de depuración remota, en vez de lanzar uno propio --
útil para usar un perfil/red distinta a la del proceso principal.

No relanza el mismo trabajo: solo coge partidos con `event_id` impar (el proceso
principal, en orden natural, tardará bastante en llegar a esa mitad) para reducir que
los dos descarguen el mismo partido -- no es una repartición perfecta (los dos leen el
progreso ya guardado solo al arrancar, no en cada partido), así que puede haber algún
partido repetido si ambos llevan mucho rato corriendo a la vez; no pasa nada, se
deduplica por `event_id` al construir el CSV final, antes de usar los datos.

Antes de ejecutar, abre el navegador que sea con el puerto de depuración remota
(usa un puerto libre -- 9222 puede estar ya ocupado por otro navegador):
    /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9223 &

Uso:
    python scripts/scrape_sofascore_opera.py [puerto]   # por defecto 9223
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_sofascore import (  # noqa: E402
    Progreso,
    cargar_cache_equipos,
    descargar_estadisticas_evento,
    RUTA_ESTADISTICAS_SALIDA,
    RUTA_PARTIDOS_SALIDA,
    _guardar_incremental,
)

PUERTO_CDP = int(sys.argv[1]) if len(sys.argv) > 1 else 9223
CDP_URL = f"http://localhost:{PUERTO_CDP}"
FECHA_INICIO_ANIO = 1990
PAUSA = 9.0
TURNO = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # por defecto, event_id impares -- ver docstring
DE_TURNOS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
# Reparte además el propio turno en dos mitades (para correr dos instancias del mismo turno
# en paralelo sin que se pisen): PARTE=0 se queda la mitad "de arriba" (ids más bajos) hasta
# el elemento del medio inclusive, PARTE=1 se queda desde ese mismo elemento del medio hasta
# el final -- se solapan en 1 partido a propósito, para no dejar un hueco si n es impar.
PARTE = int(sys.argv[4]) if len(sys.argv) > 4 else None


def main() -> None:
    fecha_inicio = datetime(FECHA_INICIO_ANIO, 1, 1, tzinfo=timezone.utc)

    progreso = Progreso.cargar()
    print(f"Retomando: {len(progreso.partidos_vistos)} partidos ya descargados (por cualquiera de las dos instancias).")

    cache_equipos = cargar_cache_equipos(FECHA_INICIO_ANIO)
    if not cache_equipos:
        print("[error] No hay caché de selecciones todavía -- deja que el script principal "
              "resuelva al menos una vez las 48 selecciones antes de lanzar este.")
        sys.exit(1)

    eventos_por_id: dict[int, dict] = {}
    for seleccion, datos in cache_equipos.items():
        for evento in datos["eventos"]:
            eventos_por_id.setdefault(evento["id"], evento)
    print(f"{len(eventos_por_id)} partidos distintos, de {len(cache_equipos)} selecciones en caché.")

    pendientes = [
        e for e in eventos_por_id.values()
        if e["id"] not in progreso.partidos_vistos and e["id"] % DE_TURNOS == TURNO
    ]
    pendientes.sort(key=lambda e: e["id"])
    if PARTE is not None:
        mitad = len(pendientes) // 2
        pendientes = pendientes[:mitad + 1] if PARTE == 0 else pendientes[mitad:]
        print(f"Mitad {PARTE} del turno (event_id %% {DE_TURNOS} == {TURNO}): "
              f"{len(pendientes)} partidos por descargar.")
    else:
        print(f"Este turno (event_id %% {DE_TURNOS} == {TURNO}): {len(pendientes)} partidos por descargar.")

    with sync_playwright() as pw:
        print(f"Conectando por CDP en {CDP_URL}...")
        navegador = pw.chromium.connect_over_cdp(CDP_URL)
        contexto = navegador.contexts[0] if navegador.contexts else navegador.new_context()
        page = contexto.new_page()
        print(f"Conectado. Si esto falla, revisa que el navegador esté abierto con "
              f"--remote-debugging-port={PUERTO_CDP}.")

        filas_partidos: list[dict] = []
        filas_estadisticas: list[dict] = []
        for i, evento in enumerate(pendientes, start=1):
            id_evento = evento["id"]
            fecha = datetime.fromtimestamp(evento["startTimestamp"], tz=timezone.utc)
            filas_partidos.append({
                "event_id": id_evento,
                "fecha": fecha.date().isoformat(),
                "torneo": evento.get("tournament", {}).get("name"),
                "equipo_local": evento["homeTeam"]["name"],
                "equipo_visitante": evento["awayTeam"]["name"],
                "goles_local": evento.get("homeScore", {}).get("normaltime"),
                "goles_visitante": evento.get("awayScore", {}).get("normaltime"),
                "estado": evento.get("status", {}).get("type"),
            })

            estadisticas = descargar_estadisticas_evento(page, id_evento, PAUSA)
            if estadisticas is not None:
                fila_stats = {"event_id": id_evento}
                for clave, (valor_local, valor_visitante) in estadisticas.items():
                    fila_stats[f"{clave}_local"] = valor_local
                    fila_stats[f"{clave}_visitante"] = valor_visitante
                filas_estadisticas.append(fila_stats)

            if i % 25 == 0 or i == len(pendientes):
                _guardar_incremental(filas_partidos, filas_estadisticas)
                print(f"  ... {i}/{len(pendientes)} procesados ({len(filas_estadisticas)} con estadísticas) "
                      "-- guardado parcial.")
                filas_partidos.clear()
                filas_estadisticas.clear()

        _guardar_incremental(filas_partidos, filas_estadisticas)
        print(f"\nHecho (este turno). Ver {RUTA_PARTIDOS_SALIDA} y {RUTA_ESTADISTICAS_SALIDA} "
              "-- deduplica por event_id antes de usarlos si las dos instancias corrieron a la vez.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido -- lo ya guardado no se pierde.")
        sys.exit(130)
