"""Tercera vía: reutiliza la conexión a Opera (misma VPN, mismo CDP) pero en vez de
pedir la URL cruda del API navega la página real del partido y escucha la respuesta de
red -- ver `scrape_sofascore_humano.py` para el porqué (patrón de tráfico más parecido a
un usuario real). Sin verificar en la práctica hasta ahora; este es el primer intento.

Coge el MISMO turno que tenía el scraper automático de Opera (event_id impar) pero
empezando por el EXTREMO CONTRARIO de esa lista, para no pisar terreno ya intentado.

Uso:
    python scripts/scrape_sofascore_opera_manual.py [n_partidos]
    (por defecto 20 partidos, para probar el enfoque antes de dejarlo horas corriendo)
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
    RUTA_ESTADISTICAS_SALIDA,
    RUTA_PARTIDOS_SALIDA,
    _guardar_incremental,
)

CDP_URL = "http://localhost:9222"
FECHA_INICIO_ANIO = 1990
TURNO, DE_TURNOS = 1, 2  # mismo turno que tenía el automático de Opera (impares)
BASE_API = "https://api.sofascore.com/api/v1"


def descargar_estadisticas_via_pagina_real(page, event_id: int) -> dict | None:
    resp = page.goto(f"{BASE_API}/event/{event_id}", timeout=20_000)
    datos_evento = resp.json()["event"]
    url_partido = f"https://www.sofascore.com/{datos_evento['slug']}/{datos_evento['customId']}#tab:details"

    capturado = {}

    def _al_recibir(response):
        if f"/event/{event_id}/statistics" in response.url and response.status == 200:
            try:
                capturado["datos"] = response.json()
            except Exception:  # noqa: BLE001
                pass

    page.on("response", _al_recibir)
    page.goto(url_partido, timeout=30_000)
    page.mouse.wheel(0, 1200)
    page.wait_for_timeout(3000)
    page.remove_listener("response", _al_recibir)

    datos = capturado.get("datos")
    if datos is None:
        return None
    periodo_all = next((p for p in datos.get("statistics", []) if p.get("period") == "ALL"), None)
    if periodo_all is None:
        return None
    plano = {}
    for grupo in periodo_all.get("groups", []):
        for item in grupo.get("statisticsItems", []):
            clave = item.get("key")
            if clave and "homeValue" in item and "awayValue" in item:
                plano[clave] = (item["homeValue"], item["awayValue"])
    return plano or None


def main() -> None:
    n_partidos = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    progreso = Progreso.cargar()
    cache_equipos = cargar_cache_equipos(FECHA_INICIO_ANIO)
    eventos_por_id: dict[int, dict] = {}
    for datos in cache_equipos.values():
        for evento in datos["eventos"]:
            eventos_por_id.setdefault(evento["id"], evento)

    pendientes = [
        e for e in eventos_por_id.values()
        if e["id"] not in progreso.partidos_vistos and e["id"] % DE_TURNOS == TURNO
    ]
    pendientes = list(reversed(pendientes))[:n_partidos]  # extremo contrario de la lista
    print(f"{len(pendientes)} partidos a probar (extremo contrario del turno impar), "
          f"vía navegación real en vez de API directa.")

    with sync_playwright() as pw:
        navegador = pw.chromium.connect_over_cdp(CDP_URL)
        contexto = navegador.contexts[0] if navegador.contexts else navegador.new_context()
        page = contexto.new_page()

        filas_partidos, filas_estadisticas = [], []
        for i, evento in enumerate(pendientes, start=1):
            id_evento = evento["id"]
            fecha = datetime.fromtimestamp(evento["startTimestamp"], tz=timezone.utc)
            print(f"[{i}/{len(pendientes)}] {evento['homeTeam']['name']} vs "
                  f"{evento['awayTeam']['name']} ({fecha.date()}) -- id {id_evento}")

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

            estadisticas = descargar_estadisticas_via_pagina_real(page, id_evento)
            if estadisticas is None:
                print("    sin estadísticas capturadas")
            else:
                print(f"    {len(estadisticas)} estadísticas capturadas")
                fila_stats = {"event_id": id_evento}
                for clave, (v_local, v_visitante) in estadisticas.items():
                    fila_stats[f"{clave}_local"] = v_local
                    fila_stats[f"{clave}_visitante"] = v_visitante
                filas_estadisticas.append(fila_stats)

        _guardar_incremental(filas_partidos, filas_estadisticas)
        print(f"\nHecho. {len(filas_estadisticas)}/{len(pendientes)} con estadísticas. "
              f"Guardado en {RUTA_PARTIDOS_SALIDA} y {RUTA_ESTADISTICAS_SALIDA}.")


if __name__ == "__main__":
    main()
