"""Descarga estadísticas avanzadas (tiros, posesión, corners, xG cuando exista...) de
SofaScore para los partidos de las 48 selecciones del Mundial 2026, desde 1990.

Por qué un script y no un notebook: esto son miles de peticiones con espera entre cada
una por respeto al servidor -- puede tardar horas, y tiene que poder interrumpirse y
retomarse sin perder lo ya descargado. Un notebook no está pensado para eso.

Cobertura real de SofaScore (comprobado ejecutando este script de verdad contra España,
323 partidos, antes de darlo por bueno): partidos completos (amistosos + clasificación +
fase final) solo desde ~2005; entre 1990 y 2004 solo hay clasificación y fase final, SIN
amistosos -- se acepta ese hueco a propósito en vez de forzar un rango más corto. De los
partidos que SÍ tiene, no todos traen estadísticas (España: 210/323, ~65%) -- pero cuando
las trae, vienen prácticamente completas: tiros, posesión, corners Y expected goals están
presentes incluso en partidos de 1990 (no solo en los recientes, contra lo que se pensaba
al principio) -- sale como columna vacía únicamente en los partidos sin estadísticas en
absoluto, nunca a medias.

IMPORTANTE -- por qué esto usa un navegador de verdad (Playwright) y no peticiones HTTP,
y por qué aun así hace falta ir despacio:
1) `requests` normal recibe 403 siempre -- huella TLS/JA3, no cabeceras: el handshake no
   se parece al de un navegador real y lo cortan antes de mirar nada más.
2) `curl_cffi` (imita esa huella TLS) sorteaba eso al principio, pero tras un rato de uso
   sostenido empezó a recibir 403 con `"reason": "challenge"` -- comprobado que NINGUNA
   huella TLS de las que soporta (Chrome, Firefox, Safari, Edge, Android...) lo evitaba.
3) Un Chromium real vía Playwright sí pasaba ese "challenge" al principio (200 OK
   comprobado en vivo) -- pero tras ~350 peticiones más volvió a aparecer el mismo 403.
   Conclusión, comprobada, no supuesta: el bloqueo NO es de huella TLS ni de motor de
   navegador -- es un límite de volumen acumulado por IP a lo largo del tiempo. Ni la
   mejor imitación de navegador lo evita, solo bajar el ritmo (o cambiar de IP). Por eso
   `--pausa` por defecto es alta (9s) -- no es prudencia de sobra, es lo que hizo falta.

Antes de ejecutar (una sola vez):
    pip install -r requirements.txt
    playwright install chromium

Salida:
    data/raw/sofascore_partidos.jsonl       -- estado de trabajo, 1 línea JSON/partido (no
    data/raw/sofascore_estadisticas.jsonl      tocar a mano; de aquí se retoma si se corta)
    data/raw/sofascore_equipos_cache.jsonl  -- historial de partidos ya listado por selección
    data/raw/sofascore_partidos.csv         -- version final, para el resto del pipeline
    data/raw/sofascore_estadisticas.csv        (se regenera entera en cada punto de guardado)

Por qué JSONL para el estado de trabajo y no CSV directamente: no todos los partidos
tienen las mismas estadísticas disponibles (un portero puede tener "highClaims" y otro
partido no traer esa clave en absoluto) -- CSV exige una tabla de columnas fija, así que
si cada tanda de 25 partidos se APPENDEA a un CSV con las columnas que ESA tanda tenga, el
fichero queda con un número de columnas distinto por tramos y se corrompe (justo lo que
pasó en la primera versión de este script). JSON Lines no tiene ese problema -- cada línea
lleva sus propias claves -- y el CSV final se reconstruye entero a partir de todo el JSONL
acumulado, dejando que pandas alinee las columnas (huecos como NaN) de forma consistente.

Uso:
    python scripts/scrape_sofascore.py
    python scripts/scrape_sofascore.py --fecha-inicio 2005 --pausa 2.0
    python scripts/scrape_sofascore.py --solo-equipos Spain Argentina  # para probar rápido
    python scripts/scrape_sofascore.py --con-cabeza  # ver el navegador mientras descarga
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from playwright.sync_api import Page, sync_playwright

DIR_RAIZ = Path(__file__).resolve().parent.parent
DIR_RAW = DIR_RAIZ / "data" / "raw"
RUTA_RESULTADOS = DIR_RAW / "results.csv"
RUTA_PARTIDOS_JSONL = DIR_RAW / "sofascore_partidos.jsonl"
RUTA_ESTADISTICAS_JSONL = DIR_RAW / "sofascore_estadisticas.jsonl"
RUTA_EQUIPOS_CACHE_JSONL = DIR_RAW / "sofascore_equipos_cache.jsonl"
RUTA_PARTIDOS_SALIDA = DIR_RAW / "sofascore_partidos.csv"
RUTA_ESTADISTICAS_SALIDA = DIR_RAW / "sofascore_estadisticas.csv"

BASE_URL = "https://api.sofascore.com/api/v1"

# Comprobado en la práctica: el bloqueo por volumen aparece cada ~200-350 peticiones y
# tarda horas en levantarse -- estos valores hacen que el script espere solo en vez de
# morirse y necesitar que alguien lo relance a mano cada vez.
ESPERA_BLOQUEO_INICIAL = 20 * 60   # 20 min la primera vez que se detecta el bloqueo
ESPERA_BLOQUEO_MAXIMA = 90 * 60    # techo de 90 min por ciclo, no sigue creciendo sin límite
CICLOS_BLOQUEO_MAXIMOS = 30        # ~45h de espera acumulada en el peor caso antes de rendirse

# Un puñado de nombres no coinciden entre el histórico del proyecto y la búsqueda de
# SofaScore (mismo motivo que ALIAS_ELORATINGS / MAPEO_NOMBRES_JSON en los notebooks 1 y
# 4-5) -- se completa según haga falta, no hay forma de saberlo sin probar contra el sitio
# real. Si el script avisa de una selección no resuelta, añade aquí su alias y vuelve a
# ejecutar (es idempotente: no repite lo ya descargado).
ALIAS_SOFASCORE: dict[str, str] = {
    "Ivory Coast": "Côte d'Ivoire",
    "United States": "USA",
    "Bosnia and Herzegovina": "Bosnia",
    "Turkey": "Türkiye",
}

# Patrón para descartar categorías inferiores/femenino/olímpico al buscar una selección
# (SofaScore devuelve todas las categorías de una federación en la misma búsqueda).
PATRON_CATEGORIA_INFERIOR = re.compile(
    r"\b(U1[5-9]|U2[0-3]|olympic|women|futsal|beach)\b", re.IGNORECASE
)


def _leer_jsonl(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def _anadir_jsonl(ruta: Path, filas: list[dict]) -> None:
    if not filas:
        return
    with open(ruta, "a", encoding="utf-8") as f:
        for fila in filas:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")


@dataclass
class Progreso:
    """Qué se ha descargado ya, para poder interrumpir y retomar sin repetir trabajo."""

    partidos_vistos: set[int]

    @classmethod
    def cargar(cls) -> "Progreso":
        partidos_vistos = {fila["event_id"] for fila in _leer_jsonl(RUTA_PARTIDOS_JSONL)}
        return cls(partidos_vistos)


def cargar_cache_equipos(fecha_inicio_anio: int) -> dict[str, dict]:
    """Selección -> {id_equipo, eventos} ya resueltos en una ejecución anterior -- listar
    el histórico de una selección son varias navegaciones de paginación (hasta ~15
    páginas), y sin esto un corte a mitad del bucle de selecciones obligaba a repetirlas
    todas desde la selección 1. Se descarta la entrada si se guardó para un
    `--fecha-inicio` distinto al de esta ejecución (el histórico cambiaría)."""
    cache = {}
    for fila in _leer_jsonl(RUTA_EQUIPOS_CACHE_JSONL):
        if fila.get("fecha_inicio_anio") == fecha_inicio_anio:
            cache[fila["seleccion"]] = fila
    return cache


def guardar_cache_equipo(seleccion: str, id_equipo: int, eventos: list[dict], fecha_inicio_anio: int) -> None:
    _anadir_jsonl(RUTA_EQUIPOS_CACHE_JSONL, [{
        "seleccion": seleccion, "id_equipo": id_equipo, "eventos": eventos,
        "fecha_inicio_anio": fecha_inicio_anio,
    }])


def _get_con_reintentos(page: Page, url: str, pausa: float, intentos: int = 5) -> dict | None:
    """Navega a `url` (un endpoint JSON de la API) con reintentos y backoff. Devuelve
    None (no lanza) en 404 -- un 404 es un resultado válido aquí (partido sin
    estadísticas, o fin de la paginación de un equipo), no un fallo.

    Un 403 puntual (throttling temporal tras muchas peticiones seguidas) se reintenta
    con una espera bastante más larga en vez de abortar a la primera. Si el ÚLTIMO
    intento también devuelve 403 -- comprobado en la práctica que esto pasa cada
    ~200-350 peticiones -- no se lanza ni se muere el script: se entra en un ciclo de
    espera LARGA (empieza en `ESPERA_BLOQUEO_INICIAL`, crece x1.6 cada vez, techo
    `ESPERA_BLOQUEO_MAXIMA`) reintentando la MISMA petición hasta que vuelva un 200 --
    así el script sobrevive solo horas sin que haga falta relanzarlo a mano cada vez
    que SofaScore corta. Solo se rinde de verdad tras `CICLOS_BLOQUEO_MAXIMOS` ciclos
    largos (con eso ya habría esperado casi dos días -- si sigue bloqueado a esas
    alturas, algo más profundo está pasando y seguir insistiendo no tiene sentido)."""
    ciclos_bloqueo = 0
    while True:
        ultimo_403 = False
        for intento in range(1, intentos + 1):
            try:
                resp = page.goto(url, timeout=20_000)
            except Exception as exc:  # noqa: BLE001 -- Playwright lanza varios tipos (timeout, conexión...)
                print(f"  [aviso] fallo de navegación en intento {intento}/{intentos} para {url}: {exc}")
                time.sleep(pausa * intento)
                continue

            if resp is None:
                print(f"  [aviso] navegación sin respuesta en intento {intento}/{intentos} para {url}")
                time.sleep(pausa * intento)
                continue

            if resp.status == 404:
                return None
            if resp.status == 200:
                # Jitter: una pausa perfectamente regular es en sí misma una señal de
                # bot para una protección por comportamiento -- variar +/-40% la hace
                # parecer menos a un script y más a tráfico real.
                time.sleep(random.uniform(pausa * 0.6, pausa * 1.4))
                return resp.json()

            ultimo_403 = resp.status == 403
            espera = pausa * intento * (5 if ultimo_403 else 1)
            print(f"  [aviso] HTTP {resp.status} en intento {intento}/{intentos} para {url} "
                  f"-- esperando {espera:.0f}s antes de reintentar")
            time.sleep(espera)

        if not ultimo_403:
            print(f"  [aviso] se agotaron los reintentos para {url}, se omite")
            return None

        ciclos_bloqueo += 1
        if ciclos_bloqueo > CICLOS_BLOQUEO_MAXIMOS:
            raise RuntimeError(
                f"403 Forbidden persistente en {url} tras {ciclos_bloqueo - 1} ciclos de espera "
                f"larga (~{ESPERA_BLOQUEO_MAXIMA * (ciclos_bloqueo - 1) / 3600:.0f}h) -- esto ya no "
                "es un bloqueo temporal normal. Lo ya descargado no se pierde; investiga antes de "
                "seguir insistiendo."
            )
        espera_larga = min(ESPERA_BLOQUEO_INICIAL * (1.6 ** (ciclos_bloqueo - 1)), ESPERA_BLOQUEO_MAXIMA)
        espera_larga = random.uniform(espera_larga * 0.85, espera_larga * 1.15)
        print(f"  [bloqueo] 403 persistente para {url} -- ciclo de espera larga "
              f"{ciclos_bloqueo}/{CICLOS_BLOQUEO_MAXIMOS}, durmiendo {espera_larga/60:.1f} min "
              f"antes de reintentar la misma petición.")
        time.sleep(espera_larga)


def resolver_equipo_sofascore(page: Page, nombre: str, pausa: float) -> int | None:
    """Busca el id de SofaScore de la selección masculina absoluta `nombre`. Descarta
    categorías inferiores y se queda con el resultado de mayor `userCount` -- la
    selección absoluta siempre tiene muchísimo más seguimiento que sus categorías
    inferiores."""
    consulta = ALIAS_SOFASCORE.get(nombre, nombre)
    datos = _get_con_reintentos(page, f"{BASE_URL}/search/all?q={consulta}&page=0", pausa)
    if datos is None:
        return None

    candidatos = [
        r["entity"] for r in datos.get("results", [])
        if r.get("type") == "team"
        and r["entity"].get("sport", {}).get("id") == 1
        and r["entity"].get("national") is True
        and r["entity"].get("gender") == "M"
        and not PATRON_CATEGORIA_INFERIOR.search(r["entity"].get("name", ""))
    ]
    if not candidatos:
        return None
    mejor = max(candidatos, key=lambda e: e.get("userCount", 0))
    return mejor["id"]


def descargar_historico_equipo(page: Page, id_equipo: int, fecha_inicio: datetime,
                                 pausa: float) -> list[dict]:
    """Pagina `/team/{id}/events/last/{page}` (más reciente primero) hasta cruzar
    `fecha_inicio` o hasta que la API devuelva 404 (fin del histórico de ese equipo)."""
    eventos = []
    pagina = 0
    while True:
        datos = _get_con_reintentos(page, f"{BASE_URL}/team/{id_equipo}/events/last/{pagina}", pausa)
        if datos is None or not datos.get("events"):
            break
        pagina_tiene_eventos_recientes = False
        for evento in datos["events"]:
            fecha = datetime.fromtimestamp(evento["startTimestamp"], tz=timezone.utc)
            if fecha < fecha_inicio:
                continue
            pagina_tiene_eventos_recientes = True
            eventos.append(evento)
        # Las páginas vienen ordenadas de más a menos reciente: si ESTA página ya no
        # trajo ningún evento posterior a fecha_inicio, las siguientes tampoco lo harán.
        if not pagina_tiene_eventos_recientes:
            break
        pagina += 1
    return eventos


def descargar_estadisticas_evento(page: Page, id_evento: int, pausa: float) -> dict | None:
    """Aplana el grupo de periodo "ALL" a un diccionario {key: (home, away)}. Devuelve
    None si el partido no tiene estadísticas (frecuente en amistosos menores o partidos
    muy antiguos) -- se omite, no se rellena con ceros (un 0 real y un "no hay dato" no
    son lo mismo)."""
    datos = _get_con_reintentos(page, f"{BASE_URL}/event/{id_evento}/statistics", pausa)
    if datos is None:
        return None
    periodo_all = next((p for p in datos.get("statistics", []) if p.get("period") == "ALL"), None)
    if periodo_all is None:
        return None

    plano: dict[str, tuple[float, float]] = {}
    for grupo in periodo_all.get("groups", []):
        for item in grupo.get("statisticsItems", []):
            clave = item.get("key")
            if clave and "homeValue" in item and "awayValue" in item:
                plano[clave] = (item["homeValue"], item["awayValue"])
    return plano or None


def cargar_selecciones_mundial_2026() -> list[str]:
    """Misma lógica que `extraer_selecciones_mundial_2026` del Notebook 1 -- se deriva
    del propio histórico en vez de mantener una lista fija aparte."""
    df = pd.read_csv(RUTA_RESULTADOS, parse_dates=["date"])
    es_mundial_2026 = (df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)
    partidos = df[es_mundial_2026]
    return sorted(set(partidos["home_team"]) | set(partidos["away_team"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fecha-inicio", type=int, default=1990, help="Año desde el que descargar (por defecto 1990)")
    parser.add_argument("--pausa", type=float, default=9.0,
                         help="Segundos de espera entre navegaciones, con jitter +/-40%% (por defecto 9.0 -- "
                              "ni la huella TLS ni un navegador real evitan el bloqueo de SofaScore, es por "
                              "volumen acumulado en la IP con el tiempo; a este ritmo la descarga completa "
                              "tarda del orden de un día largo, pero es lo que hizo falta para no repetir el "
                              "bloqueo en la práctica)")
    parser.add_argument("--solo-equipos", nargs="*", default=None,
                         help="Limita la descarga a estas selecciones (para probar rápido)")
    parser.add_argument("--con-cabeza", action="store_true",
                         help="Muestra la ventana del navegador en vez de correr headless (para depurar)")
    parser.add_argument("--turno", type=int, default=None,
                         help="Si se indica junto a --de-turnos, limita la descarga de estadísticas a los "
                              "event_id tales que event_id %% de_turnos == turno (para repartir el trabajo "
                              "con otras instancias, p.ej. scrape_sofascore_opera.py, sin solaparse)")
    parser.add_argument("--de-turnos", type=int, default=2,
                         help="Divisor usado junto a --turno (por defecto 2)")
    args = parser.parse_args()

    fecha_inicio = datetime(args.fecha_inicio, 1, 1, tzinfo=timezone.utc)
    DIR_RAW.mkdir(parents=True, exist_ok=True)

    selecciones = args.solo_equipos or cargar_selecciones_mundial_2026()
    print(f"{len(selecciones)} selecciones -- descargando desde {args.fecha_inicio}, "
          f"{args.pausa}s de pausa entre peticiones.")

    progreso = Progreso.cargar()
    if progreso.partidos_vistos:
        print(f"Retomando: {len(progreso.partidos_vistos)} partidos ya descargados, se omiten.")

    with sync_playwright() as pw:
        navegador = pw.chromium.launch(headless=not args.con_cabeza)
        page = navegador.new_page()
        try:
            _ejecutar(page, selecciones, fecha_inicio, args.pausa, progreso, args)
        finally:
            navegador.close()


def _ejecutar(page: Page, selecciones: list[str], fecha_inicio: datetime, pausa: float,
              progreso: Progreso, args: argparse.Namespace) -> None:
    eventos_por_id: dict[int, dict] = {}
    ids_equipo: dict[str, int] = {}

    cache_equipos = cargar_cache_equipos(args.fecha_inicio)
    if cache_equipos:
        print(f"{len(cache_equipos)} selecciones ya resueltas en una ejecución anterior, se reutilizan.")

    for i, seleccion in enumerate(selecciones, start=1):
        en_cache = cache_equipos.get(seleccion)
        if en_cache is not None:
            id_equipo, eventos = en_cache["id_equipo"], en_cache["eventos"]
        else:
            id_equipo = resolver_equipo_sofascore(page, seleccion, pausa)
            if id_equipo is None:
                print(f"[{i}/{len(selecciones)}] {seleccion}: NO RESUELTO -- añade un alias en "
                      f"ALIAS_SOFASCORE y vuelve a ejecutar.")
                continue
            eventos = descargar_historico_equipo(page, id_equipo, fecha_inicio, pausa)
            guardar_cache_equipo(seleccion, id_equipo, eventos, args.fecha_inicio)

        ids_equipo[seleccion] = id_equipo
        nuevos = 0
        for evento in eventos:
            if evento["id"] not in eventos_por_id:
                eventos_por_id[evento["id"]] = evento
                nuevos += 1
        marca = " (caché)" if en_cache is not None else ""
        print(f"[{i}/{len(selecciones)}] {seleccion} (id {id_equipo}){marca}: "
              f"{len(eventos)} partidos desde {args.fecha_inicio} ({nuevos} nuevos, "
              f"{len(eventos_por_id)} distintos acumulados).")

    print(f"\n{len(eventos_por_id)} partidos distintos entre las {len(ids_equipo)} selecciones resueltas.")

    filas_partidos: list[dict] = []
    filas_estadisticas: list[dict] = []
    pendientes = [e for e in eventos_por_id.values() if e["id"] not in progreso.partidos_vistos]
    if args.turno is not None:
        pendientes = [e for e in pendientes if e["id"] % args.de_turnos == args.turno]
        print(f"Turno propio (event_id %% {args.de_turnos} == {args.turno}): {len(pendientes)} partidos.")
    print(f"Descargando estadísticas de {len(pendientes)} partidos nuevos "
          f"({len(eventos_por_id) - len(pendientes)} ya estaban)...")

    for i, evento in enumerate(pendientes, start=1):
        id_evento = evento["id"]
        fecha = datetime.fromtimestamp(evento["startTimestamp"], tz=timezone.utc)
        fila_partido = {
            "event_id": id_evento,
            "fecha": fecha.date().isoformat(),
            "torneo": evento.get("tournament", {}).get("name"),
            "equipo_local": evento["homeTeam"]["name"],
            "equipo_visitante": evento["awayTeam"]["name"],
            "goles_local": evento.get("homeScore", {}).get("normaltime"),
            "goles_visitante": evento.get("awayScore", {}).get("normaltime"),
            "estado": evento.get("status", {}).get("type"),
        }
        filas_partidos.append(fila_partido)

        estadisticas = descargar_estadisticas_evento(page, id_evento, pausa)
        if estadisticas is not None:
            fila_stats = {"event_id": id_evento}
            for clave, (valor_local, valor_visitante) in estadisticas.items():
                fila_stats[f"{clave}_local"] = valor_local
                fila_stats[f"{clave}_visitante"] = valor_visitante
            filas_estadisticas.append(fila_stats)

        if i % 25 == 0 or i == len(pendientes):
            _guardar_incremental(filas_partidos, filas_estadisticas)
            print(f"  ... {i}/{len(pendientes)} partidos procesados "
                  f"({len(filas_estadisticas)} con estadísticas) -- guardado parcial.")
            filas_partidos.clear()
            filas_estadisticas.clear()

    _guardar_incremental(filas_partidos, filas_estadisticas)
    print(f"\nHecho. Ver {RUTA_PARTIDOS_SALIDA} y {RUTA_ESTADISTICAS_SALIDA}.")


def _guardar_incremental(filas_partidos: list[dict], filas_estadisticas: list[dict]) -> None:
    """Añade lo nuevo al JSONL (seguro, sin importar qué claves traiga cada fila) y
    regenera el CSV final completo a partir de TODO el JSONL acumulado -- así el CSV
    nunca queda con columnas a medias, y siempre es válido aunque se corte a mitad."""
    _anadir_jsonl(RUTA_PARTIDOS_JSONL, filas_partidos)
    _anadir_jsonl(RUTA_ESTADISTICAS_JSONL, filas_estadisticas)

    if filas_partidos:
        pd.DataFrame(_leer_jsonl(RUTA_PARTIDOS_JSONL)).to_csv(RUTA_PARTIDOS_SALIDA, index=False)
    if filas_estadisticas:
        pd.DataFrame(_leer_jsonl(RUTA_ESTADISTICAS_JSONL)).to_csv(RUTA_ESTADISTICAS_SALIDA, index=False)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrumpido -- lo ya guardado no se pierde, vuelve a ejecutar para retomar.")
        sys.exit(130)
