"""Descarga el valor de mercado de las selecciones participantes en un Mundial,
desde la página de participantes de Transfermarkt -- para el Mundial que se indique
(no solo 2026), usando el `saison_id` (temporada) correspondiente.

Por qué esto SÍ puede dar histórico real: la página "teilnehmer" (participantes) de
cada edición del Mundial muestra el valor de plantilla de cada selección TAL COMO
ERA en esa temporada -- si `saison_id` acepta años pasados, esto evita el problema
de que Transfermarkt no trackee un histórico agregado por selección: no hace falta
reconstruirlo jugador a jugador, la propia página del torneo ya lo tiene por edición.
Hay que comprobarlo edición a edición: no hay garantía de que `saison_id` seleccione
el año que uno espera (la convención de Transfermarkt es la temporada de INICIO, p.ej.
"2025" es la temporada 2025/26 -- probable candidato para un Mundial de junio 2026,
pero para Mundiales de años pares anteriores (2010, 2014, 2018, 2022) el año exacto
hay que verificarlo mirando el resultado.

IMPORTANTE -- por qué este script está pensado para que lo ejecutes TÚ, no Claude:
el robots.txt de transfermarkt.com prohíbe explícitamente a ClaudeBot y otros
crawlers de IA (GPTBot, CCBot...). Ejecutándolo tú mismo en tu terminal, eres tú
quien hace las peticiones, no un agente de IA -- por eso este script no se lanza
automáticamente.

Uso:
    python scripts/scrape_transfermarkt_valor.py                  # Mundial 2026 (saison_id=2025)
    python scripts/scrape_transfermarkt_valor.py --saison-id 2021 # probar candidato para Mundial 2022
    python scripts/scrape_transfermarkt_valor.py --saison-id 2017 --anio-mundial 2018
    python scripts/scrape_transfermarkt_valor.py --saison-id 2013 --anio-mundial 2014
    python scripts/scrape_transfermarkt_valor.py --saison-id 2009 --anio-mundial 2010

Salida:
    data/raw/transfermarkt_valor_<anio_mundial>.csv
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

DIR_RAIZ = Path(__file__).resolve().parent.parent

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"}


def limpiar_valor_mercado(valor_str: str) -> float:
    """Convierte el texto de Transfermarkt ("1,22 mil mill.", "450 mill."...) a
    millones de euros como float."""
    v = valor_str.replace("€", "").strip()
    if not v or v == "-":
        return 0.0
    v = v.replace(",", ".")
    try:
        if "mil mill." in v:
            return float(v.replace("mil mill.", "").strip()) * 1000
        if "mill." in v:
            return float(v.replace("mill.", "").strip())
        if "mil" in v:
            return float(v.replace("mil", "").strip()) / 1000
        return float(v)
    except ValueError:
        return 0.0


def extraer_transfermarkt(saison_id: int, ruta_salida: Path) -> None:
    base_url = f"https://www.transfermarkt.es/weltmeisterschaft/teilnehmer/pokalwettbewerb/FIWC/saison_id/{saison_id}/page/"
    datos = []
    pagina = 1
    equipos_pagina_anterior: list[str] = []

    print(f"Iniciando extracción (saison_id={saison_id})...")
    while True:
        url = f"{base_url}{pagina}"
        respuesta = requests.get(url, headers=HEADERS, timeout=15)
        if respuesta.status_code != 200:
            print(f"  [aviso] HTTP {respuesta.status_code} en página {pagina} -- fin.")
            break
        soup = BeautifulSoup(respuesta.content, "html.parser")
        tablas = soup.find_all("table", class_="items")
        if not tablas:
            break

        equipos_pagina_actual = []
        for tabla in tablas:
            filas = tabla.find("tbody").find_all("tr")
            for fila in filas:
                columnas = fila.find_all("td")
                if len(columnas) < 5:
                    continue
                celda_nombre = fila.find("td", class_="hauptlink")
                if not celda_nombre:
                    continue
                equipo = celda_nombre.text.strip()
                valor_raw = columnas[-1].text.strip()
                equipos_pagina_actual.append(equipo)
                datos.append({"Equipo": equipo, "Valor_Mercado_Millones_Eur": limpiar_valor_mercado(valor_raw)})

        if equipos_pagina_actual == equipos_pagina_anterior:
            break
        equipos_pagina_anterior = equipos_pagina_actual
        pagina += 1
        time.sleep(1.5)

    if not datos:
        print(f"  [aviso] 0 filas para saison_id={saison_id} -- ese año probablemente no existe/no aplica "
              f"para este torneo en Transfermarkt. Prueba con otro saison_id cercano.")
        return

    df = pd.DataFrame(datos).drop_duplicates(subset="Equipo")
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_salida, index=False, encoding="utf-8-sig")
    print(f"Guardado: {len(df)} selecciones en {ruta_salida}")
    print(df.sort_values("Valor_Mercado_Millones_Eur", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saison-id", type=int, default=2025,
                         help="Temporada de Transfermarkt a consultar (año de INICIO de temporada; "
                              "2025 -> Mundial 2026 ya comprobado que funciona)")
    parser.add_argument("--anio-mundial", type=int, default=2026,
                         help="Solo para nombrar el fichero de salida")
    args = parser.parse_args()

    ruta_salida = DIR_RAIZ / "data" / "raw" / f"transfermarkt_valor_{args.anio_mundial}.csv"
    extraer_transfermarkt(args.saison_id, ruta_salida)
