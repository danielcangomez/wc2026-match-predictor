"""Tests de las funciones más delicadas del pipeline, extraídas en caliente de
los propios notebooks (no reimplementadas aparte) para no divergir del código
que de verdad se ejecuta. Cada test carga solo las funciones puras que
necesita (nunca las celdas que entrenan modelos o leen `partidos_features.csv`
completo), así que corren en milisegundos.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

RAIZ = Path(__file__).parent.parent
NOTEBOOK_01 = RAIZ / "notebooks" / "01_adquisicion_datos.ipynb"
NOTEBOOK_04 = RAIZ / "notebooks" / "04_modelado_optuna_walkforward.ipynb"


def _extraer_funciones_de_celda(ruta_notebook: Path, contiene: str) -> str:
    """Código fuente de todas las funciones definidas en la primera celda de
    un notebook cuyo texto contenga `contiene`. Descarta cualquier sentencia
    de nivel superior que no sea `def`/import (típicamente las llamadas de
    entrenamiento o evaluación de esa misma celda, que dependen de datos que
    un test no quiere cargar).
    """
    notebook = json.loads(ruta_notebook.read_text())
    for celda in notebook["cells"]:
        if celda["cell_type"] != "code":
            continue
        fuente = "".join(celda["source"])
        if contiene in fuente:
            arbol = ast.parse(fuente)
            nodos = [n for n in arbol.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
            return ast.unparse(ast.Module(body=nodos, type_ignores=[]))
    raise ValueError(f"No se encontró ninguna celda con {contiene!r} en {ruta_notebook.name}")


@pytest.fixture
def espacio_matriz_y_resolver():
    """Namespace con `matriz_probabilidad_conjunta` y `resolver_eliminatoria`
    tal como están en el Notebook 4, con rho de Dixon-Coles en 0 -- así los
    tests de la moda de Poisson no se confunden con la corrección de 4.5."""
    def _construir(rho: float = 0.0, max_goles: int = 8):
        codigo = _extraer_funciones_de_celda(NOTEBOOK_04, "def matriz_probabilidad_conjunta")
        ns = {"np": np, "pd": pd, "poisson": poisson, "MAX_GOLES_MATRIZ": max_goles, "RHO_DIXON_COLES": rho}
        exec(codigo, ns)
        return ns
    return _construir


class TestModaPoisson:
    """El bug real: round(lambda) no es la moda de una Poisson(lambda) -- lo
    es siempre floor(lambda). Ver conversación: se descubrió porque casi
    ningún marcador previsto mostraba un 0."""

    def test_lambda_entre_0_5_y_1_predice_cero_no_uno(self, espacio_matriz_y_resolver):
        ns = espacio_matriz_y_resolver(rho=0.0)
        resultado = ns["resolver_eliminatoria"]("A", "B", lam_a=2.0, lam_b=0.81)
        assert resultado["marcador_previsto"] == "2-0"  # round(0.81) daría "2-1"

    def test_lambda_justo_por_encima_de_un_entero_no_redondea_hacia_arriba(self, espacio_matriz_y_resolver):
        ns = espacio_matriz_y_resolver(rho=0.0)
        resultado = ns["resolver_eliminatoria"]("A", "B", lam_a=1.63, lam_b=1.01)
        assert resultado["marcador_previsto"] == "1-1"  # round(1.63) daría "2-1"


class TestResolverSiemprePorProbabilidad:
    """`resolver_eliminatoria` debe decidir SIEMPRE comparando la matriz de
    probabilidad conjunta -- nunca comparando marcadores puntuales, ni
    siquiera cuando esos marcadores puntuales (floor) coinciden."""

    def test_decision_coincide_con_la_matriz_aunque_los_floors_empaten(self, espacio_matriz_y_resolver):
        ns = espacio_matriz_y_resolver(rho=0.0)
        # floor(1.9) == floor(1.6) == 1: "empatados" bajo cualquier criterio de
        # punto, pero A tiene una lambda claramente mayor.
        resultado = ns["resolver_eliminatoria"]("A", "B", lam_a=1.9, lam_b=1.6)
        matriz = ns["matriz_probabilidad_conjunta"](1.9, 1.6).values
        prob_a, prob_b = np.tril(matriz, -1).sum(), np.triu(matriz, 1).sum()

        assert prob_a > prob_b
        assert resultado["ganador"] == "A"

    def test_favorito_claro_gana_pese_a_no_estar_en_la_rejilla_de_marcadores_bajos(self, espacio_matriz_y_resolver):
        ns = espacio_matriz_y_resolver(rho=0.0)
        resultado = ns["resolver_eliminatoria"]("Argentina", "Cabo Verde", lam_a=3.8, lam_b=0.42)
        assert resultado["ganador"] == "Argentina"


class TestDixonColes:
    """La corrección de Dixon-Coles solo debe mover rho lejos de 0 cuando hay
    de verdad correlación negativa en los marcadores bajos -- con datos
    generados de forma independiente, rho ajustado debe quedarse cerca de 0."""

    def test_rho_cercano_a_cero_con_goles_independientes(self):
        codigo = _extraer_funciones_de_celda(NOTEBOOK_04, "def log_verosimilitud_dixon_coles")
        ns = {"np": np, "pd": pd, "poisson": poisson}
        exec(codigo, ns)

        rng = np.random.default_rng(0)
        lam = np.full(8000, 1.8)
        mu = np.full(8000, 1.1)
        x, y = rng.poisson(lam), rng.poisson(mu)

        candidatos_rho = np.linspace(-0.3, 0.3, 121)
        log_verosimilitudes = [ns["log_verosimilitud_dixon_coles"](r, lam, mu, x, y) for r in candidatos_rho]
        rho_ajustado = candidatos_rho[np.argmax(log_verosimilitudes)]

        assert abs(rho_ajustado) < 0.05


class TestNormalizacionLocalVisitante:
    """El JSON del Mundial no siempre lista al anfitrión como `team1` -- ver
    el caso real Chequia/México (sección 1.5 del Notebook 1)."""

    def _cargar(self):
        codigo = _extraer_funciones_de_celda(NOTEBOOK_01, "def _determinar_local_visitante")
        ns: dict = {"Path": Path, "pd": pd}
        exec(codigo, ns)
        return ns

    def test_anfitrion_es_local_aunque_el_json_lo_liste_segundo(self):
        ns = self._cargar()
        local, visitante = ns["_determinar_local_visitante"]("Czech Republic", "Mexico", "Mexico")
        assert (local, visitante) == ("Mexico", "Czech Republic")

    def test_partido_neutral_respeta_el_orden_del_json(self):
        ns = self._cargar()
        local, visitante = ns["_determinar_local_visitante"]("Qatar", "Switzerland", "United States")
        assert (local, visitante) == ("Qatar", "Switzerland")


class TestDetectarFaseGrupos:
    def test_separa_un_grupo_de_4_de_un_cruce_de_eliminatoria(self):
        codigo = _extraer_funciones_de_celda(NOTEBOOK_04, "def detectar_fase_grupos")
        ns = {"pd": pd, "nx": __import__("networkx")}
        exec(codigo, ns)

        equipos_grupo = ["A", "B", "C", "D"]
        partidos_grupo = [
            {"equipo_local": a, "equipo_visitante": b}
            for i, a in enumerate(equipos_grupo)
            for b in equipos_grupo[i + 1:]
        ]  # round-robin completo: 6 partidos
        partidos_eliminatoria = [{"equipo_local": "A", "equipo_visitante": "Z"}]
        df_mundial = pd.DataFrame(partidos_grupo + partidos_eliminatoria)

        df_grupos, df_elim = ns["detectar_fase_grupos"](df_mundial)

        assert len(df_grupos) == 6
        assert len(df_elim) == 1
        assert df_elim.iloc[0]["equipo_visitante"] == "Z"
