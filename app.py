"""Dashboard interactivo — Predicciones Machine Learning del Mundial 2026.

Lee SOLO de results/ y recalcula todas las métricas y clasificaciones en vivo, así que
cuando el script diario refresca los resultados el dashboard se actualiza solo (recargando).

Ejecutar en local:   streamlit run app.py
Desplegar gratis:    subir el repo a GitHub y conectarlo en https://share.streamlit.io
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DIR_RESULTS = Path(__file__).resolve().parent / "results"

AZUL, AQUA, AMBAR, VERDE, GRIS = "#2a78d6", "#1baf7a", "#eda100", "#008300", "#898781"
LAB = np.array(["LOCAL", "EMPATE", "VISITANTE"])

st.set_page_config(page_title="Predicciones ML · Mundial 2026", page_icon="🤖", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1180px;}
  h1, h2, h3 {font-family: system-ui, -apple-system, "Segoe UI", sans-serif;}
  [data-testid="stMetricValue"] {font-size: 2rem; font-weight: 700; color: #2a78d6;}
  [data-testid="stMetricLabel"] {opacity: 0.8; font-weight: 600;}
  .stTabs [data-baseweb="tab"] {font-size: 1.03rem; font-weight: 600;}
  div[data-testid="stDataFrame"] {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def _leer(ruta_str, _mtime):
    ruta = Path(ruta_str)
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


def cargar(nombre):
    ruta = DIR_RESULTS / nombre
    mtime = ruta.stat().st_mtime if ruta.exists() else 0
    return _leer(str(ruta), mtime)


grupos = cargar("predicciones_fase_grupos.csv")
elim = cargar("predicciones_eliminatoria.csv")
proximos = cargar("predicciones_proximos_partidos.csv")
montecarlo = cargar("simulacion_probabilidades_actual.csv")
cuadro = cargar("cuadro_completo.csv")
mapa_grupos = cargar("grupos_2026.csv")


def prev_desde_probs(df):
    return LAB[df[["prob_local", "prob_empate", "prob_visitante"]].to_numpy().argmax(1)]


def pct(x):
    return f"{x:.1%}" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "—"


def parse_marcador(s):
    try:
        a, b = str(s).split("-")
        return int(a), int(b)
    except Exception:
        return None, None


# ---------- Métricas 2026 (en vivo) ----------
gj = grupos[grupos["resultado_1x2_real"].notna()].copy() if not grupos.empty else pd.DataFrame()
ok_g = int((gj["resultado_1x2_previsto"] == gj["resultado_1x2_real"]).sum()) if not gj.empty else 0
me_g = int((gj["marcador_previsto"] == gj["marcador_real"]).sum()) if not gj.empty else 0

ej = elim[elim["resultado_1x2_real"].notna()].copy() if not elim.empty else pd.DataFrame()
if not ej.empty:
    ej["prev"] = prev_desde_probs(ej)
    ok_e = int((ej["prev"] == ej["resultado_1x2_real"]).sum())
    me_e = int((ej["marcador_previsto"] == ej["marcador_real_90min"]).sum())
else:
    ok_e = me_e = 0

n_1x2 = len(gj) + len(ej)
acc_1x2 = (ok_g + ok_e) / n_1x2 if n_1x2 else None
exacto = (me_g + me_e) / n_1x2 if n_1x2 else None

# Quién avanza a partido completo (incluye prórroga y penaltis): elim vs cuadro real
avanza_ok = avanza_n = 0
if not elim.empty and not cuadro.empty:
    cc = cuadro[cuadro["jugado"] == True]
    j = elim.merge(cc[["equipo_a", "equipo_b", "ganador"]],
                   left_on=["equipo_local", "equipo_visitante"],
                   right_on=["equipo_a", "equipo_b"], how="inner")
    avanza_ok = int((j["avanza_previsto"] == j["ganador"]).sum())
    avanza_n = len(j)

# ---------- Cabecera ----------
st.title("🤖 Predicciones Machine Learning · Mundial 2026")
st.markdown(
    "Un modelo que predice los goles de cada selección (Poisson + Dixon-Coles) y de ahí deriva "
    "el ganador, el marcador y quién avanza en cada eliminatoria. **Cada partido se predice a "
    "ciegas**, usando solo datos anteriores a su fecha.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Acierto 1X2 (90 min)", pct(acc_1x2), help=f"{ok_g + ok_e}/{n_1x2} partidos jugados")
c2.metric("Quién avanza (partido completo)", pct(avanza_ok / avanza_n) if avanza_n else "—",
          help=f"{avanza_ok}/{avanza_n} eliminatorias, incluye prórroga y penaltis")
c3.metric("Marcador exacto", pct(exacto), help=f"{me_g + me_e}/{n_1x2} partidos")
c4.metric("Referencia mercado", "~55-58%", help="Casas de apuestas; el azar puro es 33%")

st.divider()

tab_clasif, tab_pred, tab_camp = st.tabs(
    ["📊 Clasificaciones previstas", "📅 Predicción partido a partido", "🏆 Camino al título"])


# ===== TAB 1: CLASIFICACIONES PREVISTAS =====
def calcular_clasificacion(df_grupos, mapa):
    """Tabla de cada grupo a partir de los marcadores PREVISTOS por el modelo,
    con un marcador de clasificación: 🟢 primeros/segundos (pasan directos),
    🟡 los 8 mejores terceros (también clasifican en el formato de 48)."""
    g2g = dict(zip(mapa["equipo"], mapa["grupo"]))
    filas = {}
    for _, r in df_grupos.iterrows():
        gl, gv = parse_marcador(r["marcador_previsto"])
        if gl is None:
            continue
        for eq, gf, gc in [(r["equipo_local"], gl, gv), (r["equipo_visitante"], gv, gl)]:
            grp = g2g.get(eq)
            if grp is None:
                continue
            d = filas.setdefault((grp, eq), {"Grupo": grp, "Selección": eq, "PJ": 0,
                                             "Pts": 0, "GF": 0, "GC": 0})
            d["PJ"] += 1
            d["GF"] += gf
            d["GC"] += gc
            d["Pts"] += 3 if gf > gc else (1 if gf == gc else 0)
    tabla = pd.DataFrame(filas.values())
    if tabla.empty:
        return tabla
    tabla["DG"] = tabla["GF"] - tabla["GC"]
    tabla = tabla.sort_values(["Grupo", "Pts", "DG", "GF"], ascending=[True, False, False, False])
    tabla["Pos"] = tabla.groupby("Grupo").cumcount() + 1

    # 8 mejores terceros entre los 12 grupos (mismo criterio: Pts, DG, GF)
    terceros = tabla[tabla["Pos"] == 3].sort_values(["Pts", "DG", "GF"], ascending=False)
    mejores_terceros = set(terceros.head(8)["Selección"])

    def marca(row):
        if row["Pos"] <= 2:
            return "🟢"
        if row["Selección"] in mejores_terceros:
            return "🟡"
        return ""
    tabla[""] = tabla.apply(marca, axis=1)
    return tabla


with tab_clasif:
    st.subheader("Clasificación prevista de cada grupo")
    st.caption("Construida con los marcadores que predice el modelo para los 72 partidos de grupos. "
               "🟢 = los 2 primeros de cada grupo (pasan directos)  ·  🟡 = uno de los 8 mejores "
               "terceros (también clasifican, en el formato de 48 selecciones).")
    if not grupos.empty and not mapa_grupos.empty:
        clasif = calcular_clasificacion(grupos, mapa_grupos)
        letras = sorted(clasif["Grupo"].unique())
        for i in range(0, len(letras), 3):
            cols = st.columns(3)
            for col, letra in zip(cols, letras[i:i + 3]):
                sub = clasif[clasif["Grupo"] == letra][["", "Selección", "PJ", "Pts", "DG", "GF"]].copy()
                sub["DG"] = sub["DG"].map(lambda v: f"{v:+d}")
                col.markdown(f"**Grupo {letra}**")
                col.dataframe(sub, use_container_width=True, hide_index=True)
    else:
        st.info("Faltan datos de grupos o el mapa de grupos (results/grupos_2026.csv).")


# ===== TAB 2: PREDICCIÓN PARTIDO A PARTIDO =====
with tab_pred:
    if not proximos.empty:
        st.subheader("🔮 Próximos partidos (aún sin jugar)")
        p = proximos.copy()
        for c in ["prob_local", "prob_empate", "prob_visitante"]:
            p[c] = (p[c] * 100).round(0).astype(int)
        st.dataframe(
            p[["fecha", "equipo_local", "equipo_visitante", "marcador_previsto",
               "prob_local", "prob_empate", "prob_visitante", "avanza_previsto"]]
            .rename(columns={"fecha": "Fecha", "equipo_local": "Local", "equipo_visitante": "Visitante",
                             "marcador_previsto": "Marcador prev.", "prob_local": "1 (%)",
                             "prob_empate": "X (%)", "prob_visitante": "2 (%)", "avanza_previsto": "Avanza"}),
            use_container_width=True, hide_index=True)
        st.divider()

    st.subheader("📋 Fase de grupos: previsto vs. real")
    if not gj.empty:
        col_f, col_c = st.columns([1, 3])
        filtro = col_f.radio("Ver", ["Todos", "Aciertos", "Fallos"], label_visibility="collapsed")
        j = gj.copy()
        j["acierto"] = j["resultado_1x2_previsto"] == j["resultado_1x2_real"]
        if filtro == "Aciertos":
            j = j[j["acierto"]]
        elif filtro == "Fallos":
            j = j[~j["acierto"]]
        j["1X2"] = np.where(j["acierto"], "✅", "❌")
        j["Marcador"] = np.where(j["marcador_previsto"] == j["marcador_real"], "🎯", "")
        col_c.caption(f"Acierto 1X2: **{ok_g}/{len(gj)} = {ok_g/len(gj):.1%}**  ·  "
                      f"Marcador exacto: **{me_g}/{len(gj)} = {me_g/len(gj):.1%}**")
        st.dataframe(
            j[["fecha", "equipo_local", "equipo_visitante", "marcador_previsto",
               "marcador_real", "1X2", "Marcador"]]
            .rename(columns={"fecha": "Fecha", "equipo_local": "Local", "equipo_visitante": "Visitante",
                             "marcador_previsto": "Marcador prev.", "marcador_real": "Marcador real"}),
            use_container_width=True, hide_index=True, height=360)

    if not ej.empty:
        st.divider()
        st.subheader("🥊 Eliminatorias: cada cruce y su desglose")
        k = ej.copy()
        k["P. clasifica local"] = (k["prob_clasifica_local"] * 100).round(0).astype(int).astype(str) + "%"
        k["Prórroga"] = (k["prob_decide_prorroga"] * 100).round(0).astype(int).astype(str) + "%"
        k["Penaltis"] = (k["prob_decide_penaltis"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(
            k[["equipo_local", "equipo_visitante", "marcador_previsto", "marcador_real_90min",
               "avanza_previsto", "P. clasifica local", "Prórroga", "Penaltis"]]
            .rename(columns={"equipo_local": "Local", "equipo_visitante": "Visitante",
                             "marcador_previsto": "Marc. prev.", "marcador_real_90min": "Marc. real 90'",
                             "avanza_previsto": "Avanza (pred.)"}),
            use_container_width=True, hide_index=True)
        st.caption("Si el cruce llega empatado a los 90', el modelo reparte la probabilidad entre "
                   "prórroga y penaltis (medido en 296 cruces reales: 33% / 67%).")


# ===== TAB 3: CAMINO AL TÍTULO =====
with tab_camp:
    st.subheader("🏆 Probabilidad de ser campeón (Montecarlo, 3.000 simulaciones)")
    if not montecarlo.empty:
        m = montecarlo.sort_values("campeon", ascending=False).reset_index(drop=True)
        st.bar_chart(m.head(10).set_index("seleccion")["campeon"], color=AZUL, height=380, horizontal=True)
        st.markdown("**Probabilidad de alcanzar cada ronda** (top 12):")
        t = m.head(12).copy()
        for col in ["alcanza_octavos", "alcanza_cuartos", "alcanza_semis", "alcanza_final", "campeon"]:
            t[col] = (t[col] * 100).round(1).astype(str) + "%"
        st.dataframe(
            t.rename(columns={"seleccion": "Selección", "alcanza_octavos": "Octavos",
                              "alcanza_cuartos": "Cuartos", "alcanza_semis": "Semis",
                              "alcanza_final": "Final", "campeon": "🏆 Campeón"}),
            use_container_width=True, hide_index=True)
        st.caption("Incorpora todos los resultados ya jugados; solo simula lo que queda por decidir, "
                   "con el modelo reentrenado hasta el último resultado conocido.")

st.divider()
st.caption("Metodología completa en los 6 notebooks del repositorio. Todas las métricas y "
           "clasificaciones se recalculan en vivo desde `results/` — refrescables con "
           "`scripts/actualizar_diario.sh`.")
