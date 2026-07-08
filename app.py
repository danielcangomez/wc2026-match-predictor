"""Dashboard interactivo del predictor del Mundial 2026.

Lee SOLO de results/ (los CSV que genera el pipeline) y recalcula todas las métricas en
vivo -- así, cuando el script diario refresca los resultados, el dashboard se actualiza
solo sin tocar una línea de código.

Ejecutar en local:   streamlit run app.py
Desplegar gratis:    subir el repo a GitHub y conectarlo en https://share.streamlit.io
                     (apunta a app.py; no necesita el modelo ni los datos crudos, solo results/).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DIR_RESULTS = Path(__file__).resolve().parent / "results"

AZUL, AQUA, AMBAR, VERDE, GRIS = "#2a78d6", "#1baf7a", "#eda100", "#008300", "#898781"
LAB = np.array(["LOCAL", "EMPATE", "VISITANTE"])

st.set_page_config(page_title="Predictor Mundial 2026", page_icon="⚽", layout="wide")

# --- estilo: tipografía y tarjetas de métrica más limpias ---
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1150px;}
  h1, h2, h3 {font-family: system-ui, -apple-system, "Segoe UI", sans-serif;}
  [data-testid="stMetricValue"] {font-size: 1.9rem; font-weight: 700;}
  [data-testid="stMetricLabel"] {opacity: 0.75;}
  .stTabs [data-baseweb="tab"] {font-size: 1.02rem; font-weight: 600;}
  div[data-testid="stDataFrame"] {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def _leer(ruta_str, _mtime):
    # _mtime forma parte de la clave de caché: si el archivo cambia, su fecha de
    # modificación cambia y el caché se invalida solo -> el dashboard refleja los
    # resultados nuevos con solo recargar la página (sin reiniciar streamlit).
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
mercado = cargar("apuestas_benchmark_mercado.csv")
verif = cargar("verificacion_70ediciones.csv")


def prev_desde_probs(df):
    return LAB[df[["prob_local", "prob_empate", "prob_visitante"]].to_numpy().argmax(1)]


def pct(x):
    return f"{x:.1%}" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "—"


# ---------- Métricas globales (calculadas en vivo) ----------
gj = grupos[grupos["resultado_1x2_real"].notna()].copy() if not grupos.empty else pd.DataFrame()
ok_g = int((gj["resultado_1x2_previsto"] == gj["resultado_1x2_real"]).sum()) if not gj.empty else 0
me_g = int((gj["marcador_previsto"] == gj["marcador_real"]).sum()) if not gj.empty else 0

ej = elim[elim["resultado_1x2_real"].notna()].copy() if not elim.empty else pd.DataFrame()
if not ej.empty:
    ej["prev"] = prev_desde_probs(ej)
    ok_e = int((ej["prev"] == ej["resultado_1x2_real"]).sum())
    me_e = int((ej["marcador_previsto"] == ej["marcador_real_90min"]).sum())
    dec = ej[ej["resultado_1x2_real"] != "EMPATE"]
    adv_ok = int((dec["avanza_previsto"] == np.where(
        dec["resultado_1x2_real"] == "LOCAL", dec["equipo_local"], dec["equipo_visitante"])).sum())
    adv_n = len(dec)
else:
    ok_e = me_e = adv_ok = adv_n = 0

n_tot = len(gj) + len(ej)
ok_tot = ok_g + ok_e
me_tot = me_g + me_e
acc_2026 = ok_tot / n_tot if n_tot else None
exacto_2026 = me_tot / n_tot if n_tot else None

acc_amplio = (verif["acc_adoptado"] * verif["n"]).sum() / verif["n"].sum() if not verif.empty else None
n_amplio = int(verif["n"].sum()) if not verif.empty else 0

# ---------- Cabecera ----------
st.title("⚽ Predictor del Mundial 2026")
st.markdown(
    "Modela los goles de cada equipo (Poisson + Dixon-Coles) y de ahí deriva ganador, marcador "
    "y quién avanza en cada cruce. **Cada partido se predice a ciegas** — solo con datos "
    "anteriores a su fecha, sin fuga temporal.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Acierto 1X2 · 2026", pct(acc_2026), help=f"{ok_tot}/{n_tot} partidos jugados")
c2.metric("Marcador exacto · 2026", pct(exacto_2026), help=f"{me_tot}/{n_tot} partidos")
c3.metric("Quién avanza (90')", pct(adv_ok / adv_n) if adv_n else "—", help=f"{adv_ok}/{adv_n} cruces")
c4.metric("Validación amplia", pct(acc_amplio), help=f"{n_amplio} partidos, 82 ediciones 1990-2026")
c5.metric("Casas de apuestas", "~55-58%", help="Referencia del mercado; el azar es 33%")

st.divider()

# ---------- Pestañas ----------
tab_2026, tab_camp, tab_val = st.tabs(
    ["📅 Predicciones 2026", "🏆 Camino al título", "🔬 Rigor y validación"])

# ===== TAB 1: PREDICCIONES 2026 =====
with tab_2026:
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
        # filtro rápido: acertados / fallados / todos
        col_f, col_c = st.columns([1, 3])
        filtro = col_f.radio("Ver", ["Todos", "Solo aciertos", "Solo fallos"], label_visibility="collapsed")
        j = gj.copy()
        j["acierto"] = j["resultado_1x2_previsto"] == j["resultado_1x2_real"]
        if filtro == "Solo aciertos":
            j = j[j["acierto"]]
        elif filtro == "Solo fallos":
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
            use_container_width=True, hide_index=True, height=380)

    if not ej.empty:
        st.divider()
        st.subheader("🥊 Eliminatorias: cada cruce, con su desglose 90'/prórroga/penaltis")
        k = ej.copy()
        k["P. clasifica local"] = (k["prob_clasifica_local"] * 100).round(0).astype(int).astype(str) + "%"
        k["Prórroga"] = (k["prob_decide_prorroga"] * 100).round(0).astype(int).astype(str) + "%"
        k["Penaltis"] = (k["prob_decide_penaltis"] * 100).round(0).astype(int).astype(str) + "%"
        k["✓"] = np.where(k["prev"] == k["resultado_1x2_real"], "✅", "❌")
        st.dataframe(
            k[["equipo_local", "equipo_visitante", "marcador_previsto", "marcador_real_90min",
               "avanza_previsto", "P. clasifica local", "Prórroga", "Penaltis", "✓"]]
            .rename(columns={"equipo_local": "Local", "equipo_visitante": "Visitante",
                             "marcador_previsto": "Marc. prev.", "marcador_real_90min": "Marc. real 90'",
                             "avanza_previsto": "Avanza"}),
            use_container_width=True, hide_index=True)
        st.caption("Si el cruce llega empatado a los 90', el modelo reparte la probabilidad entre "
                   "prórroga (33%) y penaltis (67%) — estos últimos, una moneda al aire.")

# ===== TAB 2: CAMINO AL TÍTULO =====
with tab_camp:
    st.subheader("🏆 Probabilidad de ser campeón (Montecarlo, 3.000 simulaciones)")
    if not montecarlo.empty:
        m = montecarlo.sort_values("campeon", ascending=False).reset_index(drop=True)
        top = m.head(10).set_index("seleccion")["campeon"]
        st.bar_chart(top, color=AZUL, height=380, horizontal=True)

        st.markdown("**Probabilidad de alcanzar cada ronda** (top 12):")
        tabla = m.head(12).copy()
        for col in ["alcanza_octavos", "alcanza_cuartos", "alcanza_semis", "alcanza_final", "campeon"]:
            tabla[col] = (tabla[col] * 100).round(1).astype(str) + "%"
        st.dataframe(
            tabla.rename(columns={"seleccion": "Selección", "alcanza_octavos": "Octavos",
                                  "alcanza_cuartos": "Cuartos", "alcanza_semis": "Semis",
                                  "alcanza_final": "Final", "campeon": "🏆 Campeón"}),
            use_container_width=True, hide_index=True)
        st.caption("Incorpora todos los resultados ya jugados; solo simula lo que queda por decidir. "
                   "Se re-simula con el modelo reentrenado hasta el último resultado conocido.")

# ===== TAB 3: RIGOR =====
with tab_val:
    st.subheader("¿Por qué fiarse de estos números?")
    st.markdown(
        "El mayor error en predicción deportiva es la **fuga temporal**: entrenar con partidos "
        "que en la práctica son posteriores a los que evalúas, lo que infla la precisión de forma "
        "artificial (un modelo con *split aleatorio* puede aparentar 70%+ sin que sea real). "
        "Aquí cada partido se predice **solo con datos anteriores a su fecha**, y cada mejora se "
        "valida sobre **82 ediciones de torneos mayores (2.677 partidos, 1990-2026)** — no sobre "
        "un solo torneo, donde el ruido supera a la señal.")

    cola, colb = st.columns(2)
    with cola:
        st.metric("Accuracy 1X2 · validación amplia", pct(acc_amplio))
        if not verif.empty:
            ll = (verif["logloss_adoptado"] * verif["n"]).sum() / verif["n"].sum()
            ex = (verif["exacto_adoptado"] * verif["n"]).sum() / verif["n"].sum()
            st.metric("LogLoss", f"{ll:.3f}")
            st.metric("Marcador exacto", pct(ex))
    with colb:
        st.markdown("**Contexto de las cifras**")
        st.markdown(
            "- **Azar puro:** 33%\n"
            "- **Casas de apuestas:** ~55-58%\n"
            f"- **Este modelo (2026, a ciegas):** {pct(acc_2026)}\n"
            f"- **Este modelo (2.677 partidos):** {pct(acc_amplio)}")

    if not mercado.empty:
        st.divider()
        st.subheader("El benchmark más duro: contra el mercado de apuestas")
        mm = mercado.copy()
        mm["Mundial"] = mm["anio"].astype(str)
        mm = mm.rename(columns={"n": "Partidos", "logloss_modelo": "LogLoss modelo",
                                "logloss_mercado": "LogLoss mercado"})
        st.dataframe(mm[["Mundial", "Partidos", "LogLoss modelo", "LogLoss mercado"]].round(3),
                     use_container_width=True, hide_index=True)
        st.caption("Las cuotas de cierre integran alineaciones, lesiones y el dinero de millones "
                   "de apostantes — son el pronóstico más eficiente que existe, y ganan al modelo "
                   "en los 4 Mundiales. Se muestra sin adornos: un techo honesto vale más que un "
                   "récord inflado. Menor LogLoss = mejor.")

st.divider()
st.caption("Metodología completa en los 6 notebooks del repositorio. "
           "Datos refrescados por `scripts/actualizar_diario.sh`. "
           "Todas las métricas se recalculan en vivo desde `results/`.")
