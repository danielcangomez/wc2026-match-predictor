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

# Paleta (misma del notebook, validada para daltonismo)
AZUL, AQUA, VERDE, GRIS, ROJO = "#2a78d6", "#1baf7a", "#008300", "#898781", "#e34948"
LAB = np.array(["LOCAL", "EMPATE", "VISITANTE"])

st.set_page_config(page_title="Predictor Mundial 2026", page_icon="⚽", layout="wide")


@st.cache_data
def cargar(nombre):
    ruta = DIR_RESULTS / nombre
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


grupos = cargar("predicciones_fase_grupos.csv")
elim = cargar("predicciones_eliminatoria.csv")
proximos = cargar("predicciones_proximos_partidos.csv")
montecarlo = cargar("simulacion_probabilidades_actual.csv")
mercado = cargar("apuestas_benchmark_mercado.csv")
verif = cargar("verificacion_70ediciones.csv")


def acc_1x2_grupos(df):
    j = df[df["resultado_1x2_real"].notna()]
    if j.empty:
        return None, 0, 0
    ok = (j["resultado_1x2_previsto"] == j["resultado_1x2_real"]).sum()
    return ok / len(j), ok, len(j)


def prev_desde_probs(df):
    return LAB[df[["prob_local", "prob_empate", "prob_visitante"]].to_numpy().argmax(1)]


# ---------- Cabecera y métricas ----------
st.title("⚽ Predictor del Mundial 2026")
st.markdown(
    "Sistema de predicción probabilística: modela los goles de cada equipo (Poisson + "
    "Dixon-Coles) y de ahí deriva ganador, marcador y quién avanza en cada cruce. "
    "**Toda cifra de precisión es a ciegas** — cada partido se predice solo con datos "
    "anteriores a él (sin fuga temporal)."
)

# Métricas 2026
acc_g, ok_g, n_g = acc_1x2_grupos(grupos) if not grupos.empty else (None, 0, 0)
ej = elim[elim["resultado_1x2_real"].notna()].copy() if not elim.empty else pd.DataFrame()
if not ej.empty:
    ej["prev"] = prev_desde_probs(ej)
    ok_e = (ej["prev"] == ej["resultado_1x2_real"]).sum()
else:
    ok_e = 0
ok_tot, n_tot = ok_g + ok_e, n_g + len(ej)

# Backtest amplio
if not verif.empty:
    acc_amplio = (verif["acc_adoptado"] * verif["n"]).sum() / verif["n"].sum()
    n_amplio = int(verif["n"].sum())
else:
    acc_amplio, n_amplio = None, 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Acierto 1X2 en 2026", f"{ok_tot/n_tot:.1%}" if n_tot else "—",
          help=f"{ok_tot}/{n_tot} partidos jugados, a ciegas")
c2.metric("Validación amplia", f"{acc_amplio:.1%}" if acc_amplio else "—",
          help=f"{n_amplio} partidos de 82 ediciones de torneos mayores (1990-2026)")
c3.metric("Referencia: azar", "33%")
c4.metric("Referencia: casas de apuestas", "~55-58%")

st.divider()

# ---------- Pestañas ----------
tab_2026, tab_camp, tab_val = st.tabs(
    ["📅 Predicciones 2026", "🏆 Quién será campeón", "🔬 Rigor y validación"])

with tab_2026:
    if not proximos.empty:
        st.subheader("Próximos partidos (aún sin jugar)")
        p = proximos.copy()
        p["Local (%)"] = (p["prob_local"] * 100).round(0)
        p["Empate (%)"] = (p["prob_empate"] * 100).round(0)
        p["Visitante (%)"] = (p["prob_visitante"] * 100).round(0)
        st.dataframe(
            p[["fecha", "equipo_local", "equipo_visitante", "marcador_previsto",
               "Local (%)", "Empate (%)", "Visitante (%)", "avanza_previsto"]]
            .rename(columns={"fecha": "Fecha", "equipo_local": "Local",
                             "equipo_visitante": "Visitante", "marcador_previsto": "Marcador previsto",
                             "avanza_previsto": "Avanza"}),
            use_container_width=True, hide_index=True)

    st.subheader("Fase de grupos: previsto vs. real")
    if not grupos.empty:
        j = grupos[grupos["resultado_1x2_real"].notna()].copy()
        j["✓"] = np.where(j["resultado_1x2_previsto"] == j["resultado_1x2_real"], "✅", "❌")
        j["Marcador ✓"] = np.where(j["marcador_previsto"] == j["marcador_real"], "✅", "")
        st.dataframe(
            j[["fecha", "equipo_local", "equipo_visitante", "marcador_previsto",
               "marcador_real", "resultado_1x2_previsto", "resultado_1x2_real", "✓", "Marcador ✓"]]
            .rename(columns={"fecha": "Fecha", "equipo_local": "Local", "equipo_visitante": "Visitante",
                             "marcador_previsto": "Marcador prev.", "marcador_real": "Marcador real",
                             "resultado_1x2_previsto": "1X2 prev.", "resultado_1x2_real": "1X2 real"}),
            use_container_width=True, hide_index=True, height=350)
        me = (j["marcador_previsto"] == j["marcador_real"]).mean()
        st.caption(f"Acierto 1X2 grupos: **{acc_g:.1%}** ({ok_g}/{n_g})  ·  "
                   f"Marcador exacto: **{me:.1%}**")

with tab_camp:
    st.subheader("Probabilidad de cada selección (Montecarlo, 3.000 simulaciones)")
    if not montecarlo.empty:
        m = montecarlo.sort_values("campeon", ascending=False).head(12).copy()
        st.bar_chart(m.set_index("seleccion")["campeon"], color=AZUL, height=400)
        for col in ["alcanza_octavos", "alcanza_cuartos", "alcanza_semis", "alcanza_final", "campeon"]:
            m[col] = (m[col] * 100).round(1).astype(str) + "%"
        st.dataframe(
            m.rename(columns={"seleccion": "Selección", "alcanza_octavos": "Octavos",
                              "alcanza_cuartos": "Cuartos", "alcanza_semis": "Semis",
                              "alcanza_final": "Final", "campeon": "Campeón"}),
            use_container_width=True, hide_index=True)
        st.caption("Incorpora todos los resultados ya jugados; solo simula lo que queda por decidir.")

with tab_val:
    st.subheader("¿Por qué creer estos números?")
    st.markdown(
        "El mayor error en predicción deportiva es la **fuga temporal**: entrenar con partidos "
        "que en la práctica son posteriores a los que evalúas, lo que infla la precisión "
        "artificialmente. Aquí cada partido se predice **solo con datos anteriores a su fecha**, "
        "y cada mejora se valida sobre **82 ediciones de torneos mayores (2.677 partidos, "
        "1990-2026)**, no sobre un solo torneo.")
    if not mercado.empty:
        st.subheader("El benchmark más duro: contra el mercado de apuestas")
        mm = mercado.copy()
        mm["Mundial"] = mm["anio"].astype(str)
        st.dataframe(
            mm[["Mundial", "n", "logloss_modelo", "logloss_mercado"]]
            .rename(columns={"n": "Partidos", "logloss_modelo": "LogLoss modelo",
                             "logloss_mercado": "LogLoss mercado"}).round(3),
            use_container_width=True, hide_index=True)
        st.caption(
            "Las cuotas de cierre integran alineaciones, lesiones y el dinero de millones de "
            "apostantes — son el pronóstico más eficiente que existe, y ganan al modelo en los "
            "4 Mundiales. Se muestra sin adornos: un techo honesto vale más que un récord inflado.")

st.divider()
st.caption("Código y metodología completa: los 6 notebooks del repositorio. "
           "Datos actualizados por el pipeline (`scripts/actualizar_diario.sh`).")
