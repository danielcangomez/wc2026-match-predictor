#!/usr/bin/env bash
# Refresco diario: descarga resultados nuevos, reconstruye features y re-predice.
#
# Qué hace, en orden (cada paso solo avanza si el anterior fue bien):
#   1. Notebook 1 -> descarga el histórico y el calendario 2026 actualizados.
#   2. Notebook 2 -> reconstruye el tablón de features con los datos frescos.
#   3. Notebook 4 -> detecta rondas nuevas jugadas, las evalúa a ciegas y reentrena.
#   4. Notebook 5 -> regenera el cuadro/bracket.
#   6. (opcional) commit de results/ si hubo cambios -- descomenta el bloque de abajo.
#
# El dashboard (app.py) lee de results/, así que en cuanto esto termina, se actualiza solo.
#
# Programarlo una vez al día (macOS/Linux) con cron -- p.ej. cada día a las 9:00:
#   crontab -e
#   0 9 * * * cd /ruta/al/repo && bash scripts/actualizar_diario.sh >> logs/actualizar.log 2>&1
#
# En macOS, si prefieres launchd, un .plist con StartCalendarInterval hace lo mismo.

set -euo pipefail
cd "$(dirname "$0")/.."   # raíz del repo
mkdir -p logs

echo "===== Actualización $(date '+%Y-%m-%d %H:%M') ====="

ejecutar() {
  echo ">>> $1"
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 "notebooks/$1"
}

ejecutar "01_adquisicion_datos.ipynb"
ejecutar "02_limpieza_y_eda.ipynb"
ejecutar "04_prediccion.ipynb"
ejecutar "05_cuadro_final.ipynb"

echo ">>> Refresco completado. results/ actualizado."

# --- Commit automático (opcional): descomenta si quieres versionar cada refresco ---
# if [[ -n "$(git status --porcelain results/)" ]]; then
#   git add results/
#   git commit -m "Daily results refresh $(date '+%Y-%m-%d')"
#   git push
#   echo ">>> Cambios en results/ commiteados y subidos."
# else
#   echo ">>> Sin resultados nuevos hoy."
# fi
