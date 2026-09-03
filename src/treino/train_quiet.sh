#!/bin/bash
# Roda um treino com as telas e os LEDs apagados, sem deixar o PC suspender.
#
# Uso:
#   src/treino/train_quiet.sh .venv/bin/python src/treino/run_campaign.py
#   src/treino/train_quiet.sh --testar        # apaga por 5 s e devolve
#
# O que ele NAO faz, de proposito: nao suspende, nao hiberna, nao desliga a
# placa de video nem a rede. So apaga a saida de video e a luz dos LEDs. O
# treino continua rodando com a tela preta.

set -uo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEDS_ESTADO="$(mktemp)"
IDLE_ORIGINAL=""

apagar_tela() {
  # No GNOME sob Wayland o xset nao alcanca a saida real. O caminho e o
  # protetor de tela pelo D-Bus, e o idle-delay curto para o mutter cortar a
  # energia do monitor logo em seguida em vez de so mostrar preto.
  IDLE_ORIGINAL="$(gsettings get org.gnome.desktop.session idle-delay 2>/dev/null)"
  gsettings set org.gnome.desktop.session idle-delay 'uint32 5' 2>/dev/null
  gdbus call --session --dest org.gnome.ScreenSaver \
    --object-path /org/gnome/ScreenSaver \
    --method org.gnome.ScreenSaver.SetActive true >/dev/null 2>&1 \
    && echo "  telas: apagadas" || echo "  telas: falhou (protetor de tela nao respondeu)"
}

acender_tela() {
  gdbus call --session --dest org.gnome.ScreenSaver \
    --object-path /org/gnome/ScreenSaver \
    --method org.gnome.ScreenSaver.SetActive false >/dev/null 2>&1
  [ -n "$IDLE_ORIGINAL" ] && \
    gsettings set org.gnome.desktop.session idle-delay "$IDLE_ORIGINAL" 2>/dev/null
}

apagar_leds() {
  # Escrever 0 em brightness apaga APENAS a luz. Os LEDs de rede sao indicadores
  # da porta: apagar a luz nao mexe no enlace nem derruba a interface.
  local achou=0
  for led in /sys/class/leds/*/brightness; do
    [ -e "$led" ] || continue
    local valor
    valor="$(cat "$led" 2>/dev/null)" || continue
    echo "$led $valor" >> "$LEDS_ESTADO"
    achou=1
  done
  [ "$achou" -eq 0 ] && { echo "  LEDs: nenhum exposto pelo sistema"; return; }
  if sudo -n true 2>/dev/null; then
    while read -r led _; do echo 0 | sudo -n tee "$led" >/dev/null 2>&1; done < "$LEDS_ESTADO"
    echo "  LEDs: apagados ($(wc -l < "$LEDS_ESTADO"))"
  else
    echo "  LEDs: exigem sudo. Rode uma vez 'sudo -v' antes, ou ignore."
    : > "$LEDS_ESTADO"
  fi
}

acender_leds() {
  [ -s "$LEDS_ESTADO" ] || return
  while read -r led valor; do
    echo "$valor" | sudo -n tee "$led" >/dev/null 2>&1
  done < "$LEDS_ESTADO"
}

restaurar() {
  echo
  echo "restaurando..."
  acender_tela
  acender_leds
  rm -f "$LEDS_ESTADO"
}
trap restaurar EXIT INT TERM

if [ "${1:-}" = "--testar" ]; then
  echo "apagando por 5 s, depois devolve:"
  apagar_leds
  apagar_tela
  sleep 5
  exit 0
fi

[ $# -eq 0 ] && { echo "uso: $0 <comando de treino...>"; exit 1; }

cd "$RAIZ"
echo "silenciando a maquina:"
apagar_leds
apagar_tela
echo
echo "treino iniciado. Mexa o mouse para ver a tela; o treino nao para."
echo "acompanhe em http://127.0.0.1:8765"
echo

# systemd-inhibit impede suspensao, hibernacao, desligamento por ociosidade e
# fechamento de tampa enquanto o comando roda. Sem isso, apagar a tela poderia
# levar a maquina a dormir e matar o treino no meio.
systemd-inhibit \
  --what=sleep:idle:shutdown:handle-lid-switch \
  --who="treino YOLOv8" \
  --why="treinamento em andamento, nao interromper" \
  --mode=block \
  "$@"
