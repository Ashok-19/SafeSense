#!/usr/bin/env bash
# Start the SafeSense dashboard and open it in the browser. Everything else (camera, model, recording, tests) is
# done from the dashboard.
#   bash start.sh             start (first time: sets up .venv too)
#   bash start.sh --install   also add "SafeSense Demo" to the applications menu, so no terminal is needed
cd "$(dirname "$0")"
DIR="$(pwd)"
if [ "$1" = "--install" ]; then
  mkdir -p ~/.local/share/applications
  cat > ~/.local/share/applications/safesense-demo.desktop <<DESK
[Desktop Entry]
Type=Application
Name=SafeSense Demo
Comment=SafeSense live action recognition dashboard
Exec=bash "$DIR/start.sh"
Path=$DIR
Icon=camera-web
Terminal=true
Categories=Education;
DESK
  echo "Added 'SafeSense Demo' to the applications menu."
  shift
fi
URL="http://localhost:8000"
if curl -fs "$URL/api/ping" >/dev/null 2>&1; then   # already running: just open it
  xdg-open "$URL" >/dev/null 2>&1 || echo "open $URL"; exit 0
fi
[ -d .venv ] || bash setup.sh
source .venv/bin/activate
(for i in $(seq 1 60); do sleep 1; curl -fs "$URL/api/ping" >/dev/null 2>&1 && { xdg-open "$URL" >/dev/null 2>&1; break; }; done) &
echo "Starting the SafeSense dashboard at $URL (close this window or press Ctrl+C to stop it)"
exec python app.py "$@"
