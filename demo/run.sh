#!/usr/bin/env bash
# Start the demo. Usage: ./run.sh http://<phone-ip>:8080   (IP Webcam address shown on the phone)
#                        ./run.sh 0                         (laptop webcam)
#                        ./run.sh backup.mp4                (recorded video)
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
SRC="${1:?give the IP Webcam address, e.g. ./run.sh http://192.168.1.23:8080}"
case "$SRC" in http*) SRC="${SRC%/}"; [[ "$SRC" == */video ]] || SRC="$SRC/video";; esac
shift
python app.py --source "$SRC" "$@"
