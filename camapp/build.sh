#!/usr/bin/env bash
# Builds SafeSense Cam without Android Studio. Needs: JDK, and on Ubuntu/Debian
#   sudo apt install aapt dalvik-exchange zipalign apksigner
# plus an android.jar (API 34) at $ANDROID_JAR (default ./sdk/android.jar).
set -euo pipefail
cd "$(dirname "$0")"
JAR="${ANDROID_JAR:-sdk/android.jar}"
OUT=build; rm -rf "$OUT"; mkdir -p "$OUT/gen" "$OUT/classes"
aapt package -f -m -J "$OUT/gen" -M AndroidManifest.xml -S res -I "$JAR" -F "$OUT/app.unaligned.apk"
javac -nowarn -Xlint:-options -source 8 -target 8 -encoding UTF-8 -bootclasspath "$JAR" -d "$OUT/classes" \
      $(find src "$OUT/gen" -name '*.java')
dalvik-exchange --dex --min-sdk-version=21 --output="$OUT/classes.dex" "$OUT/classes"
(cd "$OUT" && aapt add app.unaligned.apk classes.dex >/dev/null)
zipalign -f -p 4 "$OUT/app.unaligned.apk" "$OUT/app.aligned.apk"
[ -f release.keystore ] || keytool -genkeypair -keystore release.keystore -storepass safesense -keypass safesense \
    -alias safesense -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=SafeSense Cam" >/dev/null 2>&1
apksigner sign --ks release.keystore --ks-pass pass:safesense --key-pass pass:safesense --ks-key-alias safesense \
    --out "$OUT/SafeSenseCam.apk" "$OUT/app.aligned.apk"
apksigner verify --verbose "$OUT/SafeSenseCam.apk" | head -4
ls -la "$OUT/SafeSenseCam.apk"
