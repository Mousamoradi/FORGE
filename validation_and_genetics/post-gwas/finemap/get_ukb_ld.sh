#!/bin/bash
# UK Biobank LD windows (Weissbrod et al.), one .npz + .gz pair per 3 Mb window.
set -uo pipefail
BASE=https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD
mkdir -p $ALKES_LD
while read -r w; do
  for ext in gz npz; do
    f=$ALKES_LD/$w.$ext
    want=$(curl -sfIL "$BASE/$w.$ext" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
    [ "$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null)" = "$want" ] && continue
    rm -f "$f"; curl -sfL --retry 5 -o "$f" "$BASE/$w.$ext"
  done
  python -c "import zipfile; zipfile.ZipFile('$ALKES_LD/$w.npz').namelist()" || echo "CORRUPT $w"
done < $OUT/ld/alkes_order.txt
