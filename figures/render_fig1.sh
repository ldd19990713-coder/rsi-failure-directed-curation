#!/usr/bin/env bash
# Render Figure 1 (method pipeline) from Mermaid source to PDF / SVG / PNG.
#
# Prerequisites:
#   npm install -g @mermaid-js/mermaid-cli
#   # If running headless on a server, also install puppeteer deps:
#   # apt-get install -y chromium  (or use the bundled one)
#
# Usage:
#   bash figures/render_fig1.sh
#
# Outputs (alongside this script):
#   figures/fig1_method_pipeline.pdf
#   figures/fig1_method_pipeline.svg
#   figures/fig1_method_pipeline.png

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${HERE}/fig1_method_pipeline.mmd"

if ! command -v mmdc >/dev/null 2>&1; then
    echo "ERROR: mmdc not found. Install with: npm install -g @mermaid-js/mermaid-cli" >&2
    exit 1
fi

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: Mermaid source not found: $SRC" >&2
    exit 1
fi

for EXT in pdf svg png; do
    OUT="${HERE}/fig1_method_pipeline.${EXT}"
    mmdc -i "$SRC" -o "$OUT" -b transparent
    echo "$(printf '%3s' "$EXT")  $OUT  ($(stat -c %s "$OUT" 2>/dev/null || stat -f %z "$OUT") bytes)"
done
