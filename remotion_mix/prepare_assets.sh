#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'Usage: %s BASE_TRAILER.mp4 VOCAL_FREE_INSTRUMENTAL.wav\n' "$0" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_DIR="${ROOT_DIR}/public"
SOURCE_VIDEO="$1"
SOURCE_MUSIC="$2"

command -v ffmpeg >/dev/null || {
  printf '%s\n' 'ffmpeg is required.' >&2
  exit 1
}

mkdir -p "${PUBLIC_DIR}"
cp -- "${SOURCE_VIDEO}" "${PUBLIC_DIR}/base-trailer.mp4"
cp -- "${SOURCE_MUSIC}" "${PUBLIC_DIR}/music_instrumental.wav"

# These match LINE_END_FRAMES in src/Root.tsx at 24 fps.
times=(1.48 3.90 5.90 7.60 9.18 11.09 12.31 13.28 15.00)
for index in "${!times[@]}"; do
  frame_number=$((index + 1))
  ffmpeg -loglevel error -y \
    -ss "${times[$index]}" \
    -i "${PUBLIC_DIR}/base-trailer.mp4" \
    -frames:v 1 -q:v 2 \
    "${PUBLIC_DIR}/freeze-${frame_number}.png"
done

printf '%s\n' 'Remotion assets prepared in public/.'
