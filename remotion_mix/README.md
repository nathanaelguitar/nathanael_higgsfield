# CanopyChat Remotion music mix

This composition keeps the generated dialogue/video intact and adds a local
vocal-free music bed as editorial stabs. Each spoken block ends on a frozen
frame, the first second of the instrumental plays, and the video then resumes.
It intentionally does not add captions or re-render the scene.

The source assets are intentionally local and ignored by git. Prepare them
before rendering:

```bash
./prepare_assets.sh /path/to/base-trailer.mp4 /path/to/instrumental.wav
bun install
bun run render
```

The current timing map is in `src/Root.tsx`. Change `LINE_END_FRAMES` when a
new generated trailer has different dialogue timing. The rendered output is
written to `out/canopychat_trailer_with_music.mp4`.
