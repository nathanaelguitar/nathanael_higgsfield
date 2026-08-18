# CanopyChat UGC scripts: copy deck for A/B testing

Copy deck for the CanopyChat UGC campaign. Each script below is a self-contained
talking-head script sized for `run_ugc_pipeline.py --script` (TTS/voice-clone
input) and intended to render through the standard 1080x1920 mux path.

Product claims must stay in sync with canopychat.app: on-device (iPhone)
processing, privacy by default, location-aware local answers, eco-friendly
positioning (no data-center energy for chats, revenue share to permanent carbon
removal via Stripe Climate). The product is in **iPhone beta** — never imply a
general App Store release.

## Production principles (what keeps it from feeling like an ad)

- **The name-drop is the CTA.** Say "It's called CanopyChat" once, in passing.
  No "link in bio," no "try it," no "join the beta now" in the spoken copy.
- **Start mid-thought.** First frame already talking. No "hey guys" intros.
- **End on a repeatable line** ("why didn't anyone do this sooner", "my stuff
  stays my stuff"). That is the part people screenshot and quote in comments.
- **One concrete demo beats feature talk.** If the render allows B-roll or a
  screen-insert, show a real question ("find me a quiet coffee spot around
  here") with a location-aware answer.
- **Keep eco claims soft.** "Burns energy, wastes water" is fine; specific
  kWh/liter numbers are not, unless sourced.
- **Captions on the facts only.** Not a full caption track — caption the
  data-center line and the name-drop.

## Scripts

Word counts are spoken-word counts (em-dashes and fragments included as spoken).
Estimated duration assumes ~2.5–2.8 words/sec natural conversational TTS;
validate against the actual render and trim one clause if it overruns.

### V1 — "the math" guilt/realization hook (primary)

```text
Every AI chat you send — even "what time is it" — goes to a data center. Burns
energy, wastes water. I just realized I've been doing that all day. So I
switched to an AI that runs on my phone instead. No cloud, no servers. It's
called CanopyChat. Honestly? Why didn't anyone do this sooner.
```

- 54 words, ~20s.
- Angle: environmental guilt. The data-center fact is comment-bait
  ("wait, really??") — expect high comment rate, which feeds reach.
- Caption targets: "goes to a data center" and "It's called CanopyChat".
- Closer type: question (provocative).

### V2 — "best friend" privacy hook

```text
So I realized my AI probably knows more about my day than my best friend. And
all of it was sitting in some data center. Now I use one that can't leave my
phone — it's called CanopyChat. It knows where I am, gets me stuff, but none of
it goes anywhere. My stuff stays my stuff.
```

- 56 words, ~20s.
- Angle: privacy/identity. Relatable self-deprecation up front; no eco claim,
  so it works for audiences that are skeptical of green marketing.
- Caption targets: "sitting in some data center" and "My stuff stays my stuff".
- Closer type: assertion (memorable line).

### V3 — hot take / anti-marketing hook

```text
Every company is saying their AI is eco-friendly, but here's the thing — every
question you ask has to travel to a data center. I found one that just runs on
my phone. No servers, no cloud, and it doesn't even read your photos. It's
called CanopyChat. I don't know why this isn't the default.
```

- 54 words, ~20s.
- Angle: hot take. Opening against "every company" earns attention from
  people already annoyed at AI marketing; the script then positions the
  creator as one of them.
- Caption targets: "has to travel to a data center" and "why this isn't the default".
- Closer type: question (mild).

### V4 — casual confession / "found a gem" hook

```text
Okay, I did the math on my AI usage and it's wild. Every question I ask travels
to a server farm somewhere. So I found an AI that just lives on my phone
instead — it's called CanopyChat. My chats don't leave the device, and it's
lighter on the planet. Why didn't I see this sooner.
```

- 55 words, ~20s.
- Angle: casual discovery. Least confrontational of the set; best fit for a
  general lifestyle creator voice. V4 is the control for the hook-type test.
- Caption targets: "travels to a server farm" and "It's called CanopyChat".
- Closer type: question (personal).

### V5 — demo-first hook (requires screen B-roll or insert)

```text
Watch this — I'm asking my AI to find a quiet coffee spot around here, and it
just knows where I am. No cloud, no servers — it runs on my phone. It's called
CanopyChat. My location didn't go anywhere, it just helped me out.
```

- 43 words, ~16s.
- Angle: show-don't-tell. Hook is the demo, not a claim. Use only if the
  render pipeline can carry a 3–4s screen-insert after "Watch this —";
  otherwise the promise goes unpaid and retention drops.
- Caption targets: "find a quiet coffee spot" and "My location didn't go anywhere".
- Closer type: assertion (calm).

### V6 — 10-second short (smoke-test / fast-iteration copy)

```text
Every question you ask an AI goes to a data center. This one just runs on your
phone — private by default. It's called CanopyChat. Why didn't anyone do this
sooner.
```

- 30 words, ~11s.
- Angle: compressed version of V1's fact + privacy one-liner, sized for the
  10s pipeline smoke path and for rapid hook iteration. Deliberately slightly
  over 10s — if the voice-clone render overruns, cut "private by default"
  first; if it under-runs, keep the beat after the closer.
- Use V6 to validate the hook before spending a full 20s render on V1–V5.

## A/B test matrix

One factor at a time, per the Phase 2 controlled-variant design in
`TRIBE_VIRALITY_HYPOTHESIS.md`. Same portrait, same voice, same music, same
edit template; only the listed factor changes.

| Test | Variants | Factor | First read |
|------|----------|--------|------------|
| T1: hook type | V1 vs V4 | opening claim (guilt vs casual confession) | completion rate |
| T2: hook type, privacy | V2 vs V1 | privacy vs environmental angle | share rate |
| T3: length | V1 vs V6 | 20s vs ~10s | completion rate |
| T4: closer | V1 vs V2 | question vs assertion closer | comment rate |
| T5: demo | V5 vs V1 | demo-first vs claim-first | completion rate + 3s retention |

Recommended order: T1 first (hook is the highest-leverage factor and V4 is
already the natural control), then T3 (length is cheap to test via V6), then
T2. T4 and T5 only after a winner from T1 exists — cross every later test
with the winning hook.

## Measurement

- Primary: completion rate. Secondary: share rate. Raw views are a confounded
  label (see the data section of `TRIBE_VIRALITY_HYPOTHESIS.md`).
- Log per creative: variant ID (V1–V6), test ID (T1–T5), platform, account,
  publish time, duration, voice identity, music, render settings.
- Keep near-duplicate edits of one creative in the same split; never let two
  variants of one script appear as independent rows in a model fit.
- Publish cadence: space variants 24h+ apart per account so the platform's
  deduplication doesn't suppress the later variant.

## Pipeline usage

```bash
.venv/bin/python run_ugc_pipeline.py \
  --reference portrait.jpg \
  --script 'Every AI chat you send — even "what time is it" — goes to a data center. Burns energy, wastes water. I just realized I've been doing that all day. So I switched to an AI that runs on my phone instead. No cloud, no servers. It\'s called CanopyChat. Honestly? Why didn\'t anyone do this sooner.' \
  --voice-reference authorized_voice.wav \
  --voice-reference-text "Exact words spoken in authorized_voice.wav" \
  --animation-backend echomimic --enhancer codeformer \
  --output outputs/canopychat_v1.mp4
```

Quote each variant ID in the output path (`outputs/canopychat_v1_t1.mp4`) so
renders map 1:1 to this deck. Only use reference voices with the speaker's
informed consent, and label synthetic media where the distribution channel
requires it.
