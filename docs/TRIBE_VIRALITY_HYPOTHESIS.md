# TRIBE-based virality predictor: hypothesis and experiment plan

## Hypothesis

For a short-form video, stronger predicted viewer-brain responses—especially
in the opening seconds—may correlate with higher watch-through, rewatches,
shares, comments, and overall reach.

The proposed signal is:

> A video is more likely to perform well when its opening creates a strong,
> sustained, and distinctive predicted neural response before the viewer
> scrolls away.

This is a hypothesis to test, not an assumption that “more brain activation”
automatically means “more viral.” High activation can also represent
confusion, annoyance, fear, or negative sentiment. The predictor must be
trained against real engagement outcomes and tested against simpler baselines.

## What TRIBE provides

TRIBE v2 is a multimodal brain-encoding model. It consumes video, audio, and
language features and predicts responses for an average subject on the
fsaverage5 cortical mesh. It is not trained to predict views, retention, or
shares directly.

## Affective-region virality proxy

The working hypothesis is that a stronger early response in affective-salience
and valuation circuitry may improve attention, replays, comments, and shares.
The local scorer implements this as an interpretable ranking proxy, not as a
clinical measurement or a calibrated probability of virality.

TRIBE's released output is a 20,484-vertex fsaverage5 cortical prediction. The
scorer uses cortical proxies for:

- salience: anterior insula and anterior/mid cingulate cortex;
- appraisal/valuation: orbitofrontal and medial frontal cortex, subcallosal
  cortex, and temporal pole.

The released output does not directly measure the amygdala, hypothalamus,
hippocampus, brainstem, cerebellum, cortisol, adrenaline, or heart rate. Those
remain part of the conceptual low-road/high-road hypothesis, not observed
variables in this predictor.

Given raw TRIBE predictions saved as an `.npz` containing `preds` and `times`,
rank a candidate batch with:

```bash
.tribe_venv/bin/python scripts/predict_virality.py \
  /tmp/v1_predictions.npz /tmp/v2_predictions.npz \
  --reference /path/to/frozen_reference_01.npz /path/to/frozen_reference_02.npz \
  --output /tmp/canopychat_virality.json
```

The default score weights the first five seconds with exponential temporal
decay, combines affective mean and peak response with salience and valuation
response, and converts the result to a percentile. With `--reference`, that
percentile is measured against a frozen reference distribution rather than the
current candidate batch. This is the required mode for comparing campaigns
over time. The reference pool should eventually contain videos with real
completion/share labels; before calibration, the result is an absolute
activation percentile, not a virality probability. Without `--reference`,
`meets_70th_percentile` only means “top 30% of this batch.” Neither
interpretation means that 70% of the brain is activated.

References:

- [TRIBE v2 model card and weights](https://huggingface.co/facebook/tribev2)
- [TRIBE v2 source repository](https://github.com/facebookresearch/tribev2)
- [TRIBE v2 paper](https://arxiv.org/abs/2605.04326)

### Local checkpoint status

The TRIBE prediction checkpoint and configuration are staged locally at:

```text
models/tribev2/best.ckpt
models/tribev2/config.yaml
```

The checkpoint is approximately 677 MiB. It has not been loaded into Python or
CUDA. TRIBE's auxiliary feature encoders are separate dependencies and have
not been downloaded yet.

## Pre-token screening gate

The predictor should run before any paid Seedance or other hosted video render.
The gate has two increasingly realistic passes:

1. **Script/audio pass.** Generate several script variants, synthesize cheap
   temporary speech, and rank the opening language, pacing, and audio response.
2. **Animatic pass.** For only the best script variants, assemble a low-cost
   vertical storyboard from a reference frame, placeholder cuts/captions, and
   the temporary speech. Run TRIBE on that animatic to include visual timing
   and audiovisual interaction.
3. **Hosted render.** Send only the top-ranked animatics to Seedance. Discard
   candidates below the configured relative score or minimum opening score.

The first pass is useful for eliminating weak copy, but it cannot judge the
final face, camera, or generated motion. The animatic is the budget-friendly
proxy for those properties. The gate should rank candidates and enforce a
relative cutoff; it cannot honestly guarantee virality until calibrated on
actual retention, completion, replay, and share outcomes.

TRIBE's public wrapper accepts one of `text_path`, `audio_path`, or `video_path`.
For repeatable local screening, prefer a locally generated WAV through
`audio_path`; the upstream `text_path` helper converts text through gTTS and
then transcribes it, which introduces a network dependency and a voice choice
that may not match the eventual render.

## Proposed scoring features

Given a candidate video with duration `T`, compute TRIBE predictions at each
time step and derive features from the first `0–3` seconds, the first `0–5`
seconds, and the full clip.

### Opening-window features

- Mean predicted response in `0–1s`, `1–2s`, and `2–3s`.
- Peak response and time-to-peak.
- Area under the response curve.
- Early slope: how quickly the response rises after frame zero.
- Persistence: how long the response remains above the clip baseline.
- Distinctiveness: distance between the opening response and the clip’s later
  response distribution.
- Cross-modal agreement between the video, audio, and language feature groups.

### Full-clip features

- Mean and peak response over the entire video.
- Response variance and number of salient peaks.
- Change points around cuts, caption reveals, product mentions, and punchlines.
- Audio/visual intensity and speech-rate features.
- Caption density, first-caption timing, and first branded-claim timing.

### Position-weighted neural score

Use a decaying temporal weight so early response matters more without making
the rest of the clip irrelevant:

```text
w(t) = exp(-lambda * t)
opening_score = sum_t w(t) * neural_response(t) / sum_t w(t)
```

`lambda` must be learned or cross-validated. It should not be chosen after
looking at test-set performance.

## Predictor architecture

Start with interpretable models before trying a neural ranking model:

1. Extract TRIBE neural features and ordinary media features.
2. Normalize features using training-set statistics only.
3. Train logistic regression or gradient-boosted trees for a binary outcome.
4. Train a pairwise ranking model for two versions of the same concept.
5. Calibrate the output so “0.7 predicted performance” has a meaningful
   interpretation on held-out campaigns.

The production output should be a report rather than a single unexplained
number:

```text
predicted_score: 0.68
opening_strength: strong
opening_peak_time: 0.84s
retention_risk: medium
likely_strength: fast premise establishment
recommended_edit: move the product reveal earlier by 0.4s
confidence: low
```

## Data required

The model needs a dataset of videos with platform-normalized outcomes. At
minimum, collect:

- impressions or reach;
- 1-second, 3-second, and average watch time;
- completion rate;
- rewatches;
- likes, comments, saves, and shares;
- posting platform, audience, account size, and publish date;
- video duration, caption text, audio identity, and campaign/creative ID.

Raw views are not a sufficient label because distribution, follower count,
timing, paid promotion, and platform recommendation systems confound them.
The first useful target is usually `completion_rate` or `share_rate`, with
reach treated as a secondary outcome.

Avoid leakage: near-duplicate edits from one campaign must remain in the same
train/validation/test split. Otherwise the model will memorize the creative
or account rather than learn a transferable signal.

## Experiment design

### Phase 0: resource-safe smoke test

- Shut down the Qwen service and confirm its GPU/unified-memory allocation has
  been released.
- Load only the TRIBE checkpoint and its required encoders.
- Use one short local video and one modality at a time where possible.
- Record peak system RAM, GPU-visible memory, runtime, and cache size.
- Do not run Wan, HunyuanVideo, ComfyUI, or another large model concurrently.

### Phase 1: retrospective benchmark

- Assemble a small labeled set of existing CanopyChat videos and controls.
- Run TRIBE feature extraction without changing the videos.
- Compare opening-only, full-clip, and non-neural baselines.
- Report Spearman correlation, ROC-AUC or PR-AUC, calibration error, and
  ranking accuracy.

### Phase 2: edit ranking

For each concept, create controlled variants that change only one factor:

- hook in the first second;
- first cut timing;
- subtitle timing;
- product reveal timing;
- opening shot and camera distance;
- music onset.

Ask the predictor to rank the variants before publishing. Compare its ranking
with actual platform outcomes or a blinded human preference test.

### Phase 3: pre-finalization gate

For every campaign:

1. Generate multiple scripts without calling a paid video API.
2. Run the script/audio pass and retain only the top candidates.
3. Build cheap animatics for those candidates and run the video pass.
4. Send only the finalists above the configured cutoff to the paid renderer.
5. Run TRIBE once more on the final render for post hoc comparison with the
   animatic prediction.

The predictor should advise the editor, not automatically determine whether a
video is publishable.

## Memory and execution notes

The TRIBE prediction checkpoint is small compared with its feature encoders. The default
configuration references Llama 3.2-3B, V-JEPA2, and Wav2Vec-BERT, and its
feature-extraction infrastructure can cache representations in memory.

On the DGX Spark, use an isolated process and start conservatively:

- stop Qwen first;
- use the staged checkpoint at `models/tribev2/best.ckpt` rather than fetching
  another copy;
- keep at least 30 GB of unified memory available for TRIBE and transient
  feature tensors;
- prefer BF16/FP16 encoders where the implementation supports it;
- process one candidate at a time;
- write extracted features to disk rather than retaining a large in-memory
  campaign cache;
- measure peak memory before increasing batch size or enabling all modalities.

Do not load TRIBE in the same process as Wan, HunyuanVideo, or a large Qwen
deployment until the isolated measurement is complete.

The first success criterion is not “the score feels right.” It is whether the
TRIBE-derived features improve held-out ranking or prediction over simple
baselines such as watch-time history, caption length, audio loudness, and
first-second visual motion.
