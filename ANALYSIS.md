# DAVA — end-to-end guide

**DAVA** (Data Audio/Video Analysis) turns a raw recording into a searchable,
annotated, linguistically-useful artifact. It runs six stages:

```
  ┌── Stage 1 ──┐   ┌── Stage 2 ──┐   ┌──── Stage 3 ────┐   ┌── Stage 4 ─┐
  │ preprocess  │→ │ transcribe   │→ │ analyse          │→ │ ELAN export │
  │ audio       │   │ + diarize    │   │ NER / sentiment /│   │ (.eaf)      │
  │ (Demucs +   │   │ + align      │   │ emotion / topics │   │             │
  │  normalise) │   │ (WhisperX)   │   │                  │   │             │
  └─────────────┘   └──────────────┘   └──────────────────┘   └─────────────┘
        │                  │                    │                    │
     .wav (16k)       *_whisperx.json     enriched JSON            .eaf
                                                │
                              ┌── Stage 5 ──────┴───── Stage 6 ──┐
                              │ Dashboard (one Streamlit app):    │
                              │ Overview · Words · Topics ·       │
                              │ Sentiment & Emotion · Entities ·  │
                              │ Transcript · ELAN timeline ·      │
                              │ Ask (RAG, local Ollama) · Export  │
                              └───────────────────────────────────┘
```

Everything runs **locally** — no external APIs. Retrieval is **semantic and
cross-lingual** (ask in French, match a German passage) and answers link back to
the exact audio timestamps.

---

## Repository layout

```
dava-pipeline/
├── ANALYSIS.md                     ← this file
├── README.md
├── requirements_pipeline.txt       ← deps for conda env "pipeline" (Stages 1–2)
├── requirements_dava_env.txt       ← deps for conda env "dava_env" (Stages 3–6)
├── config.yaml / config.yaml.example
├── .gitignore
│
├── pipeline/                       ← Stage 1–2 (GPU ASR machine)
│   ├── preprocess_audio_for_asr.py    Demucs music removal + loudness normalise
│   ├── process_single_file_pipeline_AG.py   WhisperX transcribe/diarize/align
│   ├── process_batch_gpu.py           batch version
│   ├── qwen_topic_detector.py         optional LLM topic tagging
│   ├── json_to_txt.py                 transcript → plain text
│   └── generate_mp4_manifest.py       batch manifest helper
│
├── analysis/                       ← Stage 3 (NER/sentiment/emotion/topics/figs)
│   ├── run_analysis.py                orchestrator + CLI
│   ├── ner.py  sentiment.py  emotion.py  topic_modeling.py  visualize.py
│   ├── common.py
│   └── lexicons/                      offline sentiment/emotion word lists
│
├── json_to_eaf.py                  ← Stage 4 (ELAN .eaf exporter)
│
├── dava_rag.py                     ← Stage 6 retrieval core (embeddings + search)
├── ollama_generator.py             ← Stage 6 local generation backend (Ollama)
├── elan_viz.py                     ← ELAN-style timeline renderer (used by dashboard)
├── elan_annotations.py             ← builds sentiment/emotion/topic tiers
├── streamlit_rag_tab.py            ← the "Ask" tab UI
│
├── dashboard/
│   └── app.py                      ← Stage 5 — the ONE dashboard (all tabs)
│
└── samples/
    └── sample_transcript.json      ← a small demo transcript to try the dashboard
```

Two Python environments keep the heavy ASR stack separate from everything else:

| conda env  | used for                                   | requirements file            |
|------------|--------------------------------------------|------------------------------|
| `pipeline` | Stage 1–2: preprocessing + WhisperX ASR    | `requirements_pipeline.txt`  |
| `dava_env` | Stage 3–6: analysis, dashboard, ELAN, RAG  | `requirements_dava_env.txt`  |

---

## 0 · One-time setup

### 0.1 Create the two environments

```bash
# --- ASR machine (GPU) ---
conda create -n pipeline python=3.10 -y
conda activate pipeline
pip install -r requirements_pipeline.txt
conda install -c conda-forge ffmpeg -y          # preprocessing needs ffmpeg on PATH

# --- analysis / dashboard / RAG ---
conda create -n dava_env python=3.10 -y
conda activate dava_env
pip install -r requirements_dava_env.txt
python -m spacy download xx_ent_wiki_sm          # multilingual NER model
```

> **torch vs. driver:** if `import torch` reports the NVIDIA driver is "too old",
> your torch was built for a newer CUDA than the driver supports. Reinstall a
> matching build, e.g. for a CUDA-12.8 driver:
> `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`

### 0.2 Hugging Face token (for diarization)

pyannote's diarization models are gated. Create a token at
<https://huggingface.co/settings/tokens>, accept the model terms, and pass it to
the pipeline via `--hf_token` (or export `HF_TOKEN`). **Never commit the token.**

### 0.3 Ollama (only for the "Ask" tab's generated answers)

The dashboard's **Ask** tab works with no LLM at all (extractive mode). To get
*synthesised* answers, install a local Ollama server.

```bash
# The self-contained release archive (includes llama-server + CUDA runners).
# NOTE: the current asset is .tar.zst, not .tgz.
curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst \
     -o ollama.tar.zst
zstd --version || conda install -c conda-forge zstd -y
mkdir -p ~/.local && tar -C ~/.local -I zstd -xf ollama.tar.zst

# put it on PATH (add these two lines to ~/.bashrc so every shell sees them)
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/ollama:$LD_LIBRARY_PATH"
ollama --version                                 # expect 0.32.x, GPU runners included

# start the server (tmux keeps it alive after logout), pinned to a GPU:
tmux new -s ollama
CUDA_VISIBLE_DEVICES=3 ollama serve              # detach: Ctrl-b then d
# in another shell — pull the multilingual model:
ollama pull qwen2.5:7b
```

> **Behind a proxy** (e.g. `habra`): Ollama reads `http_proxy`/`https_proxy`, so
> model pulls go through it automatically. Do **not** use the conda-forge `ollama`
> package — it ships only the frontend, so you'd get "llama-server binary not
> found". The `.tar.zst` release above is the complete, GPU-capable build.

---

## 1 · Preprocess the audio  *(env: `pipeline`)*

Removes music/jingles and normalises loudness, which kills Whisper's non-speech
hallucinations and raises word confidence.

```bash
conda activate pipeline
python pipeline/preprocess_audio_for_asr.py \
    --input  data/sample.mp3 \
    --output data/sample_clean16k.wav \
    --device cuda --demucs-model htdemucs --save-listenable
```

Output: a 16 kHz mono `sample_clean16k.wav` ready for ASR (plus a full-rate
`_enhanced.wav` to check by ear). Denoise stays **off** by default — after music
removal, extra denoising usually *raises* word error rate. Add
`--denoise deepfilternet` only if the vocal stem still sounds noisy.

## 2 · Transcribe, diarize, align  *(env: `pipeline`)*

```bash
CUDA_VISIBLE_DEVICES=3 python pipeline/process_single_file_pipeline_AG.py \
    --episode_path data/sample_clean16k.wav \
    --out_dir ./output --gpu_index 0 \
    --language de --asr_backend openai \
    --hf_token <YOUR_HF_TOKEN>
```

Key flags: `--asr_backend openai` uses OpenAI-Whisper; `--language de` forces
German (omit to auto-detect); `--num_speakers N` fixes the speaker count if you
know it; `--skip_alignment` drops word-level timestamps if you don't need them;
`--device cpu --asr_backend faster-whisper` runs without a GPU (int8 CTranslate2,
default model `large-v3-turbo`, override with `--asr_model`).
Output: `output/.../<name>_whisperx.json` — segments with `start`, `end`,
`text`, `speaker`, `words`, and per-segment `detected_language`.

## 3 · Analyse: NER, sentiment, emotion, topics  *(env: `dava_env`)*

You can do this **in the dashboard** (Stage 5, the **Run analysis** button) — the
easiest path — or from the command line:

```bash
conda activate dava_env
python -m analysis.run_analysis output/.../sample_clean16k_whisperx.json
#   writes the enriched JSON in place, and figures to <json_dir>/figures/
```

Useful options: `--topic-method {auto,lda,nmf,bertopic,qwen}`, `--num-topics N`,
`--ner-backend {auto,transformers,spacy,regex}`, `--skip-emotion`, etc. With only
the offline core installed it runs fully offline (spaCy NER + lexicon sentiment/
emotion + scikit-learn topics); with `sentence-transformers` present (it is, in
`dava_env`) the transformer backends switch on for higher quality.

This **enriches every segment** with: `sentiment`, `emotion`, `topic_label`,
`entities`, and their scores (see *Output format* below).

## 4 · ELAN export  *(env: `dava_env`)*

Produces a `.eaf` you can open in ELAN, with one tier per speaker plus tiers for
NER, sentiment and topics. Easiest from the dashboard's **Export** tab
(*Build / refresh EAF*), or via CLI:

```bash
python json_to_eaf.py \
    --json_file output/.../sample_clean16k_whisperx.json \
    -o output/.../sample.eaf \
    --ner_file   output/.../sample.entities.tsv \
    --topics_file output/.../sample_topics.csv
```

## 5 · The dashboard  *(env: `dava_env`)*

One Streamlit app with every feature:

```bash
conda activate dava_env
streamlit run dashboard/app.py
```

If the server is remote (e.g. `habra`), forward the port to your laptop:

```bash
ssh -L 8501:localhost:8501 mchapa@habra.d.uzh.ch
# then open http://localhost:8501
```

**Workflow inside the app:** load a transcript JSON in the left sidebar (upload,
or use `samples/sample_transcript.json`). If it isn't enriched yet, press
**▶️ Run analysis**. Then explore the tabs:

- **Overview / Words / Topics / Sentiment & Emotion / Entities / Transcript** —
  the analysis results as charts, tables and word clouds.
- **🗂️ ELAN** — the transcript as an ELAN-style tier timeline. After analysis,
  it shows your **real** sentiment / emotion / topic layers (colour-coded), with
  a time ruler and a playhead. Attach the audio to click any annotation and hear
  that exact moment.
- **💬 Ask** — see Stage 6.
- **⬇️ Export** — download the enriched JSON, figures, and the `.eaf`.

## 6 · Ask the document (RAG)  *(env: `dava_env`)*

In the **Ask** tab, type a question in German, English or French and choose:

- **Answer language** — Auto (match the question) / German / English / French.
- **Answer mode**:
  - **Extractive (no LLM)** — returns the top matching passages verbatim, with
    timestamps. Always available, fully offline; best for pinpoint questions.
  - **Ollama (local)** — feeds those passages to `qwen2.5:7b` and writes a
    concise, synthesised answer *in the chosen language*, citing `[MM:SS]`. Best
    for "what is this about / summarise / compare" questions. Requires the Ollama
    server from step 0.3.

Every answer lists its **reference passages with start/end times**, shown on a
mini timeline you can play. Retrieval is semantic and cross-lingual by design.

You can also drive it from the command line:

```bash
python ollama_generator.py "Was bedeutet bedürfnisorientierte Erziehung?" \
    output/.../sample_clean16k_whisperx.json de
```

**How it fits together (architecture):** the three RAG layers are decoupled so
you can swap any one without touching the others —
`dava_rag.py` (retrieval: chunking + multilingual embeddings + vector search) →
`ollama_generator.py` (generation: local Ollama) → `streamlit_rag_tab.py` (UI).
The seam is a plain `Passage` object and a one-method `generate()` contract.

---

## Output format (enriched segment)

After Stage 3 each segment looks like:

```json
{
  "start": 71.7, "end": 73.0,
  "text": "Die bedürfnisorientierte Erziehung ...",
  "speaker": "SPEAKER_03",
  "detected_language": "de",
  "words": [ { "word": "Die", "start": 71.7, "end": 71.9, "score": 0.98 }, ... ],
  "sentiment": "neutral", "sentiment_score": 0.71,
  "emotion": "joy", "emotion_id": 1, "emotion_score": 0.64,
  "topic_id": 2, "topic_label": "2_social_media_erziehung_schwer", "topic_score": 0.55,
  "entities": [ { "text": "Bern", "label": "LOC", "start": 12, "end": 16, "score": 0.9 } ]
}
```

The dashboard's ELAN tab maps `sentiment` / `emotion` / `topic_label` to timeline
tiers; the RAG tab windows the segments (45 s / 10 s overlap) into passages that
keep these timestamps.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch` says the NVIDIA driver is "too old" | torch built for newer CUDA than the driver. Reinstall for your driver, e.g. `--index-url .../whl/cu128`. |
| `curl` download returns ~9 bytes | it's a real 404 ("Not Found"). Ollama assets are now `.tar.zst`, not `.tgz` — use the URL in §0.3. |
| Ollama: `llama-server binary not found` | you're running the conda-forge stub. Use the `.tar.zst` release; make sure `which ollama` points at `~/.local/bin/ollama` (v0.32.x). |
| `bash: ollama: command not found` in a new shell | `~/.local/bin` isn't on PATH. Add the two `export` lines from §0.3 to `~/.bashrc`. |
| Ask tab says "Ollama not running / model not pulled" | start `ollama serve` and `ollama pull qwen2.5:7b`; the tab's readiness check reports exactly what's missing. |
| ELAN tab shows only speaker tiers | run **▶️ Run analysis** first — the sentiment/emotion/topic tiers come from the enriched fields. |
| Whisper hallucinates subtitle credits / low word confidence | run Stage 1 preprocessing first to strip music/jingles. |
| Streamlit not reachable from laptop | forward the port: `ssh -L 8501:localhost:8501 <user>@<host>`. |
| Model downloads blocked | set `http_proxy`/`https_proxy` (HF, Ollama and pip all honour them). |
