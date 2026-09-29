# DAVA: Data Audio/Video Analysis

Local, end-to-end pipeline that turns a recording into a transcribed, diarized,
annotated, and searchable document, with an ELAN export for linguistic work and a
semantic, cross-lingual (DE / EN / FR) question-and-answer that cites audio
timestamps. Nothing is sent to an outside service: transcription, analysis, and
answer generation all run on your own machine.

## What it does

Six stages, from raw audio to answers:

1. Clean the audio (remove music and noise, level the loudness).
2. Transcribe, identify speakers, and align (WhisperX).
3. Analyse: named entities, sentiment, emotion, and topics.
4. Export to ELAN (a `.eaf` file for annotation tools).
5. Explore in one Streamlit dashboard.
6. Ask the document: a local retrieval-augmented question-and-answer that cites
   the exact moments in the audio.

## Two ways to use it

- **All in the dashboard.** Open the dashboard, go to the `Transcribe` tab, upload
  an audio or video file, and it runs stages 1 and 2 for you. Then press
  `Run analysis` and explore, ask, and export. No command line needed.
- **From the command line.** Run the pipeline scripts directly, which is handy for
  batches or on a headless server. See the quick start below and `ANALYSIS.md`.

Either way you get the same result: a WhisperX JSON transcript enriched with
analysis, an ELAN export, and a dashboard to explore and question it.

## Quick start

DAVA uses two conda environments so the heavy speech-recognition stack does not
clash with the analysis and dashboard stack.

```bash
# 1a. transcription environment (GPU machine)
conda create -n pipeline python=3.10 -y && conda activate pipeline
pip install -r requirements_pipeline.txt
conda install -c conda-forge ffmpeg -y

# 1b. analysis / dashboard / question-and-answer environment
conda create -n dava_env python=3.10 -y && conda activate dava_env
pip install -r requirements_dava_env.txt
python -m spacy download xx_ent_wiki_sm

# 2. (command-line route) preprocess + transcribe   (env: pipeline)
python pipeline/preprocess_audio_for_asr.py \
    --input data/x.mp3 --output data/x_16k.wav --device cuda
python pipeline/process_single_file_pipeline_AG.py \
    --episode_path data/x_16k.wav --out_dir ./output --gpu_index 0 \
    --language de --asr_backend openai --hf_token <HF_TOKEN>

# 3. open the dashboard   (env: dava_env)
streamlit run dashboard/app.py
```

No NVIDIA GPU (laptop, Apple Silicon Mac)? Everything also runs on the CPU, just
more slowly. Use `--device cpu --asr_backend faster-whisper` for transcription
(int8 on the CPU, about twice as fast as the OpenAI engine there), and skip
the Demucs cleaning step for clean recordings, since it is very slow on a CPU.
`--asr_model small` trades some accuracy for speed; the CPU default is
`large-v3-turbo`.

To try the dashboard immediately without transcribing anything, launch it and load
the bundled sample `samples/sample_clean16k_whisperx.json` in the sidebar, then
press `Run analysis`.

> **Note.** The `pipeline` environment needs `demucs` for the audio-cleaning step
> (it is listed in `requirements_pipeline.txt`). If you created the environment
> earlier, install it with `conda activate pipeline && pip install "demucs>=4.0"`,
> or untick `Clean audio first` in the Transcribe tab to skip that step.

## The dashboard

`streamlit run dashboard/app.py` opens one app with ten tabs:

| Tab | What it shows |
|---|---|
| Transcribe | Upload audio or video and run the speech-to-text pipeline in the browser. |
| Overview | Length, word count, speakers, main language, and who spoke the most. |
| Words | The most frequent words, as a cloud and a ranked chart. |
| Topics | The main themes, discovered automatically. |
| Sentiment & Emotion | The tone over time and its overall balance. |
| Entities | People, places, and organisations mentioned. |
| Transcript | The full text, by speaker, with timecodes. |
| ELAN | A playable, time-aligned tier timeline; export to ELAN. |
| Ask | Ask a question and get an answer with timestamps. |
| Export | Download the transcript JSON, the figures, and the `.eaf`. |

The transcript JSON can be downloaded from the `Export` tab at any time, and also
directly from the `Transcribe` tab as soon as a transcription finishes.

If the dashboard runs on a remote GPU server, forward the port to your laptop:

```bash
ssh -L 8501:localhost:8501 <user>@<server>
# then open http://localhost:8501
```

## Documentation

- **[ANALYSIS.md](ANALYSIS.md)** is the full end-to-end guide: environment setup,
  the Ollama install, every stage with its exact commands, the output format, and
  troubleshooting.
- **[docs/GUIDE.md](docs/GUIDE.md)** is the same guide written for a non-technical
  reader from scratch, with a glossary of every acronym and plain-language
  explanations of the models (Whisper, BERT, Qwen, and the rest).
- **[analysis/README.md](analysis/README.md)** documents the analysis package
  (named entities, sentiment, emotion, topics) and its command-line use.

Everything is open source and runs locally, in keeping with the FAIR and Open
Science principles of CLARIN-CH.
