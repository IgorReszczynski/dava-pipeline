# DAVA Technical User Guide

**Data Audio / Video Analysis**
*From a blank machine to a running dashboard.*

Project repository: [github.com/Mchapariniya/dava-pipeline](https://github.com/Mchapariniya/dava-pipeline) · Version 1.0

---

## Contents

1. [Introduction](#1-introduction)
2. [How the pipeline works](#2-how-the-pipeline-works)
3. [Glossary of acronyms and terms](#3-glossary-of-acronyms-and-terms)
4. [The models explained](#4-the-models-explained)
5. [Prerequisites](#5-prerequisites)
6. [Getting the code](#6-getting-the-code)
7. [Setting up the two environments](#7-setting-up-the-two-environments)
8. [Installing Ollama](#8-installing-ollama-the-local-language-model)
9. [Running the pipeline, step by step](#9-running-the-pipeline-step-by-step)
10. [Running the dashboard](#10-running-the-dashboard)
11. [Using the Ask feature](#11-using-the-ask-feature)
12. [Configuration](#12-configuration)
13. [Understanding the output](#13-understanding-the-output)
14. [Troubleshooting](#14-troubleshooting)
15. [Quick command reference](#15-quick-command-reference)
16. [Appendix: folder structure](#appendix-folder-structure)

---

## 1. Introduction

DAVA is a pipeline that turns a spoken audio or video recording into a document you can read, search by meaning, and quote by the exact second. It transcribes speech to text, works out who spoke when, adds layers of analysis (topics, sentiment, emotion, and named people, places and organisations), lays everything on a time aligned timeline, and lets you ask questions about the recording in plain language. It works across German, French and English, and every part of it runs locally on your own computer, so your recordings never leave your machine.

This guide is written for someone starting from nothing. It explains, in order, how to prepare your computer, install the software, run each step of the pipeline, and open the dashboard. You do not need a background in programming to follow it, although you will be typing commands into a terminal. Sections 3 and 4 explain every abbreviation and every model in plain language, so you can refer back whenever a term is unfamiliar.

> [!NOTE]
> **What you will build.** By the end of this guide you will have two Python environments installed, a local language model running, and a web dashboard open in your browser where you can load a recording and explore every layer of analysis, including a question and answer feature.

## 2. How the pipeline works

A recording moves through six stages. Stages 1 and 2 run on the computer that has a graphics card (see [Section 5](#5-prerequisites)) and produce a transcript file. Stages 3 to 6 run inside the dashboard.

1. **Clean the audio.** Background music and noise are removed and the loudness is levelled, so the speech is as clear as possible before it is transcribed.
2. **Transcribe, identify speakers, and align.** Speech is turned into text, each line is labelled with the speaker who said it, and every word is stamped with its exact time in the recording.
3. **Analyse.** The text is enriched with topics, sentiment, emotion, and named entities (people, places, organisations).
4. **Export to ELAN.** The transcript and its annotations are written out in the standard format used by the ELAN annotation tool, so they can be shared and reused.
5. **Explore in the dashboard.** All of the above appears in one web application, with a tab for each kind of result.
6. **Ask questions.** You type a question in plain language and receive an answer that cites the exact moments in the audio it came from.

## 3. Glossary of acronyms and terms

This project, like most technical work, uses a number of abbreviations. Here is what each one means in plain language. You do not need to memorise them; use this table as a reference.

| Term | Meaning |
|---|---|
| **API** | Application Programming Interface. A defined way for one program to talk to another. |
| **ASR** | Automatic Speech Recognition. The task of turning spoken audio into written text. |
| **CLI** | Command-Line Interface. Running programs by typing commands into a terminal window. |
| **conda** | A tool that creates isolated Python environments, so the software for one project does not clash with another. |
| **cosine similarity** | A number, usually between 0 and 1, that measures how similar two pieces of text are by meaning. Higher means more similar. |
| **CSV** | Comma-Separated Values. A simple, spreadsheet-like text file where columns are separated by commas. |
| **CUDA** | Compute Unified Device Architecture. NVIDIA's system that lets programs run heavy calculations on a graphics card. |
| **diarization** | Working out who spoke when in a recording, and labelling each segment by speaker. |
| **EAF** | ELAN Annotation Format. The file format that the ELAN tool reads and writes. |
| **ELAN** | A widely used, free tool in linguistics for annotating audio and video along a timeline. |
| **embedding** | A list of numbers that represents the meaning of a piece of text, arranged so that similar meanings sit close together. |
| **FAIR** | Findable, Accessible, Interoperable, Reusable. A set of principles for producing good, shareable research data. |
| **ffmpeg** | A free, widely used tool for reading, converting and processing audio and video files. |
| **GPU** | Graphics Processing Unit, also called a graphics card. A chip that performs many calculations at once, which makes AI models run fast. |
| **Hugging Face (HF)** | A popular platform that hosts open AI models. Some models require a free access token (a kind of password) to download. |
| **JSON** | JavaScript Object Notation. A structured, human-readable text format used to store data. The transcript is a JSON file. |
| **LLM** | Large Language Model. An AI model trained on large amounts of text to understand and generate language. |
| **LUFS** | Loudness Units relative to Full Scale. A broadcasting standard for measuring how loud audio sounds to a listener. |
| **NER** | Named Entity Recognition. Automatically finding the names of people, places and organisations in text. |
| **RAG** | Retrieval-Augmented Generation. Answering a question by first retrieving the most relevant passages, then writing an answer based only on them. |
| **TSV** | Tab-Separated Values. Like CSV, but columns are separated by tab characters. |
| **UI** | User Interface. The visible part of a program that a person interacts with. |
| **VAD** | Voice Activity Detection. Automatically deciding which parts of an audio file contain speech and which are silence or noise. |

## 4. The models explained

DAVA combines several AI models, each doing one job. All of them are open source and run on your own machine. Here is what each one is, in plain terms.

### Whisper (speech to text)
Whisper is an open speech-recognition model created by OpenAI. It listens to audio and writes down what was said, and it works in many languages. It is the component that produces the raw transcript.

### WhisperX (accurate timing and speakers)
WhisperX is an extension built on top of Whisper. It adds two things Whisper alone does not do well: precise word-level timestamps (so each word is tied to an exact moment) and integration with speaker labels. It is what the pipeline actually runs for transcription.

### pyannote (who spoke when)
pyannote is an open toolkit for speaker diarization. It analyses the voices in a recording, estimates how many different people are speaking, and marks which stretches belong to which speaker. It is why the transcript can say Speaker 1, Speaker 2, and so on. Some pyannote models require a Hugging Face access token.

### Demucs (separating voice from music)
Demucs is a model that separates a piece of audio into parts, such as vocals, drums and other instruments. DAVA uses it in the cleaning step to pull the human voice away from background music and jingles, which makes transcription much more accurate.

### BERT and transformer taggers (understanding text)
BERT stands for Bidirectional Encoder Representations from Transformers. It is a family of language-understanding models that read a sentence in both directions at once to capture its meaning in context. DAVA uses BERT-style models to recognise named entities, and to judge sentiment (positive, negative, neutral) and emotion (joy, anger, sadness, and so on). When the guide mentions the transformers backend for these tasks, it means one of these BERT-based models.

### Sentence embeddings (search by meaning)
For the search and question-answering feature, DAVA turns each passage of the transcript into an embedding, a list of numbers that captures its meaning. Passages with similar meaning end up with similar numbers, so a question can be matched to the right passage even if it uses different words, and even if it is in a different language. The default model is a multilingual version of MPNet from the sentence-transformers library. Two stronger alternatives you can switch to are called multilingual E5 and BGE-M3; they understand cross-language meaning even better, at the cost of a larger download.

### Topic models: BERTopic, LDA and NMF (finding themes)
A topic model reads all the text and groups it into recurring themes. DAVA offers several methods. LDA (Latent Dirichlet Allocation) and NMF (Non-negative Matrix Factorisation) are classic statistical methods that group words that tend to appear together. BERTopic is a newer method that groups passages by meaning using embeddings, then labels each group. There is also an option to use a Qwen language model (below) to read the transcript and name the main topics directly.

### Qwen (writing answers and naming topics)
Qwen is a family of open large language models. DAVA uses a Qwen model in two places: to write the answers in the question-and-answer feature, and, optionally, to read the transcript and produce human-readable topic labels. A larger Qwen model gives better, more fluent results; a smaller one is faster and lighter. The default answering model is Qwen 2.5 with roughly seven billion parameters, written `qwen2.5:7b`.

### Ollama (running language models locally)
Ollama is not itself a model. It is a small program that downloads and runs large language models, such as Qwen, on your own computer, so that no data is sent to any outside service. DAVA talks to Ollama to generate answers in the Ask feature.

## 5. Prerequisites

### Hardware
- **A computer with an NVIDIA graphics card (GPU) is strongly recommended.** Transcription, diarization and the language models run many times faster on a GPU. The transcription step in particular is slow without one. Without a GPU (for example on a laptop or an Apple Silicon Mac) everything still works on the processor (CPU): see the `--device cpu` option in Section 9.2.
- **Disk space.** Allow at least fifteen to twenty gigabytes for the models that get downloaded on first use.

### Operating system
The instructions below are written for Linux, which is what the project is developed and run on. The same steps work on other systems with small changes to the installation commands, but Linux is assumed throughout.

### Software you need first
- **git** (to download the code).
- **Miniconda or Anaconda** (to create the Python environments). If you do not have it, download Miniconda from its official website and install it before continuing.
- **ffmpeg** (to read and convert audio). The guide shows how to install it with conda.

### Accounts
**A Hugging Face account and access token.** The speaker-diarization models are gated, which means you must accept their terms once and use a personal token to download them. Create a free account at huggingface.co, then create a token under Settings, Access Tokens. Keep this token private and never share it or commit it to a public place. You can also export it as `HF_TOKEN` so the dashboard's Transcribe tab fills it in for you.

## 6. Getting the code

Open a terminal, move to a folder where you keep projects, and download (clone) the repository:

```bash
git clone https://github.com/Mchapariniya/dava-pipeline.git
cd dava-pipeline
```

Everything from here on assumes you are inside this `dava-pipeline` folder. The folder layout is described in the [Appendix](#appendix-folder-structure).

## 7. Setting up the two environments

DAVA uses two separate conda environments. Keeping them separate stops the heavy transcription software from clashing with the analysis and dashboard software.

| Environment | Used for | Requirements file |
|---|---|---|
| **pipeline** | Stages 1 to 2: cleaning audio and transcription | `requirements_pipeline.txt` |
| **dava_env** | Stages 3 to 6: analysis, dashboard, ELAN export, and question answering | `requirements_dava_env.txt` |

### 7.1 The transcription environment (pipeline)

```bash
conda create -n pipeline python=3.10 -y
conda activate pipeline
pip install -r requirements_pipeline.txt
conda install -c conda-forge ffmpeg -y
```

This installs everything the transcription stage needs, including `demucs` for audio cleaning. If you built this environment from an older requirements file and later see `No module named 'demucs'`, run `pip install "demucs>=4.0"` here.

> [!WARNING]
> **If you see a driver error.** If importing the deep-learning library later reports that your NVIDIA driver is too old, it means the installed version was built for a newer graphics driver than yours. Reinstall a matching version, for example for a CUDA 12.8 capable driver:
> `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`

### 7.2 The analysis and dashboard environment (dava_env)

```bash
conda create -n dava_env python=3.10 -y
conda activate dava_env
pip install -r requirements_dava_env.txt
python -m spacy download xx_ent_wiki_sm
```

The last line downloads a small multilingual model used for finding names in text. The analysis environment also installs the sentence-embedding library, which enables the higher-quality text-understanding models automatically.

## 8. Installing Ollama (the local language model)

Ollama runs the language model that writes answers in the Ask feature. The Ask feature also has an Extractive mode that needs no language model at all, so this step is optional if you only want to quote passages. To get written answers, install Ollama as follows.

**Download the self-contained release.** Note that the current file ends in `.tar.zst`, not `.tgz`:

```bash
curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst -o ollama.tar.zst
zstd --version || conda install -c conda-forge zstd -y
mkdir -p ~/.local && tar -C ~/.local -I zstd -xf ollama.tar.zst
```

**Put it on your path.** Add these two lines to your `~/.bashrc` file so every new terminal can find it, then reload:

```bash
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/ollama:$LD_LIBRARY_PATH"
source ~/.bashrc
ollama --version
```

**Start the server and download the model.** Run the server in one terminal and leave it running, then pull the model in another:

```bash
ollama serve
# in a second terminal:
ollama pull qwen2.5:7b
```

> [!NOTE]
> **If your network uses a proxy.** Ollama reads the standard `http_proxy` and `https_proxy` settings, so model downloads work through a proxy automatically. Do not install Ollama through conda-forge; that package contains only the front end and will report a missing `llama-server`. The `.tar.zst` release above is the complete version.

## 9. Running the pipeline, step by step

These commands assume you have a recording to process. A small sample transcript is included in the `samples` folder so you can also try the dashboard immediately without running these steps.

If you would rather not use the command line at all, you can run stages 1 and 2 from the dashboard instead: see [The Transcribe tab](#the-transcribe-tab-create-a-transcript-in-the-browser) in Section 10.

### 9.1 Clean the audio (environment: pipeline)

This removes music and noise and levels the loudness, producing a clean 16 kHz file ready for transcription.

```bash
conda activate pipeline
python pipeline/preprocess_audio_for_asr.py \
    --input  data/your_recording.mp3 \
    --output data/your_recording_clean16k.wav \
    --device cuda --demucs-model htdemucs
```

Extra denoising is switched off by default on purpose. After the music is removed, adding heavy denoising usually makes the transcript worse, not better.

### 9.2 Transcribe, identify speakers, and align (environment: pipeline)

```bash
python pipeline/process_single_file_pipeline_AG.py \
    --episode_path data/your_recording_clean16k.wav \
    --out_dir ./output --gpu_index 0 \
    --language de --asr_backend openai \
    --hf_token YOUR_HF_TOKEN
```

Useful options: `--language` sets the language (use `de`, `fr` or `en`, or leave it out to detect automatically); `--num_speakers` fixes the number of speakers if you know it; `--asr_backend openai` selects the OpenAI Whisper engine. The result is a JSON transcript in the output folder, ending in `_whisperx.json`.

**Without a GPU.** Add `--device cpu --asr_backend faster-whisper`. The faster-whisper engine is the one built for processors and is noticeably faster there than the other two (about twice as fast as the OpenAI engine in our test). Expect transcription to take noticeably longer than on a GPU. The default model on a processor is `large-v3-turbo`; add `--asr_model small` for a quicker, somewhat less accurate transcript. The cleaning step in Section 9.1 is very slow on a processor (use `--device cpu` there too), so skip it for recordings that are already clean.

### 9.3 Analyse the transcript (environment: dava_env)

You can run the analysis with one command, or simply press the **Run analysis** button in the dashboard ([Section 10](#10-running-the-dashboard)), which does the same thing. From the command line:

```bash
conda activate dava_env
python -m analysis.run_analysis output/your_recording_clean16k_whisperx.json
```

This adds the sentiment, emotion, topic and named-entity information to every segment of the transcript, and saves charts alongside it. With only the basic packages installed it runs fully offline; with the embedding library installed (which the `dava_env` setup does) the higher-quality models turn on automatically.

### 9.4 Export to ELAN (environment: dava_env)

This produces an `.eaf` file that opens in the ELAN annotation tool, with one track per speaker plus tracks for names, sentiment and topics. The easiest way is the **Export** tab in the dashboard, or from the command line:

```bash
python json_to_eaf.py \
    --json_file output/your_recording_clean16k_whisperx.json \
    -o output/your_recording.eaf
```

## 10. Running the dashboard

The dashboard is a web application that brings every feature together. Start it from the analysis environment:

```bash
conda activate dava_env
streamlit run dashboard/app.py
```

It opens in your web browser. If you are running DAVA on a remote server, forward the port to your own computer first, then open the address it prints (usually `http://localhost:8501`):

```bash
ssh -L 8501:localhost:8501 your_user@your_server
```

### Using the dashboard

On the left is a sidebar where you load a transcript (upload a `_whisperx.json` file, or use `samples/sample_clean16k_whisperx.json` to try it out) and press **Run analysis** if the file has not been analysed yet. Across the top is a row of tabs, one for each kind of result:

| Tab | What it shows |
|---|---|
| **Transcribe** | Upload an audio or video file and run the speech-to-text pipeline from the browser. See below. |
| **Overview** | A summary: length, number of words, number of speakers, main language, plus who spoke the most. |
| **Words** | The most frequent words as a word cloud and a ranked chart. |
| **Topics** | The main themes discovered in the recording. A longer bar means the theme covers more of the recording. |
| **Sentiment & Emotion** | The emotional tone across the recording and its overall balance. |
| **Entities** | The people, places and organisations mentioned, and the mix by type. |
| **Transcript** | The full text, line by line, labelled by speaker and time. |
| **ELAN** | The transcript and its layers on a playable timeline; export to ELAN from here. |
| **Ask** | Ask a question in plain language and get an answer with timestamps (Section 11). |
| **Export** | Download the transcript JSON, the charts, and the ELAN file. The transcript JSON download is at the top of this tab. |

### The Transcribe tab (create a transcript in the browser)

The **Transcribe** tab turns a raw audio or video file into a transcript without the command line. Upload a file, set the language, the number of speakers, and the device (leave it on `auto`, or pick `cpu` on a computer without an NVIDIA GPU, which also switches to the faster-whisper engine and turns off audio cleaning), optionally paste your Hugging Face token for speaker labels (or export `HF_TOKEN` before starting the dashboard so it is filled in for you; without a token the transcript has no speaker labels), and press **Transcribe**. It cleans the audio, transcribes, identifies speakers, and aligns the words, showing a live log. When it finishes, the transcript loads automatically, you can download it there, and pressing **Run analysis** in the sidebar adds the topics, sentiment, emotion, and names.

Behind the scenes it runs the same `pipeline` scripts as the command line, in the `pipeline` environment, so nothing about the pipeline itself changes.

> [!NOTE]
> **Two setup notes for the Transcribe tab.** First, the dashboard has to find the `pipeline` environment. When both environments live under the same Miniconda it finds it automatically, and the tab's **Engine** panel shows which interpreter it will use; if it cannot find it, set `DAVA_PIPELINE_PYTHON` to the pipeline interpreter (for example `~/miniconda3/envs/pipeline/bin/python`) before starting the dashboard. Second, the audio-cleaning step needs the `demucs` package in the `pipeline` environment; if you see `No module named 'demucs'`, run `pip install "demucs>=4.0"` in that environment, or untick **Clean audio first**.

## 11. Using the Ask feature

The **Ask** tab lets you question the recording as if you were asking a colleague. Retrieval is by meaning and works across languages, so a question in French can find a passage that was spoken in German. Every answer points back to the exact moments in the audio. The main options are:

- **Answer language.** Choose the language of the answer (Auto, German, English or French). This is independent of the language of the recording.
- **Answer mode.** Extractive returns the best matching passages word for word, with timestamps, and needs no language model. Ollama writes a short, fluent answer for you and requires the Ollama server from [Section 8](#8-installing-ollama-the-local-language-model).
- **Passages.** How many excerpts the system considers when answering. More passages give broader coverage; fewer keep the answer focused.

Under **Advanced** you can change the embedding model and how the transcript is sliced into passages, but the defaults work well and most users never need to change them.

> [!TIP]
> Use **Extractive** mode for pinpoint questions, such as where a specific point was discussed. Use **Ollama** mode for summarising or comparing, such as asking what the recording is about.

## 12. Configuration

The file `config.yaml` holds default settings for the pipeline. A copy named `config.yaml.example` is provided as a starting point. Because it can hold private values such as your Hugging Face token, keep your real `config.yaml` private and never commit it to a public repository.

## 13. Understanding the output

After analysis, each segment of the transcript JSON carries the following information: the start and end time, the spoken text, the speaker label, the detected language, the individual words with their timings, a sentiment label and score, an emotion label and score, a topic label, and a list of named entities. The dashboard and the ELAN export read these fields directly. The **Export** tab lets you download this enriched JSON, the generated charts, and the ELAN file, so you can use them in other tools.

## 14. Troubleshooting

| Problem | What to do |
|---|---|
| The Transcribe tab says `No module named 'demucs'` | Install it in the pipeline environment: `conda activate pipeline && pip install "demucs>=4.0"`, or untick **Clean audio first**. |
| The Transcribe tab cannot find the pipeline environment | Set `DAVA_PIPELINE_PYTHON` to the pipeline interpreter (for example `~/miniconda3/envs/pipeline/bin/python`) before starting the dashboard. |
| Transcription stops with `CUDA was requested but is not available` | There is no usable NVIDIA GPU. Run with `--device cpu` (or pick `cpu` as the device in the Transcribe tab). |
| The deep-learning library says the NVIDIA driver is too old | Your library was built for a newer graphics driver than you have. Reinstall a matching version, for example with the `cu128` download index for a CUDA 12.8 driver. |
| A download with curl returns only about nine bytes | That tiny file is an error page. The Ollama release file now ends in `.tar.zst`, not `.tgz`. Use the address in Section 8. |
| Ollama reports `llama-server` binary not found | You are using the conda-forge package, which is incomplete. Use the `.tar.zst` release and make sure the `ollama` command points to the version in your home folder. |
| `ollama: command not found` in a new terminal | Your home folder is not on the path. Add the two export lines from Section 8 to your `~/.bashrc` file. |
| The Ask tab says Ollama is not running or the model is missing | Start the server with `ollama serve` and download the model with `ollama pull qwen2.5:7b`. The tab reports exactly what is missing. |
| The ELAN tab shows only speaker tracks | Press **Run analysis** first. The sentiment, emotion and topic tracks come from the analysis step. |
| The transcript contains odd invented phrases or low confidence | Run the audio cleaning step first to remove music and noise before transcription. |
| The dashboard does not open from your laptop | Forward the port to your machine, as shown in Section 10, then open the printed address. |
| Named entities include ordinary words | The names model can be noisy. In the sidebar **Advanced settings**, set the NER backend to `transformers` for better quality. |

## 15. Quick command reference

The whole process in order, once everything is installed:

```bash
# 1. clean the audio (environment: pipeline)
python pipeline/preprocess_audio_for_asr.py --input data/rec.mp3 --output data/rec_16k.wav --device cuda

# 2. transcribe and diarize (environment: pipeline)
python pipeline/process_single_file_pipeline_AG.py --episode_path data/rec_16k.wav \
    --out_dir ./output --gpu_index 0 --language de --asr_backend openai --hf_token YOUR_HF_TOKEN

# 3. start the local model (leave running)
ollama serve

# 4. open the dashboard (environment: dava_env) and press Run analysis inside it
streamlit run dashboard/app.py
```

## Appendix: folder structure

```text
dava-pipeline/
  ANALYSIS.md, README.md            project documentation
  requirements_pipeline.txt         packages for the pipeline environment
  requirements_dava_env.txt         packages for the dava_env environment
  config.yaml, config.yaml.example  default settings
  pipeline/                         stages 1 to 2: cleaning and transcription
  analysis/                         stage 3: sentiment, emotion, topics, names
  json_to_eaf.py                    stage 4: ELAN export
  dava_rag.py, ollama_generator.py  the Ask feature (retrieval and generation)
  elan_viz.py, elan_annotations.py  the ELAN timeline view
  streamlit_rag_tab.py              the Ask tab
  transcribe.py                     the Transcribe tab (runs stages 1 to 2 from the UI)
  dashboard/app.py                  stage 5: the dashboard
  docs/GUIDE.md, docs/index.html    this guide and the project website
  samples/                          a small sample transcript and audio clip
```

For further detail, see the `ANALYSIS.md` file in the repository, or open an issue on the [project page](https://github.com/Mchapariniya/dava-pipeline).
