#!/usr/bin/env python3
"""
DAVA Insights — a point-and-click dashboard for the audio/video analysis pipeline.

For people who don't want to touch the command line. Upload a WhisperX
transcription JSON (the output of the DAVA pipeline), press a button, and
explore: word frequencies, discovered topics, sentiment & emotion over time,
named entities, and a fully annotated transcript — then download everything,
including an ELAN ``.eaf`` file.

Run it with::

    cd V2/dashboard
    streamlit run app.py

The dashboard reuses the ``analysis`` package for all heavy lifting, so what you
see here is exactly what the batch scripts produce.
"""
from __future__ import annotations

import os
import sys
import io
import json
import time
import zipfile
import tempfile
from pathlib import Path
from collections import Counter, defaultdict

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- make the analysis package importable (dashboard lives in V2/dashboard) --- #
_V2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _V2_DIR not in sys.path:
    sys.path.insert(0, _V2_DIR)

from analysis import common as C           # noqa: E402
from analysis import run_analysis          # noqa: E402
from analysis.lexicons.emotion_lexicon import ID_EMOTION  # noqa: E402

# --------------------------------------------------------------------------- #
# Page setup & light theming
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="DAVA Insights", page_icon="🎙️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1300px;}
    h1, h2, h3 {font-family: 'Segoe UI', system-ui, sans-serif;}
    /* Theme-robust: translucent backgrounds work on both dark and light themes,
       and text inherits the active theme colour so it's always readable. */
    [data-testid="stMetric"], .stMetric {
        background: rgba(128,128,128,0.10);
        border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
        padding: 12px 16px;}
    .seg-card {border-left: 4px solid #888; background: rgba(128,128,128,0.10);
               border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;
               color: inherit;}
    .seg-card .seg-text {color: inherit; opacity: 0.95;}
    .badge {display: inline-block; padding: 1px 8px; border-radius: 10px;
            font-size: 0.72rem; margin-right: 4px; color: #fff; font-weight: 600;}
    .spk {font-weight: 700; font-size: 0.8rem; opacity: 0.75;}
    .tstamp {font-size: 0.75rem; font-family: monospace; opacity: 0.6;}
</style>
""", unsafe_allow_html=True)

SENTIMENT_COLOR = {"positive": "#55A868", "neutral": "#9aa0a6", "negative": "#C44E52"}
EMOTION_COLOR = {"happy": "#E1A429", "sad": "#4C72B0", "angry": "#C44E52",
                 "fearful": "#8172B3", "surprised": "#DD8452", "disgusted": "#55A868",
                 "neutral": "#B0B0B0", "unknown": "#D9D9D9", "other": "#CCCCCC"}
ENTITY_COLOR = {"PER": "#4C72B0", "LOC": "#55A868", "ORG": "#DD8452", "MISC": "#8172B3"}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def is_enriched(segments) -> dict:
    """Report which analysis stages are already present in the JSON."""
    return {
        "entities": any(s.get("entities") for s in segments),
        "sentiment": any(s.get("sentiment") for s in segments),
        "emotion": any(s.get("emotion") for s in segments),
        "topics": any(s.get("topic_label") for s in segments),
    }


def fmt_ts(seconds: float) -> str:
    seconds = int(round(seconds or 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@st.cache_data(show_spinner=False)
def wordcloud_png(_freqs: dict) -> bytes | None:
    """Render a word cloud to PNG bytes (cached on the frequency dict)."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None
    import matplotlib
    matplotlib.use("Agg")
    from analysis.visualize import _FONT
    wc = WordCloud(width=1100, height=520, background_color="white",
                   max_words=150, font_path=_FONT, colormap="viridis",
                   prefer_horizontal=0.92).generate_from_frequencies(_freqs)
    return wc.to_image()


def zip_dir(directory: str) -> bytes:
    """Zip all files in ``directory`` (non-recursive) into memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in Path(directory).glob("*"):
            if p.is_file():
                zf.write(p, arcname=p.name)
    buf.seek(0)
    return buf.read()


# --------------------------------------------------------------------------- #
# Sidebar — load a file and (optionally) run the analysis
# --------------------------------------------------------------------------- #

def sidebar():
    st.sidebar.title("🎙️ DAVA Insights")
    st.sidebar.caption("Explore a transcribed audio/video file — no coding needed.")

    st.sidebar.subheader("1 · Load a transcript")
    uploaded = st.sidebar.file_uploader(
        "WhisperX JSON (…_whisperx.json)", type=["json"],
        help="Upload the JSON produced by the DAVA transcription pipeline. "
             "It can be a raw transcript or one that already has analysis.")

    # Also allow picking a file that already sits on disk (handy for demos).
    default_dir = os.environ.get("DAVA_JSON_DIR", "")
    disk_choice = None
    if default_dir and os.path.isdir(default_dir):
        candidates = sorted(str(p) for p in Path(default_dir).glob("*.json"))
        if candidates:
            disk_choice = st.sidebar.selectbox(
                "…or pick one on the server", ["—"] + candidates)
            if disk_choice == "—":
                disk_choice = None

    # Resolve the chosen source into a working directory + json path.
    if uploaded is not None:
        if st.session_state.get("_src_name") != uploaded.name:
            workdir = tempfile.mkdtemp(prefix="dava_")
            jpath = os.path.join(workdir, uploaded.name)
            with open(jpath, "wb") as fh:
                fh.write(uploaded.getbuffer())
            st.session_state.update(workdir=workdir, jpath=jpath,
                                    _src_name=uploaded.name, enriched=False,
                                    summary=None)
    elif disk_choice and st.session_state.get("_src_name") != disk_choice:
        workdir = tempfile.mkdtemp(prefix="dava_")
        jpath = os.path.join(workdir, os.path.basename(disk_choice))
        import shutil
        shutil.copyfile(disk_choice, jpath)
        st.session_state.update(workdir=workdir, jpath=jpath,
                                _src_name=disk_choice, enriched=False, summary=None)

    if "jpath" not in st.session_state:
        st.sidebar.info("⬆️ Upload a file to begin.")
        return

    # Peek at the file to see what's already there.
    try:
        data = C.load_transcript(st.session_state["jpath"])
    except (ValueError, UnicodeDecodeError) as exc:  # JSONDecodeError is a ValueError
        st.sidebar.error(
            f"**{os.path.basename(st.session_state['jpath'])}** is not a WhisperX "
            f"transcript JSON, so it can't be loaded ({exc}). Upload the "
            "`…_whisperx.json` the pipeline produced, or create one in the "
            "**🎙️ Transcribe** tab.")
        # forget it, so the tabs fall back to their "load a transcript" hint
        # (and the error shows again on rerun while the file is still selected)
        for key in ("jpath", "workdir", "_src_name", "enriched", "summary"):
            st.session_state.pop(key, None)
        return
    segs = C.get_segments(data)
    status = is_enriched(segs)
    st.session_state["enriched"] = all(status.values())

    st.sidebar.subheader("2 · Analysis")
    done = [k for k, v in status.items() if v]
    todo = [k for k, v in status.items() if not v]
    if done:
        st.sidebar.success("Already present: " + ", ".join(done))
    if todo:
        st.sidebar.warning("Not yet run: " + ", ".join(todo))

    with st.sidebar.expander("Advanced settings", expanded=False):
        transformers_ok = C.has_module("transformers")
        st.caption(("🟢 `transformers` available — best-quality models."
                    if transformers_ok else
                    "🟡 `transformers` not installed — offline backends will be used."))
        ner_b = st.selectbox("NER backend", ["auto", "transformers", "spacy", "regex"], 0)
        sen_b = st.selectbox("Sentiment backend", ["auto", "transformers", "lexicon"], 0)
        emo_b = st.selectbox("Emotion backend", ["auto", "transformers", "lexicon"], 0)
        topic_m = st.selectbox("Topic method", ["auto", "lda", "nmf", "bertopic", "qwen"], 0)
        qwen_model = ""
        if topic_m == "qwen":
            qwen_model = st.text_input("Qwen model", "Qwen/Qwen3-0.6B",
                                       help="LLM used to discover topics. Downloads on first use.")
        n_topics = st.slider("Number of topics (lda/nmf/qwen)", 3, 20, 8)
        chunk = st.slider("Segments per topic-chunk", 1, 15, 5)

    col1, col2 = st.sidebar.columns(2)
    run_all = col1.button("▶️ Run analysis", use_container_width=True, type="primary")
    reload_raw = col2.button("↺ Reset", use_container_width=True)

    if reload_raw:
        for k in ("enriched", "summary"):
            st.session_state.pop(k, None)
        st.rerun()

    if run_all:
        opts = {"ner_backend": ner_b, "sentiment_backend": sen_b,
                "emotion_backend": emo_b, "topic_method": topic_m,
                "num_topics": n_topics, "chunk_size": chunk, "top_n": 25,
                "topic_qwen_model": (qwen_model or None)}
        with st.spinner("Running NER, sentiment, emotion, topics & diagrams…"):
            t0 = time.time()
            summary = run_analysis.analyze_file(
                st.session_state["jpath"], out_json=st.session_state["jpath"],
                figures_dir=os.path.join(st.session_state["workdir"], "figures"),
                opts=opts)
            st.session_state["summary"] = summary
            st.session_state["enriched"] = True
        st.sidebar.success(f"Done in {round(time.time()-t0,1)}s")
        st.rerun()


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

def tab_overview(data, segs):
    st.subheader("Overview")
    total_dur = sum(C.segment_duration(s) for s in segs)
    words = sum(len((s.get("text") or "").split()) for s in segs)
    lang = C.dominant_language(segs)
    spk = C.speakers(segs)

    c = st.columns(5)
    c[0].metric("Segments", f"{len(segs):,}")
    c[1].metric("Duration", fmt_ts(total_dur))
    c[2].metric("Words", f"{words:,}")
    c[3].metric("Speakers", len(spk))
    c[4].metric("Main language", lang.upper())

    left, right = st.columns(2)
    with left:
        st.markdown("**Language mix** (by amount of speech)")
        dist = C.language_distribution(segs)
        if dist:
            df = pd.DataFrame({"language": list(dist.keys()),
                               "characters": list(dist.values())}).head(10)
            fig = px.bar(df, x="characters", y="language", orientation="h",
                         color_discrete_sequence=["#4C72B0"])
            fig.update_layout(height=320, yaxis={"categoryorder": "total ascending"},
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Speaking time per speaker**")
        dur = defaultdict(float)
        for s in segs:
            dur[s.get("speaker") or "UNKNOWN"] += C.segment_duration(s)
        if dur:
            df = pd.DataFrame({"speaker": list(dur.keys()),
                               "minutes": [v/60 for v in dur.values()]}
                              ).sort_values("minutes", ascending=True)
            fig = px.bar(df, x="minutes", y="speaker", orientation="h",
                         color_discrete_sequence=["#55A868"])
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)


def tab_words(data, segs):
    st.subheader("Most frequent words")
    lang = C.dominant_language(segs)
    top_n = st.slider("How many words", 10, 50, 25, key="wordn")
    freqs = C.word_frequencies(segs, lang=lang)
    if not freqs:
        st.info("No words to show.")
        return
    left, right = st.columns([3, 2])
    with left:
        img = wordcloud_png(dict(freqs.most_common(150)))
        if img is not None:
            st.image(img, use_container_width=True, caption="Word cloud")
        else:
            st.info("Install `wordcloud` to see the word cloud.")
    with right:
        common = freqs.most_common(top_n)
        df = pd.DataFrame(common, columns=["word", "count"])
        fig = px.bar(df, x="count", y="word", orientation="h",
                     color_discrete_sequence=["#4C72B0"])
        fig.update_layout(height=max(320, top_n*22),
                          yaxis={"categoryorder": "total ascending"},
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


def tab_topics(data, segs):
    st.subheader("Discovered topics")
    if not any(s.get("topic_label") for s in segs):
        st.info("Topics haven't been computed yet. Press **Run analysis** in the sidebar.")
        return

    # Build a topic table from per-segment labels.
    topics_json = os.path.join(os.path.dirname(st.session_state["jpath"]),
                               f"{C.base_name_from_json(st.session_state['jpath'])}_topics.json")
    table = None
    if os.path.exists(topics_json):
        with open(topics_json, encoding="utf-8") as fh:
            table = json.load(fh).get("topics")

    counts = Counter(s.get("topic_label") for s in segs
                     if s.get("topic_label") and s.get("topic_label") != "misc")
    if not counts:
        st.info("No topics assigned.")
        return

    df = pd.DataFrame(sorted(counts.items(), key=lambda kv: kv[1], reverse=True),
                      columns=["topic", "segments"])
    fig = px.bar(df, x="segments", y="topic", orientation="h",
                 color="segments", color_continuous_scale="Viridis")
    fig.update_layout(height=max(320, len(df)*36), yaxis={"categoryorder": "total ascending"},
                      coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Explore a topic**")
    topic_pick = st.selectbox("Topic", df["topic"].tolist())
    # keywords from the table if we have it
    if table:
        row = next((t for t in table if t.get("Name") == topic_pick), None)
        if row and row.get("Top_Words"):
            st.caption("Keywords: " + row["Top_Words"])
    examples = [s for s in segs if s.get("topic_label") == topic_pick][:8]
    for s in examples:
        st.markdown(
            f"<div class='seg-card'><span class='tstamp'>{fmt_ts(s.get('start',0))}</span> "
            f"<span class='spk'>{s.get('speaker','')}</span><br>{s.get('text','')}</div>",
            unsafe_allow_html=True)


def tab_sentiment_emotion(data, segs):
    st.subheader("Sentiment & emotion")
    has_sent = any(s.get("sentiment") for s in segs)
    has_emo = any(s.get("emotion") for s in segs)
    if not (has_sent or has_emo):
        st.info("Run the analysis to see sentiment and emotion.")
        return

    if has_sent:
        st.markdown("**Sentiment over time**")
        rows = [(s.get("start", 0)/60, s.get("sentiment_score", 0),
                 s.get("sentiment", "neutral")) for s in segs if "sentiment_score" in s]
        if rows:
            df = pd.DataFrame(rows, columns=["minute", "score", "sentiment"])
            df["rolling"] = df["score"].rolling(9, center=True, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["minute"], y=df["score"], mode="markers",
                                     marker=dict(size=4, color="#c9ccd1"), name="segment"))
            fig.add_trace(go.Scatter(x=df["minute"], y=df["rolling"], mode="lines",
                                     line=dict(color="#C44E52", width=3), name="trend"))
            fig.add_hline(y=0, line_color="#333", line_width=1)
            fig.update_layout(height=340, yaxis_range=[-1.05, 1.05],
                              xaxis_title="minutes", yaxis_title="sentiment (−1…+1)",
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    if has_sent:
        with c1:
            st.markdown("**Sentiment distribution**")
            cnt = Counter(s.get("sentiment") for s in segs if s.get("sentiment"))
            order = [o for o in ["positive", "neutral", "negative"] if o in cnt]
            df = pd.DataFrame({"sentiment": order, "count": [cnt[o] for o in order]})
            fig = px.bar(df, x="sentiment", y="count", color="sentiment",
                         color_discrete_map=SENTIMENT_COLOR)
            fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
    if has_emo:
        with c2:
            st.markdown("**Emotion distribution** (excluding neutral)")
            cnt = Counter(s.get("emotion") for s in segs if s.get("emotion"))
            for k in ("neutral", "unknown", "other"):
                cnt.pop(k, None)
            if cnt:
                items = cnt.most_common()
                df = pd.DataFrame(items, columns=["emotion", "count"])
                fig = px.bar(df, x="emotion", y="count", color="emotion",
                             color_discrete_map=EMOTION_COLOR)
                fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No non-neutral emotions detected.")


def tab_entities(data, segs):
    st.subheader("Named entities")
    rows = []
    for s in segs:
        for e in s.get("entities", []) or []:
            rows.append({"entity": (e.get("text") or "").strip(),
                         "type": e.get("label", "MISC"),
                         "speaker": s.get("speaker", ""),
                         "time": fmt_ts(s.get("start", 0)),
                         "score": e.get("score")})
    if not rows:
        st.info("No entities yet. Run the analysis (NER) from the sidebar.")
        return
    df = pd.DataFrame(rows)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Most mentioned**")
        top = df["entity"].value_counts().head(15).reset_index()
        top.columns = ["entity", "mentions"]
        fig = px.bar(top, x="mentions", y="entity", orientation="h",
                     color_discrete_sequence=["#55A868"])
        fig.update_layout(height=430, yaxis={"categoryorder": "total ascending"},
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**By type**")
        tc = df["type"].value_counts().reset_index()
        tc.columns = ["type", "count"]
        fig = px.pie(tc, names="type", values="count", hole=0.45,
                     color="type", color_discrete_map=ENTITY_COLOR)
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**All entities**")
    types = sorted(df["type"].unique())
    pick = st.multiselect("Filter by type", types, default=types)
    filt = df[df["type"].isin(pick)]
    st.dataframe(filt, use_container_width=True, height=300)


def tab_transcript(data, segs):
    st.subheader("Annotated transcript")
    spk_opts = ["(all)"] + C.speakers(segs)
    c1, c2, c3 = st.columns([2, 1, 1])
    query = c1.text_input("Search text", "")
    spk_filter = c2.selectbox("Speaker", spk_opts)
    sent_filter = c3.selectbox("Sentiment", ["(all)", "positive", "neutral", "negative"])
    limit = st.slider("Max segments to show", 20, 500, 120)

    shown = 0
    for s in segs:
        text = s.get("text", "")
        if query and query.lower() not in text.lower():
            continue
        if spk_filter != "(all)" and s.get("speaker") != spk_filter:
            continue
        if sent_filter != "(all)" and s.get("sentiment") != sent_filter:
            continue
        if shown >= limit:
            break
        shown += 1

        sent = s.get("sentiment")
        border = SENTIMENT_COLOR.get(sent, "#ccc")
        badges = ""
        if sent:
            badges += (f"<span class='badge' style='background:{SENTIMENT_COLOR.get(sent,'#888')}'>"
                       f"{sent} {s.get('sentiment_score','')}</span>")
        emo = s.get("emotion")
        if emo and emo not in ("neutral", "unknown"):
            badges += (f"<span class='badge' style='background:{EMOTION_COLOR.get(emo,'#888')}'>"
                       f"{emo}</span>")
        topic = s.get("topic_label")
        if topic and topic != "misc":
            short = ", ".join(topic.split("_")[:3])
            badges += f"<span class='badge' style='background:#5b6570'>🧩 {short}</span>"
        for e in (s.get("entities") or [])[:6]:
            col = ENTITY_COLOR.get(e.get("label", "MISC"), "#777")
            badges += (f"<span class='badge' style='background:{col};opacity:.85'>"
                       f"{e.get('text','')}·{e.get('label','')}</span>")

        st.markdown(
            f"<div class='seg-card' style='border-left-color:{border}'>"
            f"<span class='tstamp'>{fmt_ts(s.get('start',0))}</span> "
            f"<span class='spk'>{s.get('speaker','')}</span><br>{text}<br>{badges}</div>",
            unsafe_allow_html=True)

    st.caption(f"Showing {shown} segment(s).")


def tab_export(data, segs):
    st.subheader("Download results")
    workdir = st.session_state["workdir"]
    base = C.base_name_from_json(st.session_state["jpath"])

    # Always offer the transcript JSON itself, wherever it physically sits
    # (uploaded, picked from disk, or produced by the Transcribe tab).
    st.markdown("**Transcript (`.json`)** — the WhisperX transcript. If you have "
                "pressed Run analysis, it also carries the sentiment, emotion, topic "
                "and entity fields.")
    with open(st.session_state["jpath"], "rb") as fh:
        st.download_button("⬇️ Download transcript JSON", fh.read(),
                           file_name=os.path.basename(st.session_state["jpath"]),
                           mime="application/json", key="dl_transcript_json",
                           type="primary")
    st.divider()

    # (Re)generate the EAF on demand from the current outputs.
    st.markdown("**ELAN annotation file (`.eaf`)** — open in ELAN for linguistic work.")
    if st.button("Build / refresh EAF"):
        try:
            import json_to_eaf  # from V2/ (on sys.path)
            eaf_path = os.path.join(workdir, f"{base}.eaf")
            tsv = os.path.join(workdir, f"{base}.entities.tsv")
            csvp = os.path.join(workdir, f"{base}_topics.csv")
            json_to_eaf.convert_json_to_eaf(
                st.session_state["jpath"], eaf_path,
                ner_path=tsv if os.path.exists(tsv) else None,
                topics_csv_path=csvp if os.path.exists(csvp) else None,
                base_name=base)
            st.success(f"EAF ready: {os.path.basename(eaf_path)}")
        except Exception as exc:
            st.error(f"EAF build failed: {exc}")

    # Offer every artefact in the working dir.
    files = sorted(p for p in Path(workdir).glob("*") if p.is_file())
    figs_dir = os.path.join(workdir, "figures")

    st.markdown("**Files**")
    for p in files:
        with open(p, "rb") as fh:
            st.download_button(f"⬇️ {p.name}", fh.read(), file_name=p.name,
                               key=f"dl_{p.name}")

    if os.path.isdir(figs_dir) and any(Path(figs_dir).glob("*.png")):
        st.markdown("**Figures**")
        st.download_button("⬇️ figures.zip", zip_dir(figs_dir),
                           file_name=f"{base}_figures.zip", key="dl_figs")
        cols = st.columns(2)
        for i, png in enumerate(sorted(Path(figs_dir).glob("*.png"))):
            cols[i % 2].image(str(png), caption=png.name, use_container_width=True)


def tab_elan(data, segs):
    """ELAN-style tier timeline of the transcript + sentiment/emotion/topic layers."""
    st.subheader("ELAN annotation timeline")
    st.caption("Tier-aligned view of the transcript with your sentiment, emotion and "
               "topic layers. Click an annotation (or the time ruler) to play that "
               "moment; the playhead follows the audio.")

    import base64
    import mimetypes
    import elan_viz
    import elan_annotations
    from streamlit.components.v1 import html as _html

    if not st.session_state.get("enriched"):
        st.info("Tip: press **▶️ Run analysis** in the sidebar first, so the "
                "sentiment / emotion / topic tiers are populated. Speaker tiers "
                "show either way.")

    up = st.file_uploader("Audio (optional, for playback)",
                          type=["wav", "mp3", "m4a", "flac", "ogg"], key="elan_audio")
    audio_src = None
    if up is not None:
        mime = mimetypes.guess_type(up.name)[0] or "audio/mpeg"
        audio_src = "data:%s;base64,%s" % (mime, base64.b64encode(up.getvalue()).decode())

    jpath = st.session_state["jpath"]
    tl = elan_annotations.build_timeline_data(
        transcript_json=jpath, annotations_json=jpath,
        show=("sentiment", "emotion", "topic"),
        keys={"sentiment": "sentiment", "emotion": "emotion", "topic": "topic_label"})

    n_ann = sum(len(t["annotations"]) for t in tl["tiers"])
    st.caption("%d tiers · %d annotations" % (len(tl["tiers"]), n_ann))
    _html(elan_viz.render_elan_html(tl, audio_src=audio_src,
                                    title="ELAN annotation timeline", height_px=600),
          height=640, scrolling=False)


def tab_ask(data, segs):
    """Semantic, cross-lingual Q&A over the transcript (extractive or local Ollama)."""
    from streamlit_rag_tab import render_rag_tab
    # The transcript already loaded in the sidebar is the default source; the tab
    # still lets the user attach audio to play the referenced moments.
    render_rag_tab(default_transcript_path=st.session_state["jpath"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    sidebar()

    st.title("DAVA Insights")
    st.caption("Transcribe · named entities · sentiment · emotion · topics · ELAN export")

    from transcribe import render_transcribe_ui   # module lives at the repo root

    have = "jpath" in st.session_state
    data = segs = None
    if have:
        data = C.load_transcript(st.session_state["jpath"])
        segs = C.get_segments(data)
        have = bool(segs)

    tabs = st.tabs(["🎙️ Transcribe", "📊 Overview", "☁️ Words", "🧩 Topics",
                    "😊 Sentiment & Emotion", "🏷️ Entities", "📝 Transcript",
                    "🗂️ ELAN", "💬 Ask", "⬇️ Export"])

    # The Transcribe tab is always available (you may not have a transcript yet).
    with tabs[0]:
        just_made = render_transcribe_ui()
    if just_made:
        st.rerun()   # reload so the analysis tabs pick up the new transcript

    if not have:
        hint = ("Load a transcript in the sidebar, or create one in the "
                "**🎙️ Transcribe** tab. Then this tab fills in.")
        for i in range(1, 10):
            with tabs[i]:
                st.info(hint)
        return

    with tabs[1]: tab_overview(data, segs)
    with tabs[2]: tab_words(data, segs)
    with tabs[3]: tab_topics(data, segs)
    with tabs[4]: tab_sentiment_emotion(data, segs)
    with tabs[5]: tab_entities(data, segs)
    with tabs[6]: tab_transcript(data, segs)
    with tabs[7]: tab_elan(data, segs)
    with tabs[8]: tab_ask(data, segs)
    with tabs[9]: tab_export(data, segs)


if __name__ == "__main__":
    main()