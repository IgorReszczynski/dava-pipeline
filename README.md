# Lecture Live

A Windows desktop app that transcribes a lecture live, while you sit in it. Whisper
(large-v3-turbo) runs on the laptop's own AMD GPU through Vulkan, so nothing goes to the
internet and no NVIDIA card or CUDA is needed.

## Setup

Needs Python 3.12 (with the `py` launcher), Visual Studio 2022 Build Tools with the C++
workload, CMake, the [Vulkan SDK](https://vulkan.lunarg.com), and the AMD graphics driver.

```powershell
.\setup.ps1             # everything, including the models (874 MB Whisper + 25 MB speaker model)
.\setup.ps1 -SkipModel  # everything except the models, e.g. on slow Wi-Fi
```

The script creates the venv `~\.venvs\lecture-live`, compiles whisper.cpp with Vulkan, downloads
the models to `models\` (resuming if interrupted), and adds a **Lecture Live** shortcut to the
desktop and Start menu. Run it again at any time; it skips what is already done.

## Using it

The app opens as a panel on the right edge of the screen, from the top down to the taskbar.
It reserves that strip the way the taskbar does, so maximized windows fit beside it instead
of covering it. Minimizing gives the space back. The panel background is slightly
see-through; the text stays solid.

1. Pick the module (course) the lecture belongs to. **Add module** takes a name and, if you
   want, any folder on your disk (**Choose folder…**); otherwise the module gets a folder under
   `transcripts\`. The bin icon only removes a module from the list; its folder stays.
2. Check the level bar. The dot lights up when the voice detector hears speech. The gear
   opens the settings: microphone, language, silence mode, gain, switches, vocabulary.
3. Press the round record button (or Ctrl+R). It turns into a rounded square while recording.
   Finished text appears with timestamps and speaker labels. The grey italic line is a live
   preview of the sentence still being spoken; it is replaced when that part is final.
4. Press it again to stop. The app finishes the audio it already has, then saves the session.
5. **Export** (the arrow icon) saves the lecture as text, subtitles, the audio recording, or the
   transcript read aloud.

The module name at the top switches module (when not recording).

### Speakers

With **Tell speakers apart** on, every Whisper segment gets a voice embedding (WeSpeaker
ResNet34, on the CPU) and joins the closest voice heard so far, or becomes a new one:
Speaker 1, Speaker 2, and so on. It only labels the text, never changes it. From one microphone
at the back this is a best effort: the lecturer is usually tracked well, short questions from
the room less reliably. `speaker_threshold` in `config.yaml` sets how similar two voices must
be to count as one (lower merges more, higher splits more).

### Export

| Choice | File |
|---|---|
| Transcript (text) | `.txt`, exactly as saved: `[hh:mm:ss] Speaker N: text` |
| Subtitles with timestamps | `.srt`, one cue per Whisper segment, to play alongside the recording |
| Lecture recording (audio) | `.mp3` or `.wav` of what the app recorded |
| Transcript read aloud (audio) | `.wav` or `.mp3` spoken by the offline Windows voice |

Export uses the lecture shown in the window; before recording anything it asks which
lecture folder to export.

The look follows Material 3 Expressive: the M3 shape scale with the Expressive corner radii
(20 for cards, 32 for the panel, full for buttons), M3 tonal colors that follow the Windows
light or dark mode, a record button that morphs shape, M3 switches and a connected button group.

### From the back of the room

- Leave **Raw microphone** on. Laptop mic arrays beamform toward the person at the keyboard and run
  noise suppression tuned for close speech. Both can treat a lecturer at the front as noise.
  Raw mode bypasses them.
- Raise **Gain** until the level bar fills past the middle while the lecturer talks. It turns
  red when samples clip; lower the gain if it does.
- Leave **Auto level** on. Each chunk's speech is brought to −20 dBFS before Whisper.
- Use **Hear last 5 s** with headphones to check what the app actually hears.
- Type course names and terms into **Vocabulary** so Whisper spells them right.
- Point the laptop's front edge at the lecturer and keep it off a desk that people tap or
  bump. A USB or clip-on mic aimed at the front helps more than any setting.

### Silence

| Mode | What reaches Whisper |
|---|---|
| **Skip it (VAD)** | Silero VAD finds speech; a 0.6 s pause ends a chunk. Silence between chunks is not sent. |
| **Send all audio** | Back-to-back 15 s chunks; every second of audio is sent, including silence. |

Long chunks are cut at their quietest moment, and every sample is sent exactly once.

## What it saves

Each lecture goes in `<module folder>\<date_time>\`:

| File | Contents |
|---|---|
| `transcript.txt` | `[hh:mm:ss] Speaker N: text`, a new line per chunk and per change of speaker |
| `segments.jsonl` | per chunk: start, end, language, gain, inference time, Whisper's segments with timestamps and speaker; also audio-overflow events |
| `audio.flac` | the 16 kHz mono recording, as the app heard it (turn off with `save_audio`) |
| `session.json` | module, device, GPU, settings, start and end time, error if any |

The module list itself is kept in `modules.json` next to `config.yaml`.
| `log.txt` | everything logged during the session, including whisper.cpp |

## Output is not altered, errors are not hidden

- The text is exactly what Whisper returned: no trimming, filtering, deduplication or
  hallucination removal. whisper.cpp's own no-speech drop is off (`no_speech_thold: 1.0`).
- Nothing falls back silently. If whisper.cpp starts without the GPU, the app refuses to run
  instead of quietly using the CPU. A failed decode, an unknown language code, a broken model
  file or a microphone that will not open all stop with the full error. Audio the device
  dropped is reported in the window and logged.
- There are no limits on session length or backlog. If Whisper falls behind, the waiting audio
  is shown and processed in order.

## Settings

`config.yaml` holds every setting, with comments. The Whisper ASR block is at the top:
`asr_model`, `lang`, beam size, `gpu_device`. `speakers`, `speaker_threshold` control speaker
labels. `panel_width` (share of the screen width) and
`background_opacity` (1.0 = solid) set the panel. The app writes the choices you make in its
window back into the file.

## How it works

```
mic (WASAPI, raw) -> mono -> 16 kHz -> 80 Hz high-pass -> gain -> Silero VAD (onnxruntime, CPU)
    -> segmenter (VAD or continuous) -> Whisper large-v3-turbo (whisper.cpp + Vulkan, Radeon GPU)
    -> speaker labels (WeSpeaker, onnxruntime, CPU) -> window + files in the module's folder
```

Code is in `lecturelive/`: `audio.py` (capture and signal chain), `vad.py`, `segmenter.py`,
`engine.py` (whisper.cpp bindings), `speakers.py`, `live.py` (threads and session files),
`modules.py`, `export.py`, `ui.py` (window), `dock.py` (reserving the right edge of the screen).

## Troubleshooting

- **"whisper.cpp found no GPU"**: update the AMD driver and check `vulkaninfo --summary`.
- **"The model file is incomplete"**: run `.\setup.ps1` again to resume the download.
- **The microphone will not open with Raw microphone on**: the driver does not support raw
  capture. Switch it off, and turn off *Audio enhancements* for the mic in Windows Settings → Sound.
- Start from a terminal with `run.cmd` to see errors printed there as well.
