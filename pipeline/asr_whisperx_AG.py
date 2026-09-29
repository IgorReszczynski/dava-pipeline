import os
import time
import json
import gc
import warnings
#import logging

# Suppress all warnings

#warnings.filterwarnings("ignore")
#os.environ['PYTHONWARNINGS'] = 'ignore'
#os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Suppress logging from dependencies
#logging.getLogger("whisperx").setLevel(logging.ERROR)
#logging.getLogger("pyannote").setLevel(logging.ERROR)
#logging.getLogger("speechbrain").setLevel(logging.ERROR)
#logging.getLogger("transformers").setLevel(logging.ERROR)
#logging.getLogger("torch").setLevel(logging.ERROR)

import torch
import numpy as np
import whisperx
from whisperx.diarize import DiarizationPipeline
from transformers import pipeline
import whisper as openai_whisper
import soundfile as sf


def _select_best_cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    best_idx, best_free = 0, -1
    for idx in range(torch.cuda.device_count()):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(idx)
        except Exception:
            free_bytes = 0
        if free_bytes > best_free:
            best_free, best_idx = free_bytes, idx
    return best_idx


# Whisper size for the openai / faster-whisper engines on a CPU when --asr_model
# is not given (on a GPU the size is picked from free memory instead).
CPU_DEFAULT_WHISPER = "large-v3-turbo"


def resolve_device(device: str | None = "auto", gpu_index: int | None = None):
    """Turn --device / --gpu_index into (torch device string, CUDA index or None).

    device: "auto" (CUDA if available, else CPU), "cuda", "cuda:N" or "cpu".
    A CUDA index of None means we run on the CPU, where every torch.cuda.* call
    must be skipped (they raise on a machine without a GPU).
    """
    device = (device or "auto").strip().lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[info] Running on CPU (no GPU). Use --asr_backend faster-whisper for speed.", flush=True)
        return "cpu", None
    if device != "cuda" and not device.startswith("cuda:"):
        raise ValueError(f"Unsupported device {device!r}: use auto, cuda, cuda:N or cpu "
                         "(Apple MPS is not supported; use cpu).")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Run with --device cpu.")
    if device.startswith("cuda:"):
        gpu_index = int(device.split(":", 1)[1])
    elif gpu_index is None:
        gpu_index = _select_best_cuda_device()
    if gpu_index >= torch.cuda.device_count():
        raise RuntimeError(f"GPU index {gpu_index} is not available. Only {torch.cuda.device_count()} GPU(s) available.")
    torch.cuda.set_device(gpu_index)
    print(f"[info] Running on GPU cuda:{gpu_index}", flush=True)
    return f"cuda:{gpu_index}", gpu_index


def default_whisper_size(gpu_idx: int | None) -> str:
    """Whisper size to use when none is given: by free GPU memory, or CPU_DEFAULT_WHISPER."""
    if gpu_idx is None:
        return CPU_DEFAULT_WHISPER
    free_bytes, _ = torch.cuda.mem_get_info(gpu_idx)
    return "large-v2" if free_bytes > 18 * 1024**3 else ("medium" if free_bytes > 8 * 1024**3 else "small")


def load_faster_whisper(model_name: str, gpu_idx: int | None, language: str | None = None):
    """Load WhisperX's own faster-whisper (CTranslate2) engine: fp16 on GPU, int8 on CPU."""
    on_gpu = gpu_idx is not None
    return whisperx.load_model(
        model_name,
        "cuda" if on_gpu else "cpu",
        device_index=gpu_idx if on_gpu else 0,
        compute_type="float16" if on_gpu else "int8",
        language=language,
        asr_options={"beam_size": 1, "best_of": 1},  # greedy decoding, like the other engines
        threads=torch.get_num_threads(),
    )


def faster_whisper_transcribe(model, audio, gpu_idx: int | None, language: str | None = None):
    """Run a load_faster_whisper() model and return the pipeline's result dict."""
    raw = model.transcribe(audio, batch_size=16 if gpu_idx is not None else 8, language=language)
    return {
        "language": raw.get("language", language),
        "language_probability": None,
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in raw.get("segments", [])
        ],
    }


def _reset_peak_mem(gpu_idx: int | None):
    if gpu_idx is not None:
        torch.cuda.reset_peak_memory_stats(gpu_idx)


def _record_peak_mem(timings: dict, stage: str, gpu_idx: int | None):
    """Store peak GPU memory for a stage in timings (nothing to record on CPU)."""
    if gpu_idx is None:
        return
    torch.cuda.synchronize(gpu_idx)
    alloc = int(torch.cuda.max_memory_allocated(gpu_idx))
    reserved = int(torch.cuda.max_memory_reserved(gpu_idx))
    timings[f"{stage}_max_mem_alloc_bytes"] = alloc
    timings[f"{stage}_max_mem_alloc_gb"] = round(alloc / (1024**3), 3)
    timings[f"{stage}_max_mem_reserved_bytes"] = reserved
    timings[f"{stage}_max_mem_reserved_gb"] = round(reserved / (1024**3), 3)


def transcribe_and_diarize(
    video_path: str,
    hf_token: str | None,
    output_dir: str = "output_asr",
    asr_backend: str = "transformers",  # "transformers" | "openai" | "faster-whisper"
    asr_model_name: str | None = None,  # None = engine default (see below)
    diarization_model: str | None = "pyannote/speaker-diarization-3.1",
    num_speakers: int | None = None,
    gpu_index: int | None = None,
    forced_language: str | None = None,
    skip_alignment: bool = False,  # Skip word-level alignment, keep only sentence-level
    device: str | None = "auto",  # "auto" | "cuda" | "cuda:N" | "cpu"
):
    os.makedirs(output_dir, exist_ok=True)

    # Check if input is already a WAV file
    if video_path.lower().endswith('.wav'):
        wav_path = video_path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
    else:
        # Convert to WAV and save to wav/ subfolder.
        # FIX: the original hard-coded format="mp4", which breaks mp3/m4a/flac/etc.
        # Let ffmpeg sniff the real format from the file itself (works for any input).
        from pydub import AudioSegment
        _ext = os.path.splitext(video_path)[1].lower().lstrip(".")
        try:
            audio = AudioSegment.from_file(video_path)            # auto-detect (preferred)
        except Exception:
            audio = AudioSegment.from_file(video_path, format=_ext or None)  # hint by extension
        audio = audio.set_channels(1).set_frame_rate(16000)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        wav_dir = os.path.join(output_dir, "wav")
        os.makedirs(wav_dir, exist_ok=True)
        wav_path = os.path.join(wav_dir, f"{base_name}.wav")
        audio.export(wav_path, format="wav")

    device, gpu_idx = resolve_device(device, gpu_index)  # gpu_idx is None on CPU
    # fp16 only on GPU; CPU kernels need float32
    torch_dtype = torch.float16 if gpu_idx is not None else torch.float32

    timings = {}

    # Load audio data once for all operations
    audio_data = whisperx.load_audio(wav_path)

    # ASR
    t0 = time.time()
    _reset_peak_mem(gpu_idx)
    
    result = None
    selected_asr_name = None
    
    if asr_backend == "transformers":
        try:
            # Explicitly load the model on the target device first
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            model_name = asr_model_name or "distil-whisper/distil-large-v3.5"
            
            # Load processor first (needed for tokenizer)
            processor = AutoProcessor.from_pretrained(model_name)
            
            # Load model and explicitly move to the device (most reliable method)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                dtype=torch_dtype,  # Use dtype instead of torch_dtype
            )
            # Explicitly move to the device - this ensures it's on the right one
            model = model.to(device)
            # Verify it's there
            first_param = next(model.parameters())
            if first_param.device != torch.device(device):
                raise RuntimeError(f"Model failed to move to {device}. Current device: {first_param.device}")
            
            # Create pipeline with model and processor on the device
            asr = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,  # Explicitly provide tokenizer
                feature_extractor=processor.feature_extractor,  # Explicitly provide feature extractor
                device=gpu_idx if gpu_idx is not None else -1,  # integer device index, -1 = CPU
                dtype=torch_dtype,  # Use dtype instead of torch_dtype
                ignore_warning=True,  # Suppress chunk_length_s warning
            )
            # Optimize generation kwargs for maximum speed
            gen_kwargs = {
                "num_beams": 1,  # Use greedy decoding (fastest)
                "do_sample": False,  # Disable sampling
                "temperature": None,  # Disable temperature
                "return_dict_in_generate": False,  # Faster output format
            }
            if forced_language:
                gen_kwargs["language"] = forced_language
            # Use optimal chunk length for speed (30s is good balance)
            # Suppress warnings during inference
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                asr_out = asr(
                    wav_path, 
                    return_timestamps=True, 
                    generate_kwargs=gen_kwargs,
                    chunk_length_s=30,  # Process in 30s chunks
                    stride_length_s=5,  # Overlap for better accuracy
                )
            chunks = asr_out.get("chunks") or []
            
            # Try to get language from output
            lang = asr_out.get("language", None)
            lang_prob = None
            
            result = {
                "language": lang,
                "language_probability": lang_prob,
                "segments": [
                    {
                        "start": (c.get("timestamp", [None, None])[0] or 0.0),
                        "end": (c.get("timestamp", [None, None])[1] or 0.0),
                        "text": (c.get("text", "").strip()),
                    }
                    for c in chunks
                    if c.get("text")
                ],
            }
            if not result["segments"]:
                result = {
                    "language": lang,
                    "language_probability": lang_prob,
                    "segments": [
                        {"start": 0.0, "end": 0.0, "text": asr_out.get("text", "").strip()}
                    ],
                }
            selected_asr_name = asr_model_name or "distil-whisper/distil-large-v3.5"
                
        except Exception as e:
            # Say why, otherwise the real error is hidden behind whatever the fallback hits next
            print(f"[warn] transformers ASR failed ({type(e).__name__}: {e}); "
                  "falling back to the openai engine.", flush=True)
            asr_backend = "openai"
            asr_model_name = None  # a transformers repo id is not an openai-whisper size

    if asr_backend == "faster-whisper":
        fw_model_name = asr_model_name or default_whisper_size(gpu_idx)
        fw_model = load_faster_whisper(fw_model_name, gpu_idx, forced_language)
        result = faster_whisper_transcribe(fw_model, audio_data, gpu_idx, forced_language)
        selected_asr_name = f"faster-whisper/{fw_model_name}"
        del fw_model
        gc.collect()
        torch.cuda.empty_cache()

    if asr_backend == "openai" and result is None:
        whisper_model_name = asr_model_name or default_whisper_size(gpu_idx)
        wmodel = openai_whisper.load_model(whisper_model_name, device=device)
        
        # Detect language if not forced
        lang = None
        lang_prob = None
        if forced_language is None:
            try:
                import librosa
                audio_data = librosa.load(wav_path, sr=16000)[0]
                # OpenAI Whisper language detection
                # Whisper uses 30 seconds at 16kHz = 480000 samples
                n_samples_30s = 30 * 16000
                mel = openai_whisper.log_mel_spectrogram(audio_data[:n_samples_30s]).to(device)
                _, probs = openai_whisper.detect_language(wmodel, mel)
                lang = max(probs, key=probs.get)
                lang_prob = float(probs[lang])
            except Exception:
                pass
        
        # Optimize OpenAI Whisper for speed
        raw = wmodel.transcribe(
            wav_path, 
            fp16=gpu_idx is not None,  # CPU has no fp16 kernels
            language=forced_language if forced_language else lang,
            beam_size=1,  # Use greedy decoding for speed
            best_of=1,  # Disable best_of for speed
            temperature=0,  # Deterministic (faster)
        )
        detected_lang = raw.get("language", lang)
        result = {
            "language": detected_lang,
            "language_probability": lang_prob,
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in raw.get("segments", [])
            ],
        }
        selected_asr_name = whisper_model_name
    
    timings["asr_s"] = time.time() - t0
    timings["asr_model"] = selected_asr_name
    timings["device"] = device
    _record_peak_mem(timings, "asr", gpu_idx)
    


    # Alignment (forced alignment for word-level timestamps)
    if skip_alignment:
        timings["align_s"] = 0.0
        timings["align_max_mem_alloc_bytes"] = 0
        timings["align_max_mem_alloc_gb"] = 0.0
    else:
        t1 = time.time()
        _reset_peak_mem(gpu_idx)
        segments = result.get("segments", [])
        # Determine language for alignment
        lang_for_alignment = result.get("language")
        if forced_language:
            lang_for_alignment = forced_language
        
        # Default to English if no language detected and none forced
        if not lang_for_alignment:
            lang_for_alignment = "en"

        if not segments:
            aligned_result = {"segments": []}
            result = aligned_result
        elif lang_for_alignment:
            # Single-language alignment (forced alignment for word-level timestamps)
            try:
                model_a, metadata = whisperx.load_align_model(language_code=lang_for_alignment, device=device)
                aligned_result = whisperx.align(segments, model_a, metadata, audio_data, device)
                del model_a
                gc.collect()
                torch.cuda.empty_cache()
                result = aligned_result
            except Exception:
                # Continue with segment-level timestamps only
                pass
        timings["align_s"] = time.time() - t1
        _record_peak_mem(timings, "align", gpu_idx)

    # Diarization (speaker labels).
    # FIX: only run when an HF token is supplied AND you've accepted the pyannote
    # model terms. For plain transcription, leave hf_token empty and this is skipped
    # (the original code always ran diarization and crashed without a token — which
    # also meant the transcript was never saved).
    t2 = time.time()
    _reset_peak_mem(gpu_idx)
    if hf_token:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                dmodel_name = diarization_model or "pyannote/speaker-diarization-3.1"
                dmodel = DiarizationPipeline(model_name=dmodel_name, use_auth_token=hf_token, device=device)
            except Exception:
                dmodel = DiarizationPipeline(model_name="pyannote/speaker-diarization@2.1", use_auth_token=hf_token, device=device)
            if num_speakers:
                dsegs = dmodel(audio_data, min_speakers=num_speakers, max_speakers=num_speakers)
            else:
                dsegs = dmodel(audio_data)
            result = whisperx.assign_word_speakers(dsegs, result)
            del dmodel
            gc.collect()
    else:
        print("[info] No hf_token provided -> skipping speaker diarization (transcription only).")
    timings["diar_s"] = time.time() - t2
    _record_peak_mem(timings, "diar", gpu_idx)

    # Initialize detected_language fields for all segments (will be filled by language detection)
    # First, set all segments to use file-level language as fallback
    file_level_lang = result.get("language")
    for segment in result.get("segments", []):
        segment["detected_language"] = file_level_lang
        segment["detected_language_probability"] = None

    # Per-segment language detection using OpenAI Whisper base (after ASR, lighter than medium)
    if result.get("segments"):
        try:
            # Load OpenAI Whisper base model for language detection (lighter than medium)
            lang_model = openai_whisper.load_model("base", device=device)
            
            segments = result.get("segments", [])
            sample_rate = 16000
            
            # Load full audio for chunk extraction - use the original WAV file
            try:
                full_audio, sr = sf.read(wav_path)
                if sr != sample_rate:
                    import librosa
                    full_audio = librosa.resample(full_audio, orig_sr=sr, target_sr=sample_rate)
                # Normalize audio to float32 range [-1, 1]
                if full_audio.dtype != np.float32:
                    if full_audio.dtype == np.int16:
                        full_audio = full_audio.astype(np.float32) / 32768.0
                    elif full_audio.dtype == np.int32:
                        full_audio = full_audio.astype(np.float32) / 2147483648.0
                    else:
                        full_audio = full_audio.astype(np.float32)
                    # Normalize to [-1, 1] range
                    if full_audio.max() > 1.0 or full_audio.min() < -1.0:
                        full_audio = full_audio / max(abs(full_audio.max()), abs(full_audio.min()))
            except Exception as e:
                # Fallback: try to use audio_data from whisperx (it's already loaded)
                try:
                    # audio_data from whisperx is already numpy array at 16kHz
                    full_audio = np.array(audio_data, dtype=np.float32)
                    # Ensure it's normalized
                    if full_audio.max() > 1.0 or full_audio.min() < -1.0:
                        full_audio = full_audio / max(abs(full_audio.max()), abs(full_audio.min()))
                except Exception:
                    # If both fail, skip language detection
                    full_audio = None
            
            if full_audio is not None:
                # Detect language for each segment
                for segment in segments:
                    start_time = segment.get("start", 0.0)
                    end_time = segment.get("end", 0.0)
                    
                    # Extract audio chunk for this segment
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    
                    # Ensure we don't go beyond audio length
                    if end_sample > len(full_audio):
                        end_sample = len(full_audio)
                    if start_sample >= end_sample:
                        segment["detected_language"] = file_level_lang
                        segment["detected_language_probability"] = None
                        continue
                    
                    segment_audio = full_audio[start_sample:end_sample].copy()
                    
                    # Skip if segment is too short (less than 0.3 seconds)
                    if len(segment_audio) < sample_rate * 0.3:
                        segment["detected_language"] = file_level_lang
                        segment["detected_language_probability"] = None
                        continue
                    
                    try:
                        # Use OpenAI Whisper for language detection
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            
                            # Ensure audio is mono and float32
                            if len(segment_audio.shape) > 1:
                                segment_audio = segment_audio.mean(axis=1)
                            segment_audio = segment_audio.astype(np.float32)
                            
                            # Normalize to [-1, 1] if needed
                            max_val = max(abs(segment_audio.max()), abs(segment_audio.min()))
                            if max_val > 1.0:
                                segment_audio = segment_audio / max_val
                            
                            # For language detection, we need at least 30 seconds or pad/truncate
                            # Whisper uses 30 seconds at 16kHz = 480000 samples
                            n_samples = 30 * sample_rate  # 30 seconds at 16kHz = 480000
                            if len(segment_audio) < n_samples:
                                # Pad with zeros to reach 30 seconds
                                padded = np.zeros(n_samples, dtype=np.float32)
                                padded[:len(segment_audio)] = segment_audio
                                segment_audio = padded
                            elif len(segment_audio) > n_samples:
                                # Truncate to 30 seconds (take middle portion for better representation)
                                start_idx = (len(segment_audio) - n_samples) // 2
                                segment_audio = segment_audio[start_idx:start_idx + n_samples]
                            
                            # Compute mel spectrogram
                            mel = openai_whisper.log_mel_spectrogram(segment_audio).to(device)
                            
                            # Detect language
                            _, probs = openai_whisper.detect_language(lang_model, mel)
                            detected_lang = max(probs, key=probs.get)
                            detected_prob = float(probs[detected_lang])
                            
                            segment["detected_language"] = detected_lang
                            segment["detected_language_probability"] = round(detected_prob, 4)
                    except Exception as e:
                        # Fallback to file-level language
                        segment["detected_language"] = file_level_lang
                        segment["detected_language_probability"] = None
            
            # Clean up language detection model
            del lang_model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            # If per-segment detection fails, use file-level language for all segments
            # (already set as fallback above, but ensure it's set)
            for segment in result.get("segments", []):
                if segment.get("detected_language") is None:
                    segment["detected_language"] = file_level_lang
                    segment["detected_language_probability"] = None

    # Save to json/ subfolder
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    out_json = os.path.join(json_dir, f"{base_name}_whisperx.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result, timings


if __name__ == "__main__":
    VIDEO_PATH = "/home/arfarh/DAVA/Conservatives_s1.mp4"
    HF_TOKEN = None
    ASR_BACKEND = "transformers"  # "transformers" | "openai"
    ASR_MODEL = "distil-whisper/distil-large-v3.5"
    NUM_SPEAKERS = None

    transcribe_and_diarize(
        video_path=VIDEO_PATH,
        hf_token=HF_TOKEN,
        output_dir="output_asr_Distil",
        asr_backend=ASR_BACKEND,
        asr_model_name=ASR_MODEL,
        diarization_model="pyannote/speaker-diarization-3.1",
        num_speakers=NUM_SPEAKERS,
        gpu_index=3,
    )