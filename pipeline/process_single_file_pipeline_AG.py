#!/usr/bin/env python3
"""
Simplified pipeline: ASR → Diarization → Forced Alignment
"""

import os
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

from asr_whisperx_AG import transcribe_and_diarize


def main():
    parser = argparse.ArgumentParser(description="ASR → Diarization → Forced Alignment")
    parser.add_argument("--episode_path", required=True, help="Path to video/audio file")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token for diarization")
    parser.add_argument("--language", default=None, help="Language code (optional)")
    parser.add_argument("--num_speakers", type=int, default=None, help="Number of speakers")
    parser.add_argument("--out_dir", default="output_pipeline", help="Output directory")
    parser.add_argument("--asr_backend", default="transformers", choices=["transformers", "openai", "faster-whisper"],
                        help="faster-whisper is much faster on CPU (int8)")
    parser.add_argument("--asr_model", default=None,
                        help="ASR model. transformers: HF repo (default distil-whisper/distil-large-v3.5); "
                             "openai / faster-whisper: Whisper size such as small, medium, large-v2, "
                             "large-v3-turbo (default: by free GPU memory, or large-v3-turbo on CPU)")
    parser.add_argument("--device", default="auto", help="auto (GPU if available), cuda, cuda:N or cpu")
    parser.add_argument("--gpu_index", type=int, default=0, help="GPU index (ignored on CPU)")
    parser.add_argument("--skip_alignment", action="store_true", help="Skip word-level alignment")
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    transcribe_and_diarize(
        video_path=args.episode_path,
        hf_token=args.hf_token,
        output_dir=args.out_dir,
        asr_backend=args.asr_backend,
        asr_model_name=args.asr_model,
        diarization_model="pyannote/speaker-diarization-3.1",
        num_speakers=args.num_speakers,
        gpu_index=args.gpu_index,
        forced_language=args.language,
        skip_alignment=args.skip_alignment,
        device=args.device,
    )


if __name__ == "__main__":
    main()
