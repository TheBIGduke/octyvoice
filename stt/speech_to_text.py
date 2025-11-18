from typing import Optional
from pathlib import Path
import logging
import numpy as np
import whisper
from config.settings import SAMPLE_RATE_STT, LANGUAGE, SELF_VOCABULARY_STT


class SpeechToText:
    def __init__(self, model_path: str, model_name: str) -> None:
        self.log = logging.getLogger("SpeechToText")    
        model_path = Path(model_path)
        self.model = whisper.load_model(
            model_name, 
            download_root=str(model_path.parent),
            device='cpu'
        )

    def stt_from_bytes(self, audio_bytes: bytes) -> Optional[str]:
        """Convert bytes (Int16) -> float32 array and run Whisper transcription."""
        if not audio_bytes: 
            return None

        # Int16 -> float32 [-1, 1]
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        if pcm.size == 0:
            return None

        audio_float = pcm.astype(np.float32) / 32768.0

        if SAMPLE_RATE_STT != 16000:
            self.log.warning(f"Whisper expects 16kHz, but settings specify {SAMPLE_RATE_STT}Hz")

        try:
            result = self.model.transcribe(
                audio_float,
                temperature=0.0, 
                fp16=False, 
                language=LANGUAGE, 
                task="transcribe",
                initial_prompt=SELF_VOCABULARY_STT,
                condition_on_previous_text=False
            )
            return result.get("text", "").strip() or None
        except Exception as e:
            self.log.error(f"Transcription error: {e}")
            return None