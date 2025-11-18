from typing import Optional
from pathlib import Path
import logging
import numpy as np
import whisper
from config.settings import SAMPLE_RATE_STT, LANGUAGE, SELF_VOCABULARY_STT

class SpeechToText:
    def __init__(self, model_path:str, model_name:str) -> None:
        self.log = logging.getLogger("Speech_To_Text")    
        model_path = Path(model_path)
        # Load Whisper model
        self.model = whisper.load_model(model_name, download_root=str(model_path.parent))

    def worker_loop(self, audio_bytes: bytes) -> Optional[str | None]:
        """Wrapper for STT that adds logging."""
        if audio_bytes is None:
            return None
        try:
            text = self.stt_from_bytes(audio_bytes)
            if text:  
                self.log.info(f"{text}")
                return text
            else:
                self.log.info(f"(empty transcription)")
                return None
        except Exception as e:
            self.log.info(f"Error in STT: {e}")
            return None

    def stt_from_bytes(self, audio_bytes: bytes) -> Optional[str]:
        """Convert bytes Int16 -> float32 tensor and run Whisper."""
        if not audio_bytes: 
            return None

        # Int16 -> float32 [-1, 1]
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        if pcm.size == 0:
            return None

        x = pcm.astype(np.float32) / 32768.0

        if SAMPLE_RATE_STT != 16000:
            self.log.warning(f"Whisper expects 16 Khz, but settings say {SAMPLE_RATE_STT}hz")

        result = self.model.transcribe(
            x,
            temperature=0.0, 
            fp16=False, 
            language=LANGUAGE, 
            task="transcribe",
            initial_prompt=SELF_VOCABULARY_STT,
            condition_on_previous_text=False
        )

        return result.get("text", "").strip() or None