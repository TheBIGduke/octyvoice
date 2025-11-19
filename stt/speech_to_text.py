from typing import Optional
from pathlib import Path
import logging
import asyncio
import numpy as np
import whisper
from concurrent.futures import ThreadPoolExecutor
from config.settings import SAMPLE_RATE_STT, LANGUAGE, SELF_VOCABULARY_STT


class SpeechToText:
    def __init__(self, model_path: str, model_name: str) -> None:
        self.log = logging.getLogger("SpeechToText")
        
        # Validate model path
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load Whisper model
        try:
            self.model = whisper.load_model(
                model_name, 
                download_root=str(model_path.parent),
                device='cpu'
            )
            self.log.info(f"Loaded Whisper model: {model_name}")
        except Exception as e:
            self.log.error(f"Failed to load Whisper model: {e}")
            raise
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Validate sample rate
        if SAMPLE_RATE_STT != 16000:
            self.log.warning(
                f"Whisper expects 16kHz audio, but SAMPLE_RATE_STT is {SAMPLE_RATE_STT}Hz. "
                f"Transcription quality may be affected."
            )

    def stt_from_bytes(self, audio_bytes: bytes, log_output: bool = True) -> Optional[str]:
        """Convert bytes (Int16) -> float32 array and run Whisper transcription."""
        if not audio_bytes: 
            if log_output:
                self.log.warning("Empty audio bytes provided")
            return None

        try:
            # Convert Int16 -> float32 [-1, 1]
            pcm = np.frombuffer(audio_bytes, dtype=np.int16)
            
            if pcm.size == 0:
                if log_output:
                    self.log.warning("Audio buffer is empty after conversion")
                return None
            
            # Check for minimum audio length (at least 0.5 seconds)
            min_samples = int(0.5 * SAMPLE_RATE_STT)
            if pcm.size < min_samples and log_output:
                self.log.warning(
                    f"Audio too short: {pcm.size} samples ({pcm.size/SAMPLE_RATE_STT:.2f}s). "
                    f"Minimum recommended: {min_samples} samples (0.5s)"
                )
            
            # Normalize to float32
            audio_float = pcm.astype(np.float32) / 32768.0
            
            # Run Whisper transcription
            result = self.model.transcribe(
                audio_float,
                temperature=0.0, 
                fp16=False, 
                language=LANGUAGE, 
                task="transcribe",
                initial_prompt=SELF_VOCABULARY_STT,
                condition_on_previous_text=False
            )
            
            # Extract and clean text
            text = result.get("text", "").strip()
            
            if not text:
                if log_output:
                    self.log.info("Transcription returned empty text")
                return None
            
            if log_output:
                self.log.info(f"Transcribed: {text[:50]}{'...' if len(text) > 50 else ''}")
            return text
            
        except Exception as e:
            if log_output:
                self.log.error(f"Transcription error: {e}")
            return None

    async def stt_from_bytes_async(self, audio_bytes: bytes, log_output: bool = True) -> Optional[str]:
        """Async version of stt_from_bytes. Runs transcription in executor."""
        loop = asyncio.get_event_loop()
        
        try:
            text = await loop.run_in_executor(
                self.executor,
                self.stt_from_bytes,
                audio_bytes,
                log_output
            )
            return text
        except Exception as e:
            if log_output:
                self.log.error(f"Async transcription error: {e}")
            return None

    def stt_from_file(self, audio_path: str) -> Optional[str]:
        """Transcribe audio from a file."""
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            self.log.error(f"Audio file not found: {audio_path}")
            return None
        
        try:
            result = self.model.transcribe(
                str(audio_path),
                temperature=0.0,
                fp16=False,
                language=LANGUAGE,
                task="transcribe",
                initial_prompt=SELF_VOCABULARY_STT,
                condition_on_previous_text=False
            )
            
            text = result.get("text", "").strip()
            
            if not text:
                self.log.info("Transcription returned empty text")
                return None
            
            return text
            
        except Exception as e:
            self.log.error(f"Error transcribing file {audio_path}: {e}")
            return None

    async def stt_from_file_async(self, audio_path: str) -> Optional[str]:
        """Async version of stt_from_file."""
        loop = asyncio.get_event_loop()
        
        try:
            text = await loop.run_in_executor(
                self.executor,
                self.stt_from_file,
                audio_path
            )
            return text
        except Exception as e:
            self.log.error(f"Async file transcription error: {e}")
            return None

    def shutdown(self):
        """Clean up executor."""
        self.executor.shutdown(wait=False)


# Example Usage
if __name__ == "__main__":
    import wave
    
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    from utils.utils import LoadModel
    
    # Load model
    model = LoadModel()
    stt_path = model.ensure_model("stt", "base.pt")
    
    print(f"Loading STT model from: {stt_path}")
    stt = SpeechToText(str(stt_path), "base")
    
    print("\nSTT ready. Testing with sample audio...")
    
    # Test with dummy audio (sync)
    import numpy as np
    dummy_audio = np.random.randint(-1000, 1000, 16000 * 3, dtype=np.int16)  # 3 seconds
    text = stt.stt_from_bytes(dummy_audio.tobytes())
    
    if text:
        print(f"Sync transcription: {text}")
    else:
        print("Sync: No transcription (expected with random noise)")
    
    # Test async version
    async def test_async():
        print("\nTesting async transcription...")
        text = await stt.stt_from_bytes_async(dummy_audio.tobytes())
        if text:
            print(f"Async transcription: {text}")
        else:
            print("Async: No transcription (expected with random noise)")
    
    asyncio.run(test_async())
    
    # Cleanup
    stt.shutdown()