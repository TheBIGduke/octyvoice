import logging
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor

from utils.utils import LoadModel
from stt.audio_listener import AudioListener
from stt.speech_to_text import SpeechToText
from tts.text_to_speech import TTS
from config.settings import AUDIO_LISTENER_FRAMES_PER_BUFFER


class OctyVoiceEngine:
    def __init__(self):
        self.log = logging.getLogger("OctyVoice")
        model = LoadModel()
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Audio Listener
        try:
            self.audio_listener = AudioListener()
        except Exception as e:
            self.log.error(f"Failed to initialize audio listener: {e}")
            raise

        # Speech to Text (Whisper)
        try:
            stt_model_path = model.ensure_model("stt", "small.pt")
            self.stt = SpeechToText(str(stt_model_path), "small")
        except Exception as e:
            self.log.error(f"Failed to load STT model: {e}")
            self.audio_listener.delete()
            raise

        # Text to Speech (Piper)
        try:
            tts_model_path = model.ensure_model("tts", "es_419-Octybot-medium.onnx")
            tts_conf_path = model.ensure_model("tts", "es_419-Octybot-medium.onnx.json")
            self.tts = TTS(str(tts_model_path), str(tts_conf_path))
        except Exception as e:
            self.log.error(f"Failed to load TTS model: {e}")
            self.audio_listener.delete()
            raise

        # Recording state
        self.recording = False
        self.frames = []
        
        self.log.info("OctyVoice ready")

    async def warmup_models(self):
        """Preload models with dummy data in parallel to speed up first real use."""
        self.log.info("Warming up models...")
        
        try:
            import numpy as np
            
            # Run STT and TTS warmup in parallel
            async def warmup_stt():
                dummy_audio = np.zeros(16000, dtype=np.int16)
                await self.stt.stt_from_bytes_async(dummy_audio.tobytes())
            
            async def warmup_tts():
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self.tts.synthesize,
                    "test"
                )
            
            # Run both warmups concurrently
            await asyncio.gather(warmup_stt(), warmup_tts())
            
            self.log.info("Model warmup complete")
        except Exception as e:
            self.log.warning(f"Model warmup failed (non-critical): {e}")

    async def record_audio(self) -> bytes:
        """Record audio asynchronously until Enter is pressed."""
        self.frames = []
        recording_task = None
        
        try:
            # Start recording in background
            recording_task = asyncio.create_task(self._record_loop())
            
            # Wait a tiny bit for recording to start
            await asyncio.sleep(0.1)
            
            # Wait for Enter key (non-blocking)
            await self._wait_for_enter()
            
        except asyncio.CancelledError:
            self.log.info("Recording cancelled")
        finally:
            # Signal recording to stop
            self.recording = False
            
            # Wait for recording task to finish
            if recording_task:
                try:
                    await asyncio.wait_for(recording_task, timeout=2.0)
                except asyncio.TimeoutError:
                    self.log.warning("Recording task timeout")
                    recording_task.cancel()
                    try:
                        await recording_task
                    except asyncio.CancelledError:
                        pass
        
        return b"".join(self.frames)

    async def _record_loop(self):
        """Background task that continuously records audio."""
        self.recording = True
        
        try:
            # Start stream
            await self.audio_listener.start_stream_async()
            
            while self.recording:
                # Read frame asynchronously
                data = await self.audio_listener.read_frame_async(AUDIO_LISTENER_FRAMES_PER_BUFFER)
                
                if data:
                    self.frames.append(data)
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.001)
                
        except Exception as e:
            self.log.error(f"Recording error: {e}")
            raise
        finally:
            # Stop stream
            try:
                await self.audio_listener.stop_stream_async()
            except Exception as e:
                self.log.error(f"Failed to stop stream: {e}")

    async def _wait_for_enter(self):
        """Wait for Enter key press asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Print message without blocking
        print("Recording... Press Enter to stop.")
        
        # Run blocking input() in executor to wait for Enter
        await loop.run_in_executor(self.executor, sys.stdin.readline)

    async def synthesize_and_play_streaming(self, text: str):
        """Synthesize and play audio with streaming (play while generating)."""
        if not text:
            return
        
        loop = asyncio.get_event_loop()
        
        # Run streaming synthesis in executor
        try:
            await loop.run_in_executor(self.executor, self.tts.synthesize_streaming, text)
        except Exception as e:
            self.log.error(f"Streaming TTS error: {e}")

    async def run(self):
        """Main async event loop."""
        print("\n=== OctyVoice is running (Async Mode) ===\n")
        print("Press Ctrl+C at any time to exit\n")
        
        # Warmup models in background
        await self.warmup_models()
        
        try:
            while True:
                try:
                    # Wait for user to press Enter to start
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.executor,
                        input,
                        "Press Enter to start recording\n"
                    )

                    # Record audio
                    try:
                        audio_data = await self.record_audio()
                    except Exception as e:
                        print(f"Recording error: {e}")
                        print("Please check your microphone and try again.\n")
                        continue

                    if not audio_data:
                        print("No audio data recorded.\n")
                        continue

                    print("Processing...")

                    # Transcribe in background
                    try:
                        text = await self.stt.stt_from_bytes_async(audio_data)
                    except Exception as e:
                        self.log.error(f"Transcription error: {e}")
                        print("Transcription failed. Please try again.\n")
                        continue

                    if text:
                        print(f"Transcribed: {text}")

                        # Synthesize and play with streaming
                        try:
                            await self.synthesize_and_play_streaming(text)
                        except Exception as e:
                            self.log.error(f"TTS error: {e}")
                            print("Text-to-speech failed.\n")
                    else:
                        print("Could not understand the audio\n")

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.log.error(f"Unexpected error in main loop: {e}")
                    print(f"An error occurred: {e}\n")
                    continue

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n=== Stopping OctyVoice ===\n")
            await self.stop()

    async def stop(self):
        """Clean up resources asynchronously."""
        self.log.info("Cleaning up resources...")
        
        # Stop recording if active
        self.recording = False
        
        # Cleanup in parallel
        try:
            await asyncio.gather(
                self.audio_listener.delete_async(),
                asyncio.to_thread(self.tts.stop_tts),
                return_exceptions=True
            )
        except Exception as e:
            self.log.error(f"Error during cleanup: {e}")
        
        # Shutdown executor
        self.executor.shutdown(wait=False)
        
        self.log.info("Cleanup complete")


def main():
    """Entry point that sets up asyncio."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )
    
    try:
        engine = OctyVoiceEngine()
        
        # Run async event loop
        if sys.platform == 'win32':
            # Windows requires specific event loop policy
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        asyncio.run(engine.run())
        
    except Exception as e:
        logging.error(f"Failed to start OctyVoice: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()