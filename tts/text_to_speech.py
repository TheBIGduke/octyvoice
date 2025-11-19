import io
import wave
import numpy as np
import pyaudio
import logging
import asyncio
from pathlib import Path
from piper.voice import PiperVoice, SynthesisConfig
from typing import Optional, Iterator, Callable
from concurrent.futures import ThreadPoolExecutor

from config.settings import (
    SAMPLE_RATE_TTS,
    SAVE_WAV_TTS,
    PATH_TO_SAVE_TTS,
    NAME_OF_OUTS_TTS,
    VOLUME_TTS,
    LENGTH_SCALE_TTS
)


class TTS:
    def __init__(self, model_path: str, model_path_conf: str):
        self.log = logging.getLogger("TextToSpeech")
        
        # Validate paths
        model_path = Path(model_path)
        model_path_conf = Path(model_path_conf)
        
        if not model_path.exists():
            raise FileNotFoundError(f"TTS model not found: {model_path}")
        if not model_path_conf.exists():
            raise FileNotFoundError(f"TTS config not found: {model_path_conf}")
        
        self.log.info("Loading Piper TTS model...")
        
        try:
            self.voice = PiperVoice.load(model_path=str(model_path), config_path=str(model_path_conf))
        except Exception as e:
            self.log.error(f"Failed to load Piper voice: {e}")
            raise
        
        self.sample_rate = SAMPLE_RATE_TTS
        self.count_of_audios = 0
        self.out_path = Path(PATH_TO_SAVE_TTS) / Path(NAME_OF_OUTS_TTS) / Path(f"{NAME_OF_OUTS_TTS}_{self.count_of_audios}.wav")
        
        # Validate synthesis config
        if VOLUME_TTS <= 0:
            self.log.warning(f"Invalid VOLUME_TTS={VOLUME_TTS}, using 1.0")
            volume = 1.0
        else:
            volume = VOLUME_TTS
        
        if LENGTH_SCALE_TTS <= 0:
            self.log.warning(f"Invalid LENGTH_SCALE_TTS={LENGTH_SCALE_TTS}, using 1.0")
            length_scale = 1.0
        else:
            length_scale = LENGTH_SCALE_TTS
        
        self.syn_config = SynthesisConfig(
            volume=volume,
            length_scale=length_scale,
            noise_scale=1.0,
            noise_w_scale=1.0,
            normalize_audio=False,
        )

        # Initialize PyAudio once for performance
        self.pa = None
        self.stream = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        try:
            self.pa = pyaudio.PyAudio()
        except Exception as e:
            self.log.error(f"Failed to initialize PyAudio: {e}")
            raise
        
        self.log.info("Text-To-Speech initialized (streaming enabled)")

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Convert text to speech using Piper, return mono audio float32 [-1,1]"""
        if not text or not text.strip():
            self.log.warning("Empty text provided for synthesis")
            return None
        
        try:
            # Save to file if enabled
            if SAVE_WAV_TTS:
                self.out_path.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(self.out_path), "wb") as wav_file:
                    self.voice.synthesize_wav(text, wav_file, syn_config=self.syn_config)
                self.count_of_audios += 1
                self.out_path = Path(PATH_TO_SAVE_TTS) / Path(NAME_OF_OUTS_TTS) / Path(f"{NAME_OF_OUTS_TTS}_{self.count_of_audios}.wav")

            # Generate audio in memory
            mem = io.BytesIO()
            with wave.open(mem, "wb") as w:
                self.voice.synthesize_wav(text, w, syn_config=self.syn_config)

            # Read WAV from buffer and return as normalized float32
            mem.seek(0)
            with wave.open(mem, "rb") as r:
                frames = r.readframes(r.getnframes())
                pcm_i16 = np.frombuffer(frames, dtype=np.int16)
            
            # Convert to float32 [-1, 1]
            audio_f32 = pcm_i16.astype(np.float32) / 32768.0
            
            self.log.debug(f"Synthesized {len(audio_f32)} samples ({len(audio_f32)/self.sample_rate:.2f}s)")
            return audio_f32
            
        except Exception as e:
            self.log.error(f"Synthesis error: {e}")
            return None

    def synthesize_stream_raw(self, text: str) -> Iterator[bytes]:
        """
        Generate audio in chunks using Piper's streaming API.
        Yields raw audio bytes (int16) as they are generated.
        """
        if not text or not text.strip():
            self.log.warning("Empty text provided for streaming synthesis")
            return
        
        try:
            # Use Piper's streaming synthesis
            for audio_bytes in self.voice.synthesize_stream_raw(text, syn_config=self.syn_config):
                yield audio_bytes
                
        except Exception as e:
            self.log.error(f"Streaming synthesis error: {e}")

    def synthesize_streaming(self, text: str, amplitude_callback: Optional[Callable] = None) -> bool:
        """
        Synthesize and play audio in real-time (streaming mode).
        Audio plays as it's being generated, reducing perceived latency.
        
        Returns True on success, False on failure.
        """
        if not text or not text.strip():
            self.log.warning("Empty text provided for streaming synthesis")
            return False
        
        try:
            self.start_stream()
        except Exception as e:
            self.log.error(f"Failed to start audio stream: {e}")
            return False

        try:
            chunk_buffer = []
            chunk_size = 1024  # Samples per chunk for playback
            
            # Stream audio chunks from Piper
            for audio_chunk in self.synthesize_stream_raw(text):
                # Convert bytes to int16 array
                pcm_i16 = np.frombuffer(audio_chunk, dtype=np.int16)
                chunk_buffer.append(pcm_i16)
                
                # When we have enough samples, play them
                while len(chunk_buffer) > 0 and sum(len(c) for c in chunk_buffer) >= chunk_size:
                    # Concatenate buffer
                    audio_data = np.concatenate(chunk_buffer)
                    
                    # Take chunk_size samples
                    chunk = audio_data[:chunk_size]
                    remaining = audio_data[chunk_size:]
                    
                    # Clear buffer and add remaining
                    chunk_buffer = [remaining] if len(remaining) > 0 else []
                    
                    # Play chunk
                    try:
                        self.stream.write(chunk.tobytes())
                        
                        # Amplitude callback
                        if amplitude_callback:
                            try:
                                amplitude = np.abs(chunk.astype(np.float32)).mean()
                                amplitude_callback(amplitude)
                            except Exception as e:
                                self.log.warning(f"Amplitude callback error: {e}")
                    except Exception as e:
                        self.log.error(f"Error writing audio chunk: {e}")
                        return False
            
            # Play remaining audio in buffer
            if chunk_buffer:
                remaining_audio = np.concatenate(chunk_buffer)
                try:
                    self.stream.write(remaining_audio.tobytes())
                except Exception as e:
                    self.log.error(f"Error writing final audio chunk: {e}")
                    return False
            
            self.log.debug("Streaming synthesis complete")
            return True
            
        except Exception as e:
            self.log.error(f"Error during streaming synthesis: {e}")
            return False

    async def synthesize_streaming_async(self, text: str, amplitude_callback: Optional[Callable] = None) -> bool:
        """
        Async version of streaming synthesis.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.synthesize_streaming,
            text,
            amplitude_callback
        )

    def play_audio_with_amplitude(self, audio_data, amplitude_callback=None):
        """
        Play the given float32 numpy array (single-channel).
        If amplitude_callback is provided, pass the amplitude of each chunk.
        Returns True on success, False on failure.
        """
        if audio_data is None:
            self.log.warning("No audio data to play")
            return False
        
        # Convert to numpy array if needed
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        
        if len(audio_data) == 0:
            self.log.warning("Empty audio array")
            return False

        try:
            self.start_stream()
        except Exception as e:
            self.log.error(f"Failed to start audio stream: {e}")
            return False

        try:
            # Clip and convert to int16
            audio_int16 = np.clip(audio_data * 32767.0, -32767.0, 32767.0).astype(np.int16)

            chunk_size = 1024
            idx = 0
            total_frames = len(audio_int16)

            while idx < total_frames:
                chunk_end = min(idx + chunk_size, total_frames)
                chunk = audio_int16[idx:chunk_end]
                
                try:
                    self.stream.write(chunk.tobytes())
                except Exception as e:
                    self.log.error(f"Error writing audio chunk: {e}")
                    return False

                if amplitude_callback:
                    try:
                        amplitude = np.abs(chunk.astype(np.float32)).mean()
                        amplitude_callback(amplitude)
                    except Exception as e:
                        self.log.warning(f"Amplitude callback error: {e}")

                idx += chunk_size
            
            return True
            
        except Exception as e:
            self.log.error(f"Error during audio playback: {e}")
            return False

    async def play_audio_async(self, audio_data, amplitude_callback=None):
        """Async version of play_audio_with_amplitude."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.play_audio_with_amplitude,
            audio_data,
            amplitude_callback
        )

    def start_stream(self):
        """Start the audio stream if not already started."""
        if self.stream is not None:
            return  # Already started
        
        if self.pa is None:
            raise RuntimeError("PyAudio not initialized")
        
        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True
            )
            self.log.debug("Audio output stream started")
        except Exception as e:
            self.log.error(f"Failed to open audio stream: {e}")
            raise

    def stop_stream(self):
        """Stop the audio stream without terminating PyAudio."""
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.log.debug("Audio stream stopped")
            except Exception as e:
                self.log.error(f"Error stopping stream: {e}")
            finally:
                self.stream = None

    def stop_tts(self):
        """Stop the audio stream and clean up. Called externally on engine shutdown."""
        self.log.debug("Stopping TTS")
        
        # Stop stream first
        self.stop_stream()
        
        # Shutdown executor
        self.executor.shutdown(wait=False)
        
        # Terminate PyAudio
        if self.pa is not None:
            try:
                self.pa.terminate()
                self.log.debug("PyAudio terminated")
            except Exception as e:
                self.log.error(f"Error terminating PyAudio: {e}")
            finally:
                self.pa = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_tts()
        return False


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    from utils.utils import LoadModel
    
    model = LoadModel()
    
    try:
        with TTS(str(model.ensure_model("tts", "es_419-Octybot-medium.onnx")), 
                 str(model.ensure_model("tts", "es_419-Octybot-medium.onnx.json"))) as tts:
            
            print("Text-to-Speech test (Streaming Mode) - Press Ctrl+C to exit\n")
            print("Type 'stream' or 'normal' to switch modes\n")
            
            mode = "stream"
            
            while True:
                text = input("Write something: ")
                
                if text.lower() == "stream":
                    mode = "stream"
                    print("Switched to streaming mode")
                    continue
                elif text.lower() == "normal":
                    mode = "normal"
                    print("Switched to normal mode")
                    continue
                
                if text and text.strip():
                    if mode == "stream":
                        print("Synthesizing and playing (streaming)...")
                        success = tts.synthesize_streaming(text)
                        if not success:
                            print("Streaming playback failed")
                    else:
                        print("Synthesizing...")
                        audio = tts.synthesize(text)
                        if audio is not None:
                            print("Playing...")
                            success = tts.play_audio_with_amplitude(audio)
                            if not success:
                                print("Playback failed")
                        else:
                            print("Synthesis failed")
                else:
                    print("Please enter some text")
                    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")