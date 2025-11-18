import logging
import threading
import sys

from utils.utils import LoadModel
from stt.audio_listener import AudioListener
from stt.speech_to_text import SpeechToText
from tts.text_to_speech import TTS
from config.settings import AUDIO_LISTENER_FRAMES_PER_BUFFER


class OctyVoiceEngine:
    def __init__(self):
        self.log = logging.getLogger("OctyVoice")
        model = LoadModel()
        
        # Audio Listener
        self.audio_listener = AudioListener()

        # Speech to Text (Whisper)
        self.stt = SpeechToText(str(model.ensure_model("stt")[0]), "small")  # Other Model "base", id = 1

        # Text to Speech (Piper)
        self.tts = TTS(str(model.ensure_model("tts")[0]), str(model.ensure_model("tts")[1]))

        self.log.info("OctyVoice Ready")

    def record_until_interrupt(self):
        """Records audio in a separate thread until Enter is pressed. Returns bytes."""
        frames = []
        stop_event = threading.Event()

        def _record_loop():
            try:
                self.audio_listener.start_stream()
                while not stop_event.is_set():
                    try:
                        data = self.audio_listener.read_frame(AUDIO_LISTENER_FRAMES_PER_BUFFER)
                        if data:
                            frames.append(data)
                    except Exception as e:
                        self.log.error(f"Recording error: {e}")
            except Exception as e:
                self.log.error(f"Failed to start recording: {e}")
            finally:
                try:
                    self.audio_listener.stop_stream()
                except Exception as e:
                    self.log.error(f"Failed to stop audio stream: {e}")

        # Start recording thread
        t = threading.Thread(target=_record_loop, daemon=True)
        t.start()

        try:
            input(" Recording... Press Enter to stop.\n")
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            stop_event.set()
            t.join()

        return b"".join(frames)

    def run(self):
        print("\n--- OctyVoice is running ---\n")
        try:
            while True:
                input("Press Enter to start recording\n")

                # Record
                audio_data = self.record_until_interrupt()

                if not audio_data:
                    print("No audio data recorded.")
                    continue

                print(" Processing...")

                # Transcribe
                text = self.stt.stt_from_bytes(audio_data)

                if text:
                    print(f" Transcribed Text: {text}")

                    # Echo logic
                    response_text = f"You said: {text}"

                    # Synthesize and play response
                    audio_out = self.tts.synthesize(response_text)
                    self.tts.play_audio_with_amplitude(audio_out)
                else:
                    print(" Could not understand the audio")

        except KeyboardInterrupt:
            print("\n--- Stopping OctyVoice ---\n")
            self.stop()
            sys.exit(0)

    def stop(self):
        try:
            self.audio_listener.delete()
            self.tts.stop_tts()
        except Exception:
            pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    engine = OctyVoiceEngine()
    engine.run()