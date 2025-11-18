import logging
import threading

from utils.utils import LoadModel
from stt.audio_listener import AudioListener
from stt.speech_to_text import SpeechToText
from tts.text_to_speech import TTS
from config.settings import AUDIO_LISTENER_FRAMES_PER_BUFFER

# TODO: import WakeWord and your LLM client here
# from wake_word.wake_word import WakeWord
# from llm.client import LLMClient


class OctyVoiceEngine:
    def __init__(self):
        self.log = logging.getLogger("OctyVoice")
        model = LoadModel()
        
        # Audio Listener
        self.audio_listener = AudioListener()

        # Speech to Text (Whisper)
        # Ensure WakeWord is imported or remove if unused
        # self.wake_word = WakeWord(str(model.ensure_model("wake_word")[0]))
        self.stt = SpeechToText(str(model.ensure_model("stt")[0]), "small")  # Other Model "base", id = 1

        # Text to Speech (Piper)
        self.tts = TTS(str(model.ensure_model("tts")[0]), str(model.ensure_model("tts")[1]))
        
        # TODO: create/assign your LLM instance to self.llm or remove usage
        # self.llm = LLMClient(...)

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
                        break
            finally:
                try:
                    self.audio_listener.stop_stream()
                except Exception:
                    pass

        t = threading.Thread(target=_record_loop, daemon=True)
        t.start()

        try:
            input(" Recording... Press Enter to stop.\n")
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
                text_transcribed = self.stt.transcribe_audio_bytes(audio_data)

                if text_transcribed:
                    print(f" Transcribed Text: {text_transcribed}")

                    # Echo logic
                    response = f"You said: {text_transcribed}"

                    # Synthesize and play response
                    wav_data = self.tts.synthesize(response)
                    self.tts.play_audio_with_amplitude(wav_data)
                else:
                    print(" No transcribed text.")
        except KeyboardInterrupt:
            print("\n--- Stopping OctyVoice ---\n")
            self.stop()

    def stop(self):
        try:
            self.audio_listener.delete()
        except Exception:
            pass
        try:
            self.tts.stop_tts()
        except Exception:
            pass
        self.log.info("OctyVoice Stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    engine = OctyVoiceEngine()
    engine.run()