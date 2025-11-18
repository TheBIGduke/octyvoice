# Configuration file

"""Global"""
LANGUAGE = "es"
MODELS_PATH = "config/models.yml"

"""Audio Listener - node to capture audio from microphone"""
AUDIO_LISTENER_DEVICE_ID: int | None = None  # Auto-detect best device, or specify device ID
AUDIO_LISTENER_CHANNELS = 1  # mono
AUDIO_LISTENER_SAMPLE_RATE = 16000
AUDIO_LISTENER_FRAMES_PER_BUFFER = 1000

"""Text-to-Speech"""
SAMPLE_RATE_TTS = 24000
VOLUME_TTS = 2.0  # Volume multiplier
SPEED_TTS = 1.0  # 1.0 = normal speed, >1.0 = slower
PATH_TO_SAVE_TTS = "tts/audios"  # Path to save audio files
NAME_OF_OUTS_TTS = "output"  # Prefix for output files
SAVE_WAV_TTS = False

"""Speech-to-Text"""
SAMPLE_RATE_STT = 16000  # Whisper works at 16kHz
SELF_VOCABULARY_STT = "Octybot"
