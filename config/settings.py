import os
#Configuration file

"""Global"""
LANGUAGE = "es"
MODELS_PATH = "config/models.yml"

"""Audio Listener is the node to hear something from the MIC"""
AUDIO_LISTENER_DEVICE_ID: int | None = None #The system is prepared to detect the best device, but if you want to force a device, put the id here
AUDIO_LISTENER_CHANNELS = 1 # "mono" or "stereo"
AUDIO_LISTENER_SAMPLE_RATE = 16000
AUDIO_LISTENER_FRAMES_PER_BUFFER = 1000

"""Text-to-Speech"""
SAMPLE_RATE_TTS = 24000
DEVICE_SELECTOR_TTS = "cpu" # "cpu" or "cuda"
VOLUME_TTS = 2.0 #Volume  of the TTS
SPEED_TTS = 1.0 # 1.0 = Fast and 2.0 = slow
PATH_TO_SAVE_TTS = "tts/audios" #Specify the PATH where we are going to save the Info
NAME_OF_OUTS_TTS = "output" #This is the name that your file is going to revive Ex: test_0.wav -> A subfolder /test is gonna be created
SAVE_WAV_TTS = False

"""Speech-to-Text"""
SAMPLE_RATE_STT = 16000 #Whisper works at this sample_rate doesn't change unless it is necessary
SELF_VOCABULARY_STT = "Octybot" 