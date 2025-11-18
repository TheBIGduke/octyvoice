# OctyVoice Engine

[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**OctyVoice Engine** is a lightweight Python package for real-time speech-to-text and text-to-speech conversion. It provides a simple offline voice pipeline that captures audio from your microphone, transcribes it using Whisper, and responds using Piper TTS. This is a simplified version focused solely on the STT-to-TTS core functionality.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Based On](#based-on)
- [Troubleshooting](#troubleshooting)

---

<h2 id="features">Features</h2>

- **Real-time Speech-to-Text** – Uses OpenAI Whisper for accurate transcription
- **Text-to-Speech Synthesis** – Piper TTS for natural-sounding voice output
- **Offline Operation** – No cloud dependencies, all processing runs locally
- **Simple Echo Pipeline** – Press Enter to record, automatic transcription and playback
- **Configurable Audio** – Adjust sample rates, volumes, and audio devices
- **Lightweight** – Minimal dependencies, focused on core functionality

---

<h2 id="installation">Installation</h2>

> [!IMPORTANT]
> This implementation was tested on Ubuntu 22.04 with Python 3.10.12

### Prerequisites
- Python 3.10 or higher
- Git
- System audio libraries (PortAudio, FFmpeg)

### Cloning this Repo
```bash
# Clone the repository
git clone https://github.com/TheBIGduke/octyvoice.git
cd octyvoice
```

### Setup

#### For automatic installation and setup:
```bash
bash installer.sh
```

The installer will:
1. Install system dependencies (PortAudio, FFmpeg)
2. Install yq for YAML processing
3. Create Python virtual environment
4. Install Python dependencies
5. Download required models (Whisper, Piper)

#### For manual installation:

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-dev python3-venv build-essential curl unzip
sudo apt install -y portaudio19-dev ffmpeg

# Install yq for model downloads
sudo snap install yq

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download models
bash utils/download_models.sh
```

You're done when you see:
```
Models ready.
```

---

<h2 id="configuration">Configuration</h2>

> [!WARNING]
> Audio models can be large. Ensure you have enough disk space (~500MB for models).

### Settings (`config/settings.py`)

All runtime settings are in **`config/settings.py`**. Edit the file and restart scripts to apply changes.

#### Global Settings
```python
LANGUAGE = "es"  # Language for Whisper transcription
```

#### Audio Listener Settings
```python
AUDIO_LISTENER_DEVICE_ID = None  # Auto-detect, or specify device ID
AUDIO_LISTENER_CHANNELS = 1      # Mono audio
AUDIO_LISTENER_SAMPLE_RATE = 16000
AUDIO_LISTENER_FRAMES_PER_BUFFER = 1000
```

#### Text-to-Speech Settings
```python
SAMPLE_RATE_TTS = 24000   # Piper TTS sample rate
VOLUME_TTS = 2.0          # Volume multiplier
SPEED_TTS = 1.0           # Speech speed (1.0 = normal)
SAVE_WAV_TTS = False      # Save audio files to disk
PATH_TO_SAVE_TTS = "tts/audios"
NAME_OF_OUTS_TTS = "output"
```

#### Speech-to-Text Settings
```python
SAMPLE_RATE_STT = 16000        # Whisper expects 16kHz
SELF_VOCABULARY_STT = "Octybot"  # Custom vocabulary hint
```

### Model Configuration (`config/models.yml`)

Define which models to download and use:

```yaml
stt:
  - name: small.pt
    url: "https://openaipublic.azureedge.net/main/whisper/models/..."

tts:
  - name: es_419-Octybot-medium.onnx
    url: "https://drive.google.com/uc?export=download&id=..."
  - name: es_419-Octybot-medium.onnx.json
    url: "https://drive.google.com/uc?export=download&id=..."
```

Models are cached in `~/.cache/Local-LLM-for-Robots/`

---

<h2 id="quick-start">Quick Start</h2>

```bash
cd octyvoice
source .venv/bin/activate
```

### Run the Voice Pipeline

Start the main program:
```bash
python main.py
```

**Usage:**
1. Press Enter to start recording
2. Speak into your microphone
3. Press Enter again to stop recording
4. The system will transcribe your speech
5. You'll hear "You said: [your text]" played back

### Test Individual Modules

#### Audio Listener
```bash
python -m stt.audio_listener
```

#### Speech-to-Text
```bash
python -m stt.speech_to_text
```

#### Text-to-Speech
```bash
python -m tts.text_to_speech
```

> [!TIP]
> If you have problems launching modules, try: `./.venv/bin/python -m stt.speech_to_text`

---

<h2 id="usage">Usage</h2>

### Basic Example

```python
from utils.utils import LoadModel
from stt.audio_listener import AudioListener
from stt.speech_to_text import SpeechToText
from tts.text_to_speech import TTS

# Initialize models
model = LoadModel()
audio_listener = AudioListener()
stt = SpeechToText(str(model.ensure_model("stt")[1]), "small")
tts = TTS(str(model.ensure_model("tts")[0]), str(model.ensure_model("tts")[1]))

# Record audio
audio_listener.start_stream()
# ... capture audio frames ...
audio_listener.stop_stream()

# Transcribe
text = stt.stt_from_bytes(audio_data)

# Synthesize and play
audio_out = tts.synthesize(text)
tts.play_audio_with_amplitude(audio_out)
```

### Recording Audio

```python
from stt.audio_listener import AudioListener

listener = AudioListener()
listener.start_stream()

# Capture frames
frames = []
for _ in range(100):
    data = listener.read_frame(1000)
    frames.append(data)

listener.stop_stream()
audio_bytes = b"".join(frames)
```

### Transcribing Audio

```python
from stt.speech_to_text import SpeechToText

stt = SpeechToText("path/to/model.pt", "small")
text = stt.stt_from_bytes(audio_bytes)
print(f"Transcribed: {text}")
```

### Synthesizing Speech

```python
from tts.text_to_speech import TTS

tts = TTS("path/to/model.onnx", "path/to/model.onnx.json")
audio = tts.synthesize("Hello world")
tts.play_audio_with_amplitude(audio)
```

---

<h2 id="project-structure">Project Structure</h2>

```
OctyVoice-Engine/
├── config/
│   ├── __init__.py
│   ├── models.yml          # Model catalog (download URLs)
│   └── settings.py         # Runtime configuration
│
├── stt/
│   ├── audio_listener.py   # Microphone audio capture
│   └── speech_to_text.py   # Whisper STT wrapper
│
├── tts/
│   └── text_to_speech.py   # Piper TTS synthesis and playback
│
├── utils/
│   ├── __init__.py
│   ├── download_models.sh  # Model download script
│   └── utils.py            # Model loading utilities
│
├── main.py                 # Main voice pipeline
├── requirements.txt        # Python dependencies
├── installer.sh            # Automatic setup script
├── .gitignore
└── README.md
```

### File Responsibilities

#### main.py
- **OctyVoiceEngine class** – Main pipeline coordinator
- **Recording logic** – Threaded audio capture with Enter key control
- **Echo functionality** – Transcribe → Synthesize → Play
- **Resource management** – Proper cleanup on exit

#### stt/audio_listener.py
- **PyAudio wrapper** – Simplified audio capture interface
- **Device detection** – Auto-select best microphone
- **Stream management** – Start/stop recording safely
- **Frame reading** – Efficient audio data buffering

#### stt/speech_to_text.py
- **Whisper integration** – Load and run OpenAI Whisper models
- **Audio preprocessing** – Convert raw bytes to float32 arrays
- **Transcription** – Convert speech to text with language support
- **Error handling** – Graceful failure on invalid audio

#### tts/text_to_speech.py
- **Piper TTS wrapper** – Text-to-speech synthesis
- **Audio playback** – Real-time streaming with PyAudio
- **Volume control** – Configurable amplitude
- **Speed adjustment** – Configurable speech rate
- **Optional saving** – Write WAV files to disk

#### utils/utils.py
- **Model management** – Load models from cache
- **YAML parsing** – Read model configuration
- **Path validation** – Ensure models exist before use

#### utils/download_models.sh
- **Automated downloads** – Fetch models from URLs
- **Cache management** – Store models in user cache directory
- **Dependency checking** – Verify yq and curl/wget availability

---

<h2 id="based-on"> Based On</h2>

This project is a streamlined derivative of [**Local-LLM-for-Robots**](https://github.com/JossueE/Local-LLM-for-Robots) by JossueE. The original repository provides a complete robot voice interaction system including wake word detection, LLM integration, and avatar visualization. 

**OctyVoice Engine** extracts and simplifies the core STT/TTS pipeline for users who need just the voice conversion functionality without the additional robot-specific features.

If you need the full robot interaction system, please visit the [original repository](https://github.com/JossueE/Local-LLM-for-Robots).

---

<h2 id="troubleshooting">Troubleshooting</h2>

### Models Not Found

**Error: "Model file does not exist"**

```bash
# Re-download models
bash utils/download_models.sh

# Check cache directory
ls ~/.cache/Local-LLM-for-Robots/stt/
ls ~/.cache/Local-LLM-for-Robots/tts/
```

---

### Audio Input Issues

**No audio detected or "Audio stream has not been started"**

1. **List available audio devices:**
   ```python
   import pyaudio
   pa = pyaudio.PyAudio()
   for i in range(pa.get_device_count()):
       info = pa.get_device_info_by_index(i)
       print(f"[{i}] {info['name']} (in={info['maxInputChannels']})")
   ```

2. **Specify device in settings.py:**
   ```python
   AUDIO_LISTENER_DEVICE_ID = 5  # Use your device index
   ```

3. **Check permissions:**
   ```bash
   # Linux: Add user to audio group
   sudo usermod -a -G audio $USER
   # Logout and login again
   ```

**Linux PulseAudio issues:**
```bash
# Restart PulseAudio
pulseaudio --kill
pulseaudio --start
```

---

### Transcription Problems

**Empty transcriptions or "Could not understand the audio"**

- **Record longer audio** – Whisper needs at least 1-2 seconds
- **Check microphone volume** – Speak louder or increase system volume
- **Verify language setting:**
  ```python
  # config/settings.py
  LANGUAGE = "es"  # Change to your language code
  ```
- **Try different Whisper model:**
  ```python
  # main.py - Change from "small" to "base"
  self.stt = SpeechToText(str(model.ensure_model("stt")[0]), "base")
  ```

---

### TTS Playback Issues

**No audio output or distorted sound**

1. **Check system volume** – Ensure speakers/headphones are working
2. **Adjust TTS volume in settings.py:**
   ```python
   VOLUME_TTS = 1.0  # Reduce if too loud
   ```
3. **Verify sample rate:**
   ```python
   SAMPLE_RATE_TTS = 24000  # Piper default
   ```
4. **Test PyAudio output:**
   ```python
   import pyaudio
   import numpy as np
   
   pa = pyaudio.PyAudio()
   stream = pa.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
   
   # Play test tone
   tone = (np.sin(2 * np.pi * 440 * np.arange(24000) / 24000) * 32767).astype(np.int16)
   stream.write(tone.tobytes())
   ```

---

### Performance Issues

**High CPU usage or slow transcription**

- **Use smaller Whisper model:**
  ```python
  # main.py - Use "base" instead of "small"
  self.stt = SpeechToText(str(model.ensure_model("stt")[0]), "base")
  ```
- **Reduce buffer size:**
  ```python
  # config/settings.py
  AUDIO_LISTENER_FRAMES_PER_BUFFER = 512  # From 1000
  ```

**Slow TTS synthesis**

- TTS uses CPU by default
- First synthesis is slow (model loading)
- Subsequent calls are faster

---

### Installation Issues

**Error: "No module named 'pyaudio'"**

```bash
# Install PortAudio development files first
sudo apt install portaudio19-dev

# Then install Python package
pip install pyaudio
```

**Error: "command 'yq' not found"**

```bash
sudo snap install yq
```

**Virtual environment activation fails**

```bash
# Recreate virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Common Errors

**KeyboardInterrupt not working**

- Press Ctrl+C twice
- Or use Ctrl+Z then `kill %1`

**"Recording... Press Enter to stop" stuck**

- Press Enter (not Space or other keys)
- Check terminal has focus
- Try clicking terminal window first

**Audio files keep growing (if SAVE_WAV_TTS = True)**

```python
# Disable saving in settings.py
SAVE_WAV_TTS = False

# Or clean up old files
rm -rf tts/audios/*
```