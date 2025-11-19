# OctyVoice Engine

[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)

**OctyVoice Engine** is a lightweight, async-first Python package for real-time speech-to-text and text-to-speech conversion. It provides a high-performance offline voice pipeline that captures audio from your microphone, transcribes it using Whisper, and responds using Piper TTS with streaming playback.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Based On](#based-on)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Asynchronous Architecture** – Non-blocking I/O for responsive performance
- **Real-time Speech-to-Text** – Uses OpenAI Whisper for accurate transcription
- **Streaming Text-to-Speech** – Piper TTS with low-latency streaming playback
- **Offline Operation** – No cloud dependencies, all processing runs locally
- **Simple Voice Pipeline** – Press Enter to record, automatic transcription and playback
- **Configurable Audio** – Adjust sample rates, volumes, and audio devices
- **Model Warmup** – Preloads models on startup for faster first-use performance
- **Lightweight** – Minimal dependencies, focused on core functionality

---

## Installation

> [!IMPORTANT]
> Tested on Ubuntu 22.04 with Python 3.10.12. Should work on most Linux distributions and Windows.

### Prerequisites
- Python 3.10 or higher
- Git
- System audio libraries (PortAudio, FFmpeg)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/TheBIGduke/octyvoice.git
cd octyvoice

# Run automatic installer
bash installer.sh
```

The installer will:
1. Install system dependencies (PortAudio, FFmpeg)
2. Install yq for YAML processing
3. Create Python virtual environment
4. Install Python dependencies
5. Download required models (Whisper, Piper)

### Manual Installation

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

Setup is complete when you see:
```
Models ready.
```

---

## Configuration

> [!WARNING]
> Audio models can be large. Ensure you have enough disk space (~500MB for models).

### Runtime Settings (`config/settings.py`)

All configuration is centralized in `config/settings.py`. Edit and restart to apply changes.

#### Global Configuration
```python
LANGUAGE = "es"  # Language for Whisper transcription (ISO 639-1 code)
MODELS_PATH = "config/models.yml"  # Path to model catalog
```

#### Audio Listener
```python
AUDIO_LISTENER_DEVICE_ID = None  # Auto-detect, or specify device ID
AUDIO_LISTENER_CHANNELS = 1      # Mono audio
AUDIO_LISTENER_SAMPLE_RATE = 16000  # 16kHz for Whisper
AUDIO_LISTENER_FRAMES_PER_BUFFER = 512  # Buffer size
```

#### Text-to-Speech
```python
SAMPLE_RATE_TTS = 24000       # Piper TTS sample rate
VOLUME_TTS = 2.0              # Volume multiplier (1.0 = normal)
LENGTH_SCALE_TTS = 1.0        # Speed (1.0 = normal, >1.0 = slower)
SAVE_WAV_TTS = False          # Save audio files to disk
PATH_TO_SAVE_TTS = "tts/audios"
NAME_OF_OUTS_TTS = "output"
```

#### Speech-to-Text
```python
SAMPLE_RATE_STT = 16000           # Whisper expects 16kHz
SELF_VOCABULARY_STT = "Octybot"   # Custom vocabulary hint
```

### Model Configuration (`config/models.yml`)

Define which models to download:

```yaml
stt:
  # Whisper models
  - name: base.pt
    url: "https://openaipublic.azureedge.net/main/whisper/models/..."
  - name: small.pt
    url: "https://openaipublic.azureedge.net/main/whisper/models/..."

tts:
  # Piper models
  - name: es_419-Octybot-medium.onnx
    url: "https://drive.google.com/uc?export=download&id=..."
  - name: es_419-Octybot-medium.onnx.json
    url: "https://drive.google.com/uc?export=download&id=..."
```

**Cache Location:**
- Default: `~/.cache/OctyVoice/`
- Custom: Set `OCTYVOICE_CACHE` environment variable

---

## Quick Start

```bash
cd octyvoice
source .venv/bin/activate
```

### Run the Voice Pipeline

```bash
python main.py
```

**Usage:**
1. Press **Enter** to start recording
2. Speak into your microphone
3. Press **Enter** again to stop
4. Listen to the transcription playback (streams as it generates)

Press **Ctrl+C** to exit.

### Test Individual Components

```bash
# Test audio capture (3 second recording, sync + async)
python -m stt.audio_listener

# Test text-to-speech (interactive prompt with streaming support)
python -m tts.text_to_speech

# Test speech-to-text (dummy audio test)
python -m stt.speech_to_text
```

> [!TIP]
> If module imports fail, try: `./.venv/bin/python -m stt.audio_listener`

---

## Usage Examples

### Basic Async Voice Pipeline

```python
import asyncio
from utils.utils import LoadModel
from stt.audio_listener import AudioListener
from stt.speech_to_text import SpeechToText
from tts.text_to_speech import TTS

async def main():
    # Initialize models
    model = LoadModel()
    audio_listener = AudioListener()

    stt_model = model.ensure_model("stt", "small.pt")
    stt = SpeechToText(str(stt_model), "small")

    tts_model = model.ensure_model("tts", "es_419-Octybot-medium.onnx")
    tts_config = model.ensure_model("tts", "es_419-Octybot-medium.onnx.json")
    tts = TTS(str(tts_model), str(tts_config))

    # Record audio asynchronously
    await audio_listener.start_stream_async()
    frames = []
    for _ in range(100):
        data = await audio_listener.read_frame_async(1024)
        frames.append(data)
    await audio_listener.stop_stream_async()

    # Transcribe
    audio_bytes = b"".join(frames)
    text = await stt.stt_from_bytes_async(audio_bytes)
    print(f"Transcribed: {text}")

    # Synthesize and play with streaming
    await tts.synthesize_streaming_async(text)

    # Cleanup
    await audio_listener.delete_async()
    tts.stop_tts()

asyncio.run(main())
```

### Recording Audio (Async)

```python
import asyncio
from stt.audio_listener import AudioListener

async def record():
    listener = AudioListener()
    await listener.start_stream_async()

    # Capture frames
    frames = []
    for _ in range(100):
        data = await listener.read_frame_async(1000)
        frames.append(data)

    await listener.stop_stream_async()
    audio_bytes = b"".join(frames)

    # Cleanup
    await listener.delete_async()
    return audio_bytes

asyncio.run(record())
```

### Transcribing Audio (Async)

```python
import asyncio
from stt.speech_to_text import SpeechToText
from utils.utils import LoadModel

async def transcribe(audio_bytes):
    model = LoadModel()
    stt_path = model.ensure_model("stt", "small.pt")
    stt = SpeechToText(str(stt_path), "small")

    # Transcribe from bytes
    text = await stt.stt_from_bytes_async(audio_bytes)
    print(f"Transcribed: {text}")
    
    stt.shutdown()
    return text

# Usage
asyncio.run(transcribe(audio_bytes))
```

### Synthesizing Speech with Streaming

```python
import asyncio
from tts.text_to_speech import TTS
from utils.utils import LoadModel

async def speak(text):
    model = LoadModel()
    tts_model = model.ensure_model("tts", "es_419-Octybot-medium.onnx")
    tts_config = model.ensure_model("tts", "es_419-Octybot-medium.onnx.json")

    tts = TTS(str(tts_model), str(tts_config))

    # Stream audio (plays while generating for lower latency)
    success = await tts.synthesize_streaming_async(text)
    print(f"Streaming playback: {'success' if success else 'failed'}")

    # Cleanup
    tts.stop_tts()

asyncio.run(speak("Hello world"))
```

### Using Amplitude Callback

```python
def on_amplitude(amp):
    """Called for each audio chunk during playback"""
    print(f"Current amplitude: {amp:.2f}")

# With streaming
tts.synthesize_streaming("Testing amplitude", amplitude_callback=on_amplitude)

# Or with regular playback
audio = tts.synthesize("Testing amplitude")
tts.play_audio_with_amplitude(audio, amplitude_callback=on_amplitude)
```

### Synchronous Usage (Still Supported)

```python
from stt.audio_listener import AudioListener
from stt.speech_to_text import SpeechToText
from tts.text_to_speech import TTS
from utils.utils import LoadModel

# All components support sync operations
model = LoadModel()
listener = AudioListener()
stt_path = model.ensure_model("stt", "base.pt")
stt = SpeechToText(str(stt_path), "base")

# Sync recording
listener.start_stream()
frames = [listener.read_frame(1024) for _ in range(100)]
listener.stop_stream()

# Sync transcription
audio = b"".join(frames)
text = stt.stt_from_bytes(audio)

# Sync TTS
tts_model = model.ensure_model("tts", "es_419-Octybot-medium.onnx")
tts_config = model.ensure_model("tts", "es_419-Octybot-medium.onnx.json")
tts = TTS(str(tts_model), str(tts_config))
tts.synthesize_streaming(text)  # Sync streaming

# Cleanup
listener.delete()
tts.stop_tts()
```

---

## Project Structure

```
octyvoice/
├── config/
│   ├── __init__.py
│   ├── models.yml          # Model catalog with download URLs
│   └── settings.py         # Runtime configuration
│
├── stt/                    # Speech-to-Text components
│   ├── audio_listener.py   # Async microphone capture (PyAudio wrapper)
│   └── speech_to_text.py   # Async Whisper transcription
│
├── tts/                    # Text-to-Speech components
│   └── text_to_speech.py   # Streaming Piper synthesis and playback
│
├── utils/
│   ├── __init__.py
│   ├── download_models.sh  # Automated model downloader
│   └── utils.py            # Model loading and validation
│
├── main.py                 # Async voice pipeline with streaming
├── requirements.txt        # Python dependencies
├── installer.sh            # Automated setup script
├── .gitignore
└── README.md
```

### Component Responsibilities

**main.py** – `OctyVoiceEngine` class
- Async event loop orchestration
- Non-blocking audio recording with Enter key control
- Concurrent model warmup on startup
- Streaming transcription and synthesis flow
- Graceful resource cleanup with async shutdown

**stt/audio_listener.py** – `AudioListener` class
- Async PyAudio wrapper for microphone input
- Auto-detects audio devices (prefers PulseAudio on Linux)
- Non-blocking stream lifecycle management
- ThreadPoolExecutor for blocking I/O operations
- Both sync and async APIs supported

**stt/speech_to_text.py** – `SpeechToText` class
- Loads Whisper models from cache
- Async transcription using ThreadPoolExecutor
- Converts raw audio bytes (int16) to float32 arrays
- Language and vocabulary support
- Handles minimum audio length validation

**tts/text_to_speech.py** – `TTS` class
- Piper TTS voice synthesis
- **Streaming playback** – Audio plays while generating (lower latency)
- Real-time audio output via PyAudio
- Configurable volume and speech speed
- Optional WAV file saving
- Amplitude callback support for visualizations
- Both sync and async streaming APIs

**utils/utils.py** – `LoadModel` class
- Parses `config/models.yml` configuration
- Resolves model paths in cache directory
- Validates model existence before use
- Lists available models and validates cache

**utils/download_models.sh**
- Downloads models from URLs in `models.yml`
- Stores in `~/.cache/OctyVoice/` (or `$OCTYVOICE_CACHE`)
- Skips already downloaded files
- Uses curl or wget with retry logic

---

## Based On

This project is derived from [**Local-LLM-for-Robots**](https://github.com/JossueE/Local-LLM-for-Robots) by JossueE. The original repository provides a complete robot voice interaction system including wake word detection, LLM integration, and avatar visualization.

**OctyVoice Engine** extracts and modernizes the core STT/TTS pipeline with:
- Fully async architecture for better performance
- Streaming TTS for reduced latency
- Enhanced error handling and logging
- Improved device detection
- Simplified API for users who need just voice conversion functionality

For the full system, visit the [original repository](https://github.com/JossueE/Local-LLM-for-Robots).

---

## Troubleshooting

### Models Not Found

**Error:** `FileNotFoundError: Model file does not exist`

```bash
# Re-download models
bash utils/download_models.sh

# Verify cache directory
ls ~/.cache/OctyVoice/stt/
ls ~/.cache/OctyVoice/tts/

# Check custom cache location
echo $OCTYVOICE_CACHE
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
       if info['maxInputChannels'] > 0:
           print(f"[{i}] {info['name']} (channels={info['maxInputChannels']})")
   ```

2. **Specify device in `config/settings.py`:**
   ```python
   AUDIO_LISTENER_DEVICE_ID = 5  # Use your device index
   ```

3. **Check permissions (Linux):**
   ```bash
   sudo usermod -a -G audio $USER
   # Logout and login again
   ```

4. **Test microphone:**
   ```bash
   arecord -d 5 test.wav
   aplay test.wav
   ```

**PulseAudio issues (Linux):**
```bash
# Restart PulseAudio
pulseaudio --kill
pulseaudio --start
```

---

### Transcription Problems

**Empty transcriptions or "Could not understand the audio"**

- **Record longer audio** – Whisper needs at least 0.5-1 seconds of speech (automatic warning)
- **Check microphone volume** – Speak louder or increase system input gain
- **Verify language setting:**
  ```python
  # config/settings.py
  LANGUAGE = "es"  # Change to your language code
  ```
- **Try different Whisper model:**
  ```python
  # main.py - Change model size
  stt_model_path = model.ensure_model("stt", "small.pt")  # Instead of base.pt
  self.stt = SpeechToText(str(stt_model_path), "small")
  ```

**Async transcription timeout:**
- The async implementation uses ThreadPoolExecutor with no timeout
- If transcription hangs, check if the audio data is valid
- Long audio files may take time – this is expected behavior

---

### TTS Playback Issues

**No audio output or distorted sound**

1. **Check system volume** – Ensure speakers/headphones are working
   ```bash
   speaker-test -t wav -c 2
   ```

2. **Adjust TTS volume in `config/settings.py`:**
   ```python
   VOLUME_TTS = 1.0  # Reduce from 2.0 if too loud
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
   stream = pa.open(format=pyaudio.paInt16, channels=1, 
                    rate=24000, output=True)
   
   # Play 440Hz tone for 1 second
   tone = (np.sin(2*np.pi*440*np.arange(24000)/24000)*32767).astype(np.int16)
   stream.write(tone.tobytes())
   stream.close()
   pa.terminate()
   ```

**Streaming playback stuttering:**
- Increase buffer size in `start_stream()` (default: 2048 frames)
- Reduce system load (close other applications)
- Check CPU usage during playback

---

### Performance Issues

**High CPU usage or slow transcription**

- Use smaller Whisper model: `base.pt` instead of `small.pt`
- The async architecture already optimizes CPU usage
- First transcription after startup may be slower (model loading)
- Consider warmup phase: models are preloaded in background

**Slow TTS synthesis**

- First synthesis is slower (model loading + warmup)
- Subsequent calls are much faster
- Streaming mode reduces perceived latency
- This is expected behavior with CPU-based inference

**Async/await issues:**
- Ensure you're using `asyncio.run()` for top-level async code
- Don't mix blocking operations in async functions
- Use `asyncio.to_thread()` or executors for CPU-bound tasks

---

### Installation Issues

**Error: `No module named 'pyaudio'`**

```bash
# Install PortAudio development files first
sudo apt install portaudio19-dev
pip install pyaudio
```

**Error: `command 'yq' not found`**

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

**Windows-specific issues:**

```python
# Windows requires specific event loop policy (handled automatically in main.py)
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

---

### Common Runtime Errors

**`KeyboardInterrupt` not working**

- Press Ctrl+C twice if needed
- The async implementation handles cleanup gracefully
- Wait for "Cleanup complete" message

**"Recording... Press Enter to stop" stuck**

- Press **Enter** (not Space or other keys)
- Ensure terminal window has focus
- The async recording loop should respond immediately

**Audio files accumulating (if `SAVE_WAV_TTS = True`)**

```python
# Disable saving in config/settings.py
SAVE_WAV_TTS = False

# Or clean up old files
rm -rf tts/audios/*
```

**Error: `RuntimeError: Audio stream has not been started`**

- Ensure `start_stream()` or `start_stream_async()` is called first
- Check that stream didn't fail to start due to device issues
- Verify device_index is valid

**Async context warnings:**

```python
# Always use async context properly
async with AudioListener() as al:
    await al.start_stream_async()
    # ... work ...
    await al.stop_stream_async()
```

**ThreadPoolExecutor warnings on shutdown:**
- Normal behavior during cleanup
- Executors are shut down with `wait=False` for faster exit
- No action needed if cleanup completes successfully

---