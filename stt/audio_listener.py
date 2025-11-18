import pyaudio
import logging

from config.settings import (
    AUDIO_LISTENER_DEVICE_ID,
    AUDIO_LISTENER_SAMPLE_RATE,
    AUDIO_LISTENER_CHANNELS,
    AUDIO_LISTENER_FRAMES_PER_BUFFER
)


def define_device_id(pa: pyaudio.PyAudio = None, preferred: int = AUDIO_LISTENER_DEVICE_ID, log: logging.Logger = None) -> int:
    """Define the device ID to use for audio input."""
    if preferred is not None:
        return preferred
    
    if pa is None:
        if log:
            log.warning("PyAudio instance not started, cannot list devices.")
        return None

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)   
        if info.get('maxInputChannels', 0) > 0:
            if log:
                log.info(f"[{i}] {info['name']} (in={info['maxInputChannels']}, rate={int(info.get('defaultSampleRate', 0))})")
            if info['name'].lower() == "pulse":
                if log:
                    log.info(f"Using PulseAudio device by default: {i}")
                return i
    return None


class AudioListener:
    def __init__(self):
        self.log = logging.getLogger("AudioListener")  
        self.sample_rate = AUDIO_LISTENER_SAMPLE_RATE
        self.audio_interface = pyaudio.PyAudio()
        self.device_index = define_device_id(self.audio_interface, AUDIO_LISTENER_DEVICE_ID, self.log)
        self.channels = AUDIO_LISTENER_CHANNELS 
        self.frames_per_buffer = AUDIO_LISTENER_FRAMES_PER_BUFFER
        self.stream = None
        self.log.info(f"AudioListener initialized (device={self.device_index}, rate={self.sample_rate}Hz, channels={self.channels})")

    def start_stream(self):
        """Start the audio stream if not already started."""
        if self.stream is None:
            self.stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.frames_per_buffer,
            )

    def read_frame(self, frame_samples: int) -> bytes:
        """Read a frame of audio data from the stream."""
        if self.stream is None:
            raise RuntimeError("Audio stream has not been started.")
        return self.stream.read(frame_samples, exception_on_overflow=False)

    def stop_stream(self):
        """Stop the audio stream if it is running."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def delete(self):
        """Clean up the audio interface and stream."""
        if self.stream is not None:
            self.stop_stream()
        self.audio_interface.terminate()


# Example Usage
if __name__ == "__main__":
    import time
    
    al = AudioListener()
    test_duration = 3
    
    al.start_stream()
    time.sleep(test_duration)
    data = al.read_frame(3200)
    
    print(f"During {test_duration} seconds, read {len(data)} bytes. AudioListener is working correctly.")
    al.stop_stream()
