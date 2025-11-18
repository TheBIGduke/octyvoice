import pyaudio
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from config.settings import (
    AUDIO_LISTENER_DEVICE_ID,
    AUDIO_LISTENER_SAMPLE_RATE,
    AUDIO_LISTENER_CHANNELS,
    AUDIO_LISTENER_FRAMES_PER_BUFFER
)


def define_device_id(pa: pyaudio.PyAudio = None, preferred: int = AUDIO_LISTENER_DEVICE_ID, log: logging.Logger = None) -> int:
    """Define the device ID to use for audio input"""
    if preferred is not None:
        # Validate the preferred device exists and has input channels
        if pa is not None:
            try:
                info = pa.get_device_info_by_index(preferred)
                if info.get('maxInputChannels', 0) > 0:
                    if log:
                        log.info(f"Using specified device [{preferred}]: {info['name']}")
                    return preferred
                else:
                    if log:
                        log.warning(f"Specified device [{preferred}] has no input channels, auto-detecting...")
            except Exception as e:
                if log:
                    log.warning(f"Specified device [{preferred}] not found: {e}, auto-detecting...")
        else:
            return preferred
    
    if pa is None:
        if log:
            log.warning("PyAudio instance not provided, cannot auto-detect device.")
        return None

    # Auto-detect best input device
    pulse_device = None
    first_input_device = None
    
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)   
            max_input = info.get('maxInputChannels', 0)
            
            if max_input > 0:
                if log:
                    log.info(f"[{i}] {info['name']} (in={max_input}, rate={int(info.get('defaultSampleRate', 0))})")
                
                # Remember first input device as fallback
                if first_input_device is None:
                    first_input_device = i
                
                # Prefer PulseAudio on Linux
                if info['name'].lower() == "pulse" or "pulse" in info['name'].lower():
                    pulse_device = i
                    if log:
                        log.info(f"Found PulseAudio device: {i}")
        except Exception as e:
            if log:
                log.warning(f"Error checking device {i}: {e}")
            continue
    
    # Return best available device
    if pulse_device is not None:
        if log:
            log.info(f"Using PulseAudio device by default: {pulse_device}")
        return pulse_device
    elif first_input_device is not None:
        if log:
            log.info(f"Using first available input device: {first_input_device}")
        return first_input_device
    else:
        if log:
            log.error("No input devices found!")
        return None


class AudioListener:
    def __init__(self):
        self.log = logging.getLogger("AudioListener")  
        self.sample_rate = AUDIO_LISTENER_SAMPLE_RATE
        self.audio_interface = None
        self.stream = None
        self.device_index = None
        self.channels = AUDIO_LISTENER_CHANNELS 
        self.frames_per_buffer = AUDIO_LISTENER_FRAMES_PER_BUFFER
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize PyAudio
        try:
            self.audio_interface = pyaudio.PyAudio()
        except Exception as e:
            self.log.error(f"Failed to initialize PyAudio: {e}")
            raise
        
        # Detect device
        self.device_index = define_device_id(self.audio_interface, AUDIO_LISTENER_DEVICE_ID, self.log)
        
        if self.device_index is None:
            self.audio_interface.terminate()
            raise RuntimeError("No suitable audio input device found. Please check your microphone.")
        
        self.log.info(f"AudioListener initialized (device={self.device_index}, rate={self.sample_rate}Hz, channels={self.channels})")

    def start_stream(self):
        """Start the audio stream if not already started"""
        if self.stream is not None:
            self.log.warning("Stream already started")
            return
        
        try:
            self.stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.frames_per_buffer,
            )
            self.log.debug("Audio stream started")
        except Exception as e:
            self.log.error(f"Failed to start audio stream: {e}")
            self.stream = None
            raise

    async def start_stream(self):
        """Async version of start_stream"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.start_stream)

    def read_frame(self, frame_samples: int) -> bytes:
        """Read a frame of audio data from the stream"""
        if self.stream is None:
            raise RuntimeError("Audio stream has not been started. Call start_stream() first")
        
        try:
            return self.stream.read(frame_samples, exception_on_overflow=False)
        except Exception as e:
            self.log.error(f"Error reading audio frame: {e}")
            raise

    async def read_frame(self, frame_samples: int) -> bytes:
        """Async version of read_frame. Returns audio data from stream"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.read_frame, frame_samples)

    def stop_stream(self):
        """Stop the audio stream if it is running"""
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.log.debug("Audio stream stopped")
            except Exception as e:
                self.log.error(f"Error stopping stream: {e}")
            finally:
                self.stream = None

    async def stop_stream(self):
        """Async version of stop_stream"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.stop_stream)

    def delete(self):
        """Clean up the audio interface and stream"""
        self.log.debug("Deleting AudioListener")
        
        if self.stream is not None:
            self.stop_stream()
        
        # Shutdown executor
        self.executor.shutdown(wait=False)
        
        if self.audio_interface is not None:
            try:
                self.audio_interface.terminate()
                self.log.debug("PyAudio terminated")
            except Exception as e:
                self.log.error(f"Error terminating PyAudio: {e}")
            finally:
                self.audio_interface = None

    async def delete(self):
        """Async version of delete"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.delete)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.delete()
        return False


# Example Usage
if __name__ == "__main__":
    import time
    
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    # Test synchronous version
    print("Testing AudioListener (sync)...")
    try:
        with AudioListener() as al:
            test_duration = 3
            print(f"Recording for {test_duration} seconds...")
            
            al.start_stream()
            start_time = time.time()
            
            frames = []
            while time.time() - start_time < test_duration:
                data = al.read_frame(1600)  # 0.1 seconds at 16kHz
                frames.append(data)
            
            al.stop_stream()
            
            total_bytes = sum(len(f) for f in frames)
            print(f"Recorded {total_bytes} bytes in {test_duration} seconds.")
    except Exception as e:
        print(f"Sync test failed: {e}")
    
    # Test async version
    print("\nTesting AudioListener (async)...")
    
    async def test():
        al = AudioListener()
        try:
            test_duration = 3
            print(f"Recording for {test_duration} seconds (async)...")
            
            await al.start_stream()
            start_time = time.time()
            
            frames = []
            while time.time() - start_time < test_duration:
                data = await al.read_frame(1600)
                frames.append(data)
            
            await al.stop_stream()
            
            total_bytes = sum(len(f) for f in frames)
            print(f"Recorded {total_bytes} bytes in {test_duration} seconds (async).")
            print("AudioListener async is working correctly.")
        except Exception as e:
            print(f"Async test failed: {e}")
        finally:
            await al.delete()
    
    asyncio.run(test())