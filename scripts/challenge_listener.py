#--------------------------------Imports--------------------------------------
import pvporcupine
import pyaudio
from threading import Thread
import struct
from dotenv import load_dotenv
import os
#----------------------------------------------------------------------------

#-----------------------------Access key handling-----------------------------
load_dotenv()  # Load environment variables from .env

access_key = os.getenv("PORCUPINE_ACCESS_KEY")
#-----------------------------------------------------------------------------

def start_audio_listener(trigger_event, keyword_path):
    """
    Starts a background thread to listen for the wake word using Porcupine.
    
    Args:
        trigger_event: multiprocessing.Event instance to signal detection.
        keyword_path: Path to your .ppn keyword file (e.g., 'challenge_en_windows.ppn')
    """

    porcupine = pvporcupine.create( # Load specific model
        access_key=access_key,
        keyword_paths=[keyword_path]
    )

    #---------------------Open Microphone Input---------------------
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )
    #---------------------------------------------------------------

    def listen():
        try:
            while True:
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False) # Pulse-Code Modulation, Get small chunk of raw audio
                pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm) # Formats audio for porcupine to process

                keyword_index = porcupine.process(pcm_unpacked) # Listens for keyword in audio
                if keyword_index >= 0: # Keyword detected
                    print("Keyword detected!")
                    trigger_event.set()

        except Exception as e:
            print(f"Audio listener error: {e}")

        finally:
            #---------Close Audio Streams---------
            audio_stream.stop_stream()
            audio_stream.close()
            pa.terminate()
            porcupine.delete()
            #-------------------------------------

    Thread(target=listen, daemon=True).start()
