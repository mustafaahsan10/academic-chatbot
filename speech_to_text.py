import os
import io
import numpy as np
import time
import threading
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import pydub
import queue
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Load environment variables
load_dotenv()

class SpeechTranscriber:
    def __init__(self, model_size="gpt-4o-transcribe"):
        """Initialize the speech transcriber with the specified model."""
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        self.model = model_size
        self.is_recording = False
        self.audio_data = pydub.AudioSegment.empty()
        self.webrtc_ctx = None
        
    def create_webrtc_streamer(self, key="speech-transcriber"):
        """Create and return WebRTC streamer context"""
        self.webrtc_ctx = webrtc_streamer(
            key=key,
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"audio": True},
            async_processing=True,
        )
        return self.webrtc_ctx
        
    def start_recording(self, webrtc_ctx=None):
        """
        Start recording audio from the microphone.
        Returns True if recording started successfully, False otherwise.
        """
        if self.is_recording:
            print("Already recording")
            return False
            
        try:
            # Use provided webrtc_ctx or existing one
            if webrtc_ctx:
                self.webrtc_ctx = webrtc_ctx
                
            if not self.webrtc_ctx:
                st.error("WebRTC context not available")
                return False
            
            self.audio_data = pydub.AudioSegment.empty()
            self.is_recording = True
            
            print("Recording started...")
            return True
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False
    
    def collect_audio_frames(self):
        """
        Collect audio frames from WebRTC while recording.
        Call this repeatedly while recording to accumulate audio.
        """
        if not self.webrtc_ctx or not self.webrtc_ctx.audio_receiver or not self.is_recording:
            return False
            
        try:
            audio_frames = self.webrtc_ctx.audio_receiver.get_frames(timeout=0.1)
            
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                self.audio_data += sound
                
            return len(audio_frames) > 0
            
        except queue.Empty:
            return False
        except Exception as e:
            print(f"Error collecting audio: {e}")
            return False
        
    def stop_recording(self, temp_file="temp_audio.wav"):
        """
        Stop recording and save the audio to a file.
        
        Args:
            temp_file: Path to save the temporary audio file
            
        Returns:
            Path to the recorded audio file or None if no audio was recorded
        """
        if not self.is_recording:
            print("Not recording")
            return None
            
        try:
            # Stop recording
            self.is_recording = False
            
            # Collect any remaining frames
            for _ in range(10):  # Try multiple times to get remaining frames
                if self.webrtc_ctx and self.webrtc_ctx.audio_receiver:
                    try:
                        audio_frames = self.webrtc_ctx.audio_receiver.get_frames(timeout=0.1)
                        for audio_frame in audio_frames:
                            sound = pydub.AudioSegment(
                                data=audio_frame.to_ndarray().tobytes(),
                                sample_width=audio_frame.format.bytes,
                                frame_rate=audio_frame.sample_rate,
                                channels=len(audio_frame.layout.channels),
                            )
                            self.audio_data += sound
                    except queue.Empty:
                        break
                    except Exception:
                        break
            
            print("Recording stopped.")
            
            # Check if we got any audio data
            if len(self.audio_data) == 0:
                print("No audio data captured")
                return None
                
            # Check minimum duration (at least 0.5 seconds)
            duration_ms = len(self.audio_data)
            if duration_ms < 500:
                print(f"Audio too short: {duration_ms}ms")
                return None
                
            # Convert to mono and save the recorded audio
            audio_mono = self.audio_data.set_channels(1)
            audio_mono.export(temp_file, format="wav")
            
            print(f"Audio saved: {duration_ms}ms duration")
            return temp_file
            
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return None
    
    def transcribe_audio(self, audio_file, prompt="", response_format="text"):
        """
        Transcribe an audio file using OpenAI's speech-to-text API.
        
        Args:
            audio_file: Path to the audio file
            prompt: Optional prompt to guide the transcription
            response_format: Format of the response (text or json)
            
        Returns:
            Transcribed text
        """
        if not audio_file or not os.path.exists(audio_file):
            return ""
            
        try:
            print(f"Transcribing with OpenAI's {self.model}...")
            
            with open(audio_file, "rb") as file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model=self.model,
                    file=file,
                    response_format=response_format,
                    prompt=prompt
                )
            
            # Clean up the temporary file
            if os.path.exists(audio_file):
                os.remove(audio_file)
                  # Return the transcription text
            # If response format is "text", the response is already a string
            if response_format == "text":
                return transcription
            # Otherwise, access the text property
            return transcription.text
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""
    
    def get_audio_duration(self):
        """Get the current duration of recorded audio in seconds"""
        return len(self.audio_data) / 1000.0 if self.audio_data else 0.0

# For testing
if __name__ == "__main__":
    transcriber = SpeechTranscriber()
    print("Press Enter to start recording...")
    input()
    transcriber.start_recording()
    print("Press Enter to stop recording...")
    input()
    audio_file = transcriber.stop_recording()
    if audio_file:
        text = transcriber.transcriber.transcribe_audio(audio_file)
        print(f"Transcription: {text}")
        
    else:
        print("No audio recorded.")