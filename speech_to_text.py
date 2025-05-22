import os
import tempfile
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

class SpeechTranscriber:
    def __init__(self, model_size="gpt-4o-transcribe"):
        """Initialize the speech transcriber with the specified model."""
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        self.model = model_size
        self.is_recording = False
        self.audio_data = None
        
    def start_recording(self):
        """
        Start recording audio from the microphone.
        Returns True if recording started successfully, False otherwise.
        """
        if self.is_recording:
            print("Already recording")
            return False
            
        try:
            self.audio_data = None
            self.is_recording = True
            
            print("Recording started...")
            return True
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False
    
    def collect_audio_frames(self):
        """
        Placeholder for compatibility - audio_input handles recording automatically
        """
        return True
        
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
            
            print("Recording stopped.")
            
            # Check if we got any audio data from the widget
            if self.audio_data is None:
                print("No audio data captured")
                return None
                
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                # audio_data is an UploadedFile (BytesIO subclass)
                self.audio_data.seek(0)  # Reset file pointer
                tmp_file.write(self.audio_data.read())
                temp_file = tmp_file.name
            
            print(f"Audio saved to: {temp_file}")
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
        if self.audio_data:
            # For UploadedFile, we can get the size but not exact duration
            # Return a placeholder
            return 1.0  # Placeholder duration
        return 0.0

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
        text = transcriber.transcribe_audio(audio_file)
        print(f"Transcription: {text}")
        
    else:
        print("No audio recorded.")
    
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
        if self.audio_data:
            # For audio bytes, we can't easily determine duration without decoding
            # Return a placeholder or estimate
            return len(self.audio_data) / 16000  # Rough estimate assuming 16kHz sample rate
        return 0.0

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