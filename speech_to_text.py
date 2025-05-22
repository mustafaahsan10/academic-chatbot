import os
import io
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
        
    def transcribe_audio_bytes(self, audio_bytes, prompt="", response_format="text"):
        """
        Transcribe audio bytes using OpenAI's speech-to-text API.
        
        Args:
            audio_bytes: Audio data as bytes
            prompt: Optional prompt to guide the transcription
            response_format: Format of the response (text or json)
            
        Returns:
            Transcribed text
        """
        if not audio_bytes:
            return ""
            
        try:
            print(f"Transcribing with OpenAI's {self.model}...")
            
            # Create a temporary file-like object
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"  # OpenAI API needs a filename
            
            transcription = self.openai_client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                response_format=response_format,
                prompt=prompt
            )
            
            # Return the transcription text
            if response_format == "text":
                return transcription
            return transcription.text
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""
    
    def transcribe_audio_file(self, audio_file_path, prompt="", response_format="text"):
        """
        Transcribe an audio file using OpenAI's speech-to-text API.
        
        Args:
            audio_file_path: Path to the audio file
            prompt: Optional prompt to guide the transcription
            response_format: Format of the response (text or json)
            
        Returns:
            Transcribed text
        """
        if not audio_file_path or not os.path.exists(audio_file_path):
            return ""
            
        try:
            print(f"Transcribing with OpenAI's {self.model}...")
            
            with open(audio_file_path, "rb") as file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model=self.model,
                    file=file,
                    response_format=response_format,
                    prompt=prompt
                )
            
            # Return the transcription text
            if response_format == "text":
                return transcription
            return transcription.text
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""

# For testing
if __name__ == "__main__":
    transcriber = SpeechTranscriber()
    print("Audio transcriber initialized successfully!")