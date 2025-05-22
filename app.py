import streamlit as st
import os
from dotenv import load_dotenv
import logging
import tempfile
import time
import io

# Import the base query classifier
from base.query_classifier import classify_query_sync

# Import RAG pipeline
from modules.classified_chatbot import rag_pipeline_simple

from Library.DB_endpoint import db_endpoint

# Import speech transcriber
from speech_to_text import SpeechTranscriber

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="University Assistant",
    page_icon="🎓",
    layout="wide"
)

# Define available models
AVAILABLE_MODELS = {
    "OpenAI GPT-4o Mini($0.15/M)": "gpt-4o-mini",
    "OpenAI GPT-4.1 Mini($0.40/M)": "gpt-4.1-mini",
    "OpenAI GPT-4.1-nano($0.10/M)": "gpt-4.1-nano",
    "Anthropic Claude 3 Haiku($0.25/M)": "claude-3-haiku",
    "Anthropic Claude 3 Sonnet($3/M)": "claude-3-sonnet",
    "Anthropic Claude 3 Opus($15/M)": "claude-3-opus",
    "Google Gemini 2.0 Flash($0.10/M)": "gemini-2.0-flash-001",
}

# Define which models should use OpenRouter instead of OpenAI API
OPENROUTER_MODELS = [
    "claude-3-haiku",
    "claude-3-sonnet",
    "claude-3-opus",
    "gemini-2.0-flash-001",
]

# Module map
MODULES = {
    "course_information": {
        "name": "Course Information",
        "collection_name": "admission_course_guide_json",
        "description": "Information about course content, prerequisites, credit hours, etc."
    },
    "class_schedules": {
        "name": "Class Schedules",
        "collection_name": "class_schedule_json",
        "description": "Details about when and where classes meet"
    },
    "exam_alerts": {
        "name": "Exam Data",
        "collection_name": "exam_data_json",
        "description": "Information about exam dates, deadlines, and assessments"
    },
    "study_resources": {
        "name": "Study Resources",
        "collection_name": "study_resource_json",
        "description": "Materials for studying including textbooks and online resources"
    },
    "professors": {
        "name": "Professors",
        "collection_name": "professor_data_json",
        "description": "Faculty information, office hours, and contact details"
    },
    "library": {
        "name": "Library",
        "collection_name": "library_data_json",
        "description":"Information about available books and library resources"
    }
}

def get_module_response(query: str, language: str = "English") -> str:
    """
    Route the query to the appropriate module and get a response using RAG pipeline
    
    Args:
        query: The user's question
        language: The language for the response
        
    Returns:
        Formatted response string
    """
    try:
        # First, classify the query to determine which module should handle it
        classification = classify_query_sync(query)
        module_name = classification.module
        confidence = classification.confidence
        reasoning = classification.reasoning
        
        logger.info(f"Query classified as '{module_name}' with confidence {confidence}")
        
        # Get the response function for the module
        # Special handling for library module
        if module_name == "library":
            logger.info("Processing query through library orchestrator")
            try:
                # Import the library orchestrator
                from Library import process_library_query
                
                
                # Process the library query through the orchestrator with the model
                result = process_library_query(query)

                if result is not None:
                    response = result["response"]
                    return response
            except Exception as e:
                logger.error(f"Error in library orchestrator: {e}", exc_info=True)
                return "Sorry, there was an error processing your library query."
        
        # Special handling for professors module - check if it's a meeting request
        if module_name == "professors":
            logger.info("Processing query through professors orchestrator")
            try:
                # Import the professor orchestrator
                from modules.professors import process_professor_query
                
                # Check if query is about scheduling a meeting
                if module_name in MODULES:
                    collection_name = MODULES[module_name]["collection_name"]
                else:
                    collection_name = "professor_data_json"
                
                result = process_professor_query(query, collection_name)
                
                # If it's a meeting request, return the generated response
                if result is not None and result.get("is_meeting_request", False):
                    response = result["response"]
                    
                    # Add debug info if in development
                    if os.getenv("APP_ENV") == "development":
                        debug_info = f"\n\n---\nDebug: Query classified as '{module_name}' (confidence: {confidence:.2f})\n"
                        debug_info += f"Meeting intent detected (confidence: {result.get('confidence', 0):.2f})\n"
                        debug_info += f"Reasoning: {result.get('reasoning', '')}"
                        response += debug_info
                    
                    return response
                
                # If not a meeting request, continue with RAG pipeline below
                logger.info("Not a meeting request, using regular RAG pipeline")
            except Exception as e:
                logger.error(f"Error in professors orchestrator: {e}", exc_info=True)
                # Continue with RAG pipeline if there's an error
        
        # For other modules or non-meeting professor queries, use the RAG pipeline
        # Determine which collection to use
        if module_name in MODULES:
            collection_name = MODULES[module_name]["collection_name"]
        else:
            # Fallback to study resources if module not found
            logger.warning(f"Module '{module_name}' not found, falling back to study resources")
            collection_name = MODULES["study_resources"]["collection_name"]
        
        # Modify query to include language preference
        language_prefix = ""
        if language.lower() == "arabic":
            language_prefix = "Please respond in Arabic: "
        
        modified_query = language_prefix + query
        
        # Get the selected model from session state
        model = st.session_state.get("model", "openai/gpt-4o-mini")
        
        # Check if we need to prepend the provider name for OpenRouter models
        if st.session_state.get("use_openrouter", False):
            # For OpenRouter models that aren't OpenAI, need to prepend provider
            if "gpt" not in model:
                if "claude" in model:
                    model = f"anthropic/{model}"
                elif "gemini" in model:
                    model = f"google/{model}"
            else:
                model = f"openai/{model}"
        
        # Use RAG pipeline to get response
        result = rag_pipeline_simple(modified_query, collection_name, model)
        response = result["response"]
        
        # Add debug info if in development
        debug_info = ""
        if os.getenv("APP_ENV") == "development":
            debug_info = f"\n\n---\nDebug: Query classified as '{module_name}' (confidence: {confidence:.2f})\nReasoning: {reasoning}"
        
        return response + debug_info
            
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        if language.lower() == "arabic":
            return "عذرًا، حدث خطأ أثناء معالجة سؤالك. يرجى المحاولة مرة أخرى لاحقًا."
        else:
            return "I'm sorry, an error occurred while processing your question. Please try again later."

# Initialize the speech transcriber
@st.cache_resource(ttl="1h")
def get_speech_transcriber():
    return SpeechTranscriber()

# Main app
def main():
    # Custom CSS to remove button gaps
    st.markdown("""
    <style>
    /* Remove padding around buttons */
    div.stButton > button {
        margin: 0;
        padding: 0.4rem 1rem;
    }
    
    /* Adjust button container spacing */
    div.row-widget.stButton {
        padding: 0;
        margin: 0;
    }
    
    /* Hide the audio input widget */
    .stAudioInput {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎓 University Assistant")
    
    # Initialize session state
    if "recording" not in st.session_state:
        st.session_state.recording = False
        
    if "transcriber" not in st.session_state:
        st.session_state.transcriber = get_speech_transcriber()
        
    if "audio_processed" not in st.session_state:
        st.session_state.audio_processed = False
        
    if "current_audio_key" not in st.session_state:
        st.session_state.current_audio_key = 0
    
    # Main content area (upper section)
    main_area = st.container()
    
    # Chat and controls area (bottom section)
    bottom_container = st.container()
    
    # Main content area with app information
    with main_area:
        # Sidebar with settings only (no audio input)
        with st.sidebar:
            st.subheader("Settings")
            language = st.selectbox("Language / اللغة", ["English", "Arabic"])
            
            # Model selection dropdown - default to first model
            model_options = list(AVAILABLE_MODELS.keys())
            selected_model_name = st.selectbox(
                "Select AI Model",
                model_options,
                index=0
            )
            
            selected_model_id = AVAILABLE_MODELS[selected_model_name]
            
            # Store model ID and whether it's an OpenRouter model in session state
            if "model" not in st.session_state or st.session_state.model != selected_model_id:
                st.session_state.model = selected_model_id
                st.session_state.use_openrouter = selected_model_id in OPENROUTER_MODELS
            
            # Clear conversation button
            if st.button("Clear Conversation"):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()
        
        st.subheader("University Information Assistant")
        st.write("Ask me anything about courses, schedules, exams, faculty, library resources, admission, or tuition!")
        
        # Display model being used
        st.caption(f"Currently using: {selected_model_name}")
        
        # Initialize chat history in session state if needed
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Recording status indicator
        recording_status = st.empty()
        
        # Hidden audio input widget (we'll trigger it programmatically)
        audio_container = st.container()
        with audio_container:
            audio_data = st.audio_input(
                "Record your question", 
                key=f"audio_recorder_{st.session_state.current_audio_key}"
            )
    
    # Bottom container for chat history and input controls
    with bottom_container:
        # Display a separator line
        st.markdown("---")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Create a layout for input and buttons
        col_input, col_buttons = st.columns([6, 1])
        
        # Text input
        with col_input:
            prompt = st.chat_input("Ask me a question...")
            
        # Buttons side by side
        with col_buttons:
            button_cols = st.columns(2)
            
            # Start button
            with button_cols[0]:
                start_button = st.button("🎤", key="start_rec", disabled=st.session_state.recording)
            
            # Stop button
            with button_cols[1]:
                stop_button = st.button("⏹️", key="stop_rec", disabled=not st.session_state.recording)
    
    # Handle start recording
    if start_button:
        st.session_state.recording = True
        st.session_state.audio_processed = False
        recording_status.markdown("🔴 **Recording... Click stop when done**")
        st.rerun()
    
    # Handle stop recording
    if stop_button:
        st.session_state.recording = False
        recording_status.markdown("⏸️ **Processing...**")
        st.rerun()
    
    # Process audio if available and not yet processed
    if audio_data and not st.session_state.recording and not st.session_state.audio_processed:
        st.session_state.audio_processed = True
        
        with st.spinner("Transcribing audio..."):
            try:
                # Save audio data to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    audio_data.seek(0)
                    tmp_file.write(audio_data.read())
                    temp_file = tmp_file.name
                
                # Transcribe the audio
                transcribed_text = st.session_state.transcriber.transcribe_audio(temp_file)
                
                if transcribed_text:
                    # Add to session state message history
                    st.session_state.messages.append({"role": "user", "content": transcribed_text})
                    
                    # Generate response
                    with st.spinner("Generating response..."):
                        response = get_module_response(transcribed_text, language=language)
                    
                    # Update message history with response
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Increment audio key to reset the widget
                    st.session_state.current_audio_key += 1
                    
                    # Clear recording status
                    recording_status.empty()
                    
                    # Rerun to show new messages
                    st.rerun()
                else:
                    st.warning("Could not transcribe audio. Please try again.")
                    recording_status.empty()
                    
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                logger.error(f"Audio processing error: {e}", exc_info=True)
                recording_status.empty()
    
    # Handle text input
    if prompt:
        # Add to session state message history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show a loading spinner while generating response
        with st.spinner("Generating response..."):
            # Generate response
            response = get_module_response(prompt, language=language)
        
        # Update message history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the UI with the new messages
        st.rerun()

if __name__ == "__main__":
    main()