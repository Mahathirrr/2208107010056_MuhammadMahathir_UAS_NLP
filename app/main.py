import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import tempfile

# Import modules for STT, LLM, and TTS
from .stt import transcribe_speech_to_text
from .llm import generate_response
from .tts import transcribe_text_to_speech

app = FastAPI(title="Voice Chatbot API")

@app.get("/")
def read_root():
    return {"message": "Voice Chatbot API is running"}

@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):
    # Read uploaded audio file
    audio_content = await file.read()

    # Step 1: Convert speech to text using Whisper
    text_transcript = transcribe_speech_to_text(audio_content, file_ext=".wav")

    if text_transcript.startswith("[ERROR]"):
        return {"error": text_transcript}

    # Step 2: Generate response using Gemini LLM
    llm_response = generate_response(text_transcript)

    if llm_response.startswith("[ERROR]"):
        return {"error": llm_response}

    # Step 3: Convert text response to speech using Coqui TTS
    audio_response_path = transcribe_text_to_speech(llm_response)

    if audio_response_path.startswith("[ERROR]"):
        return {"error": audio_response_path}

    # Return the audio file
    return FileResponse(
        path=audio_response_path,
        media_type="audio/wav",
        filename="response.wav"
    )