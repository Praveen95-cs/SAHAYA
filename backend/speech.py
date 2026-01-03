from google.cloud import speech
import io

client = speech.SpeechClient()

def speech_to_text(audio_path: str) -> str:
    with io.open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-IN",
        enable_automatic_punctuation=True
    )

    response = client.recognize(config=config, audio=audio)

    if not response.results:
        return ""

    return response.results[0].alternatives[0].transcript
