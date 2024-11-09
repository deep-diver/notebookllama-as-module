import io
import json
import torch
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from pydub import AudioSegment

from transformers import BarkModel, AutoProcessor, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

def get_parler_model(model_name="parler-tts/parler-tts-mini-v1", device="cpu"):
    """
    Get the Parler TTS model and tokenizer
    """
    parler_model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
    parler_tokenizer = AutoTokenizer.from_pretrained(model_name)
    return {
        "tokenizer": parler_tokenizer, 
        "model": parler_model
    }

def get_bark_model(model_name="suno/bark-small", device="cpu"):
    """
    Get the Bark TTS model and processor
    """
    bark_processor = AutoProcessor.from_pretrained(model_name)
    bark_model = BarkModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    return {
        "processor": bark_processor,
        "model": bark_model,
        "sampling_rate": 24000
    }

def generate_speaker1_audio(text, parler_model, parler_tokenizer, device="cpu"):
    """Generate audio using ParlerTTS for Speaker 1"""

    speaker1_description = """Laura's voice is expressive and dramatic in delivery, speaking at a moderately fast pace with a very close recording that almost has no background noise."""

    input_ids = parler_tokenizer(speaker1_description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = parler_tokenizer(text, return_tensors="pt").input_ids.to(device)
    generation = parler_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr, parler_model.config.sampling_rate

def generate_speaker2_audio(text, bark_processor, bark_model, bark_sampling_rate, device="cpu"):
    """Generate audio using Bark for Speaker 2"""

    inputs = bark_processor(text, voice_preset="v2/en_speaker_6").to(device)
    speech_output = bark_model.generate(**inputs, temperature=0.9, semantic_temperature=0.8)
    audio_arr = speech_output[0].cpu().numpy()
    return audio_arr, bark_sampling_rate

def numpy_to_audio_segment(audio_arr, sampling_rate):
    """Convert numpy array to AudioSegment"""
    # Convert to 16-bit PCM
    audio_int16 = (audio_arr * 32767).astype(np.int16)

    # Create WAV file in memory
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sampling_rate, audio_int16)
    byte_io.seek(0)

    # Convert to AudioSegment
    return AudioSegment.from_wav(byte_io)

def script_to_speech(conversations, device="cpu"):
    parler_model = get_parler_model(device=device)
    bark_model = get_bark_model(device=device)

    final_audio = None

    for conversation in tqdm(conversations, desc="Generating podcast segments", unit="segment"):
        speaker1_text = conversation["Alex"]
        speaker2_text = conversation["Jamie"]

        audio_arr1, rate1 = generate_speaker1_audio(
            speaker1_text, parler_model["model"], parler_model["tokenizer"]
        )
        audio_arr2, rate2 = generate_speaker2_audio(
            speaker2_text, bark_model["processor"], bark_model["model"], bark_model["sampling_rate"]
        )

        audio_segment1 = numpy_to_audio_segment(audio_arr1, rate1)
        audio_segment2 = numpy_to_audio_segment(audio_arr2, rate2)

        audio_segment = audio_segment1 + audio_segment2

        # Add to final audio
        if final_audio is None:
            final_audio = audio_segment
        else:
            final_audio += audio_segment

    return final_audio
