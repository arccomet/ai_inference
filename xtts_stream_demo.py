import os
import uuid

import time
import torch
import torchaudio

# By using XTTS you agree to CPML license https://coqui.ai/cpml
os.environ["COQUI_TOS_AGREED"] = "1"

import base64
import csv
from io import StringIO
import datetime

from scipy.io.wavfile import write
from pydub import AudioSegment
# import sounddevice as sd
import numpy as np

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))

'''# This will trigger downloading model
print("Downloading if not downloaded Coqui XTTS V2")
from TTS.utils.manage import ModelManager


ModelManager().download_model(model_name)
print("XTTS downloaded")'''


config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=os.path.join(model_path, "model.pth"),
    vocab_path=os.path.join(model_path, "vocab.json"),
    eval=True,
    use_deepspeed=False
)
model.cuda()

# This is for debugging purposes only
DEVICE_ASSERT_DETECTED = 0
DEVICE_ASSERT_PROMPT = None
DEVICE_ASSERT_LANG = None

# supported_languages=["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh-cn"]
supported_languages = config.languages


def predict(prompt, language, audio_file_pth):
    """
    yields: final_result_file, chunk_file, text, speaker_wav
    """


    print("Predicting")
    print(prompt)
    speaker_wav = audio_file_pth

    global DEVICE_ASSERT_DETECTED
    if DEVICE_ASSERT_DETECTED:
        global DEVICE_ASSERT_PROMPT
        global DEVICE_ASSERT_LANG
        # It will likely never come here as we restart space on first unrecoverable error now
        print(f"Unrecoverable exception caused by language:{DEVICE_ASSERT_LANG} prompt:{DEVICE_ASSERT_PROMPT}")

    metrics_text = ""

    wav_chunks = []
    try:
        t_latent = time.time()
        try:
            (gpt_cond_latent, speaker_embedding) = model.get_conditioning_latents(
                audio_path=speaker_wav,
                gpt_cond_len=6,
                max_ref_length=10,
                sound_norm_refs=False,
            )
            # old (v1.1) gpt_cond_latent, _, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav)
        except Exception as e:
            print("Speaker encoding error", str(e))
            print("It appears something wrong with reference, did you unmute your microphone?")
            return (
                None,
                None,
                None,
                None,
            )

        latent_calculation_time = time.time() - t_latent
        ##metrics_text=f"Embedding calculation time: {latent_calculation_time:.2f} seconds\n"

        t_inference = time.time()

        chunks = model.inference_stream(
            prompt,
            language,
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=80)

        first_chunk = True
        for i, chunk in enumerate(chunks):
            if first_chunk:
                first_chunk_time = time.time() - t_inference
                metrics_text += f"Latency to first audio chunk: {round(first_chunk_time * 1000)} milliseconds\n"
                first_chunk = False

            wav_chunks.append(chunk)
            print(f"Received chunk {i} of audio length {chunk.shape[-1]}")

            out_file = f'{i}.wav'
            write(out_file, 24000, chunk.detach().cpu().numpy().squeeze())
            audio = AudioSegment.from_file(out_file)
            audio.export(out_file, format='wav')

            # Stream output here --------------------------------------------------------
            print(out_file)
            yield None, out_file, metrics_text, None
            # Stream output here --------------------------------------------------------

    except RuntimeError as e:
        if "device-side assert" in str(e):
            # cannot do anything on cuda device side error, need tor estart
            print(f"Exit due to: Unrecoverable exception caused by language:{language} prompt:{prompt}", flush=True)
            print("Cuda device-assert Runtime encountered need restart")
            if not DEVICE_ASSERT_DETECTED:
                DEVICE_ASSERT_DETECTED = 1
                DEVICE_ASSERT_PROMPT = prompt
                DEVICE_ASSERT_LANG = language

            # just before restarting save what caused the issue so we can handle it in future
            # Uploading Error data only happens for unrecovarable error
            error_time = datetime.datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
            error_data = [error_time, prompt, language, audio_file_pth]
            error_data = [str(e) if type(e) != str else e for e in error_data]
            print(error_data)
            print(speaker_wav)
            write_io = StringIO()
            csv.writer(write_io).writerows(error_data)
            csv_upload = write_io.getvalue().encode()

            filename = error_time + "_xtts-stream_" + str(uuid.uuid4()) + ".csv"
            print("Writing error csv")

            # speaker_wav
            print("Writing error reference audio")
            speaker_filename = error_time + "_reference_xtts-stream_" + str(uuid.uuid4()) + ".wav"

            quit()
        else:
            if "Failed to decode" in str(e):
                print("Speaker encoding error", str(e))
            else:
                print("RuntimeError: non device-side assert error:", str(e))

            return (
                None,
                None,
                None,
                None,
            )

    wav = torch.cat(wav_chunks, dim=0)
    torchaudio.save(f"output_{prompt[5:15]}.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)

    # second_of_silence = AudioSegment.silent()  # use default
    # second_of_silence.export("sil.wav", format='wav')

    yield wav_chunks, None, metrics_text, speaker_wav


import nltk
nltk.download('punkt')


def break_into_chunks(text, word_count):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []

    word_count_so_far = 0
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        if word_count_so_far + len(words) <= word_count:
            current_chunk.append(sentence)
            word_count_so_far += len(words)
        else:
            if len(current_chunk) > 0:  # Ensures we have a complete sentence
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                word_count_so_far = len(words)

    # Add the last chunk if it's not empty
    if len(current_chunk) > 0:
        chunks.append(' '.join(current_chunk))

    print(len(''.join(chunks)))

    return chunks


'''def play_audio(file_path):
    audio = AudioSegment.from_wav(file_path)

    # Extract raw audio data as a NumPy array
    audio_data = audio.get_array_of_samples()

    # Play the audio using sounddevice
    sd.play(audio_data, audio.frame_rate)
    sd.wait()  # Wait until playback is finished'''


if __name__ == '__main__':
    input_prompt = """Assume that x11 is initialized to 11 and x12 is initialized to 22. Suppose you executed the code below on a version of the pipeline from Section 4.6 that does not handle data hazards (i.e., the programmer is responsible for addressing data hazards by inserting NOP instructions where necessary). What would the final values of register x15 be? Assume the register file is written at the beginning of the cycle and read at the end of a cycle. Therefore, an ID stage will return the results of a WB state occurring during the same cycle. See Section 4.8 and Figure 4.68 for details."""

    audio_chunks = []
    text_chunks = break_into_chunks(input_prompt, 20)

    for text_chunk in text_chunks:
        generator = predict(prompt=text_chunk, language="en", audio_file_pth="wav/input_wavs/Kara_Voice_trimmed_5min.wav")
        for data_output in generator:
            # if data_output[1]:
                # play_audio(data_output[1])
            if data_output[0]:
                audio_chunks.extend(data_output[0])

    second_of_silence = AudioSegment.silent()
    wav = torch.cat(audio_chunks, dim=0)
    torchaudio.save("final_xtts_stream_output.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)

