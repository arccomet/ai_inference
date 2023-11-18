import os

import time
import torch
import torchaudio

# By using XTTS you agree to CPML license https://coqui.ai/cpml
# os.environ["COQUI_TOS_AGREED"] = "1"

import utils

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir


LOW_VRAM = False
SAMPLE_RATE = 24000


model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))

# This will trigger downloading model
print("Downloading if not downloaded Coqui XTTS V2")
from TTS.utils.manage import ModelManager
ModelManager().download_model(model_name)
print("XTTS downloaded")

# Config here
config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))
# supported_languages=["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh-cn"]
supported_languages = config.languages

import nltk
nltk.download('punkt')


def break_into_chunks(text, word_count):
    sentences = nltk.sent_tokenize(text)
    print(sentences)
    chunks = []
    current_chunk = []

    word_count_so_far = 0
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        print(sentence)
        print(len(current_chunk))
        if word_count_so_far + len(words) <= word_count or len(current_chunk) == 0:
            current_chunk.append(sentence)
            word_count_so_far += len(words)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk.clear()
            word_count_so_far = 0

            # Put into next chunk
            current_chunk.append(sentence)

    # Add the last chunk if it's not empty
    if len(current_chunk) > 0:
        chunks.append(' '.join(current_chunk))

    print(len(''.join(chunks)))

    return chunks


class XttsTTS:
    def __init__(self):
        self.model = None
        if not LOW_VRAM:
            self.load_model()

        self.speaker_wav = "wav/input_wavs/Kara_Voice_trimmed_5min.wav"

    def get_model(self):
        if LOW_VRAM:
            self.load_model()
            return self.model
        else:
            return self.model

    def load_model(self):
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config,
            checkpoint_path=os.path.join(model_path, "model.pth"),
            vocab_path=os.path.join(model_path, "vocab.json"),
            eval=True,
            use_deepspeed=False
        )
        self.model.cuda()

    def tts_full(self, input_prompt, output_dir):
        audio_chunks = []
        text_chunks = break_into_chunks(input_prompt, 16)

        for text_chunk in text_chunks:
            generator = self.predict(prompt=text_chunk, language="en",
                                audio_file_pth=self.speaker_wav)
            for data_output in generator:
                if data_output[0]:
                    audio_chunks.extend(data_output[0])

        wav = torch.cat(audio_chunks, dim=0)

        output_path = output_dir + f"/xtts_output_{utils.time_str()}.wav"
        torchaudio.save(output_path, wav.squeeze().unsqueeze(0).cpu(), 24000)

        if LOW_VRAM:
            del self.model
            torch.cuda.empty_cache()

        return output_path

    def tts_stream(self, input_prompt, output_dir):
        audio_chunks = []
        text_chunks = break_into_chunks(input_prompt, 20)

        print("TTS => ", text_chunks)

        for text_chunk in text_chunks:
            generator = self.predict(prompt=text_chunk, language="en", audio_file_pth=self.speaker_wav)
            for data_output in generator:
                if data_output[0]:
                    audio_chunks.extend(data_output[0])
                    chunk = data_output[0]
                    output_path = output_dir + f"/xtts_output_{utils.time_str()}.wav"
                    wav = torch.cat(chunk, dim=0)
                    torchaudio.save(output_path, wav.squeeze().unsqueeze(0).cpu(), SAMPLE_RATE)
                    yield output_path, 0
        if LOW_VRAM:
            del self.model
            torch.cuda.empty_cache()

        wav = torch.cat(audio_chunks, dim=0)
        output_path = output_dir + f"/xtts_output_{utils.time_str()}.wav"
        torchaudio.save(output_path, wav.squeeze().unsqueeze(0).cpu(), SAMPLE_RATE)
        yield output_path, 1

    def predict(self, prompt, language, audio_file_pth):
        print("Predicting")
        print(prompt)
        speaker_wav = audio_file_pth

        metrics_text = ""

        wav_chunks = []
        try:
            t_latent = time.time()
            try:
                (gpt_cond_latent, speaker_embedding) = self.get_model().get_conditioning_latents(
                    audio_path=speaker_wav,
                    gpt_cond_len=6,
                    max_ref_length=10,
                    sound_norm_refs=False,
                )
            except Exception as e:
                print("Speaker encoding error", str(e))
                print("It appears something wrong with reference, did you unmute your microphone?")
                return (
                    None,
                    None,
                    None,
                    None,
                )

            t_inference = time.time()

            chunks = self.get_model().inference_stream(
                prompt,
                language,
                gpt_cond_latent,
                speaker_embedding,
                stream_chunk_size=40)

            first_chunk = True
            for i, chunk in enumerate(chunks):
                if first_chunk:
                    first_chunk_time = time.time() - t_inference
                    metrics_text += f"Latency to first audio chunk: {round(first_chunk_time * 1000)} milliseconds\n"
                    first_chunk = False

                wav_chunks.append(chunk)
                print(f"Received chunk {i} of audio length {chunk.shape[-1]}")

                # out_file = f'{i}.wav'
                # write(out_file, 24000, chunk.detach().cpu().numpy().squeeze())
                # audio = AudioSegment.from_file(out_file)
                # audio.export(out_file, format='wav')

                # Stream output here --------------------------------------------------------
                yield None, None, metrics_text, None
                # Stream output here --------------------------------------------------------

        except RuntimeError as e:
            if "device-side assert" in str(e):
                # cannot do anything on cuda device side error, need tor estart
                print(f"Exit due to: Unrecoverable exception caused by language:{language} prompt:{prompt}", flush=True)
                print("Cuda device-assert Runtime encountered need restart")

                '''# just before restarting save what caused the issue so we can handle it in future
                # Uploading Error data only happens for unrecovarable error
                error_time = datetime.datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
                error_data = [error_time, prompt, language, audio_file_pth]
                error_data = [str(e) if type(e) != str else e for e in error_data]
                print(error_data)
                print(speaker_wav)
                write_io = StringIO()

                filename = error_time + "_xtts-stream_" + str(uuid.uuid4()) + ".csv"
                print("Writing error csv")

                # speaker_wav
                print("Writing error reference audio")
                speaker_filename = error_time + "_reference_xtts-stream_" + str(uuid.uuid4()) + ".wav"'''

                quit()
            else:
                if "Failed to decode" in str(e):
                    print("Speaker encoding error", str(e))
                else:
                    print("RuntimeError: non device-side assert error:", str(e))

                return None, None, None, None

        # wav = torch.cat(wav_chunks, dim=0)
        # torchaudio.save(f"output_{prompt[5:15]}.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)

        # wav_bytes_io = BytesIO()
        # torchaudio.save(wav_bytes_io, wav.squeeze().unsqueeze(0).cpu(), 24000)
        # wav_bytes = wav_bytes_io.getvalue()
        # base64_encoded = base64.b64encode(wav_bytes).decode('utf-8')

        yield wav_chunks, None, metrics_text, speaker_wav


if __name__ == '__main__':
    t = XttsTTS()
    t.tts_full("What are you uh... what are you doing? Hahahahaha.", "wav/outputs")




