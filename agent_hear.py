import time

import whisperx
import torch
import numpy as np


LOW_VRAM = False


class AgentHear:
    def __init__(self):
        if LOW_VRAM:
            self.model = None
        else:
            self.model = self.load_whisper_model()
        self.whisper_batch_size = 16

    @staticmethod
    def load_whisper_model():
        return whisperx.load_model("large-v2", "cuda", compute_type="float16", language="en")

    def get_transcription(self, audio_data):
        audio = np.frombuffer(audio_data, np.int16).flatten().astype(np.float32) / 32768.0
        # whisperx.load_audio(audio_file)
        if LOW_VRAM:
            self.model = self.load_whisper_model()
        transcribe_result = self.model.transcribe(audio, batch_size=self.whisper_batch_size)["segments"]

        transcribe_result_text = ""
        if transcribe_result:
            transcribe_result_text = transcribe_result[0]['text']

        print(transcribe_result_text)

        if LOW_VRAM:
            del self.model
            torch.cuda.empty_cache()

        return transcribe_result_text.strip()

    def pipe(self):
        transcribe_result = self.model.transcribe("final_xtts_stream_output.mp3", batch_size=self.whisper_batch_size)["segments"]
        transcribe_result_text = ""
        if transcribe_result:
            transcribe_result_text = transcribe_result[0]['text']

        print(transcribe_result_text)

        return transcribe_result_text


if __name__ == '__main__':
    a = AgentHear()
    time.sleep(6)
    while True:
        text = input()
        if text.isnumeric():
            a.get_transcription(int(text))
