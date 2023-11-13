import base64
import threading
import time
import wave

from pydub import AudioSegment
import sounddevice as sd
import io

from audio_recorder import AudioRecorder
import requests

import multiprocessing


url = "http://46.188.73.96:40175/receive_audio"
check_result_url = "http://46.188.73.96:40175/try_get_response"


class AudioPlayer:
    def __init__(self):
        self.run_thread = threading.Thread(target=self.run_audio_player)
        # self.run_thread.daemon = True
        # self.run_thread.start()

    def run_audio_player(self):
        while True:
            time.sleep(3)
            data = self.try_get_audio()
            if data.get("payload"):
                decoded_payload = base64.b64decode(data["payload"])
                audio_io = io.BytesIO(decoded_payload)
                save_path = "wav/example_server_results/result.wav"
                with wave.open(save_path, "wb") as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(4)  # 32-bit audio
                    wav_file.setframerate(int(data["sampleRate"])/2)  # Sample rate (adjust as needed)
                    wav_file.writeframes(audio_io.read())

                play_audio(save_path)

    @staticmethod
    def try_get_audio():
        # JSON data to send
        data = {"message": "Try get result",
                "text": "GET"}

        # Send POST request
        response = requests.post(check_result_url, json=data)

        if response.status_code == 200:
            response_data = response.json()
            return response_data
        else:
            print(f'Error: {response.text}')


def play_audio(file_path):
    print(f"playing {file_path}")

    audio = AudioSegment.from_wav(file_path)

    # Extract raw audio data as a NumPy array
    audio_data = audio.get_array_of_samples()

    sd.play(audio_data, audio.frame_rate)
    sd.wait()


def make_api_call(audio_duration):
    audio_file_path = recorder.audio_sample_to_wav(audio_duration)

    wav_file = AudioSegment.from_wav(audio_file_path)
    wav_data = wav_file.raw_data
    encoded_wav = base64.b64encode(wav_data).decode('utf-8')

    # JSON data to send
    data = {"message": "Audio file sent from python client.",
            "payload": encoded_wav}

    # Send POST request
    response = requests.post(url, json=data)

    if response.status_code == 200:
        response_data = response.json()
        return response_data
    else:
        print(f'Error: {response.text}')


def user_loop():
    record_start_time = time.time()
    is_recording = False

    while True:
        if input():
            if is_recording:
                make_api_call(time.time()-record_start_time)
                is_recording = False
                print("Sent")
            else:
                record_start_time = time.time()
                print("Start")
                is_recording = True


if __name__ == '__main__':
    recorder = AudioRecorder()
    time.sleep(1)

    player = AudioPlayer()

    user_thread = threading.Thread(target=user_loop)
    user_thread.start()

    player.run_audio_player()

