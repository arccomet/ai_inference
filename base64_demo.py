import base64
from pydub import AudioSegment
import wave
import sounddevice as sd

wav_file = AudioSegment.from_wav("wav/outputs/final_xtts_stream_output.wav")
wav_data = wav_file.raw_data
encoded_wav = base64.b64encode(wav_data).decode('utf-8')

data = {"payload": encoded_wav}

decoded_payload = base64.b64decode(data["payload"])
# audio_io = io.BytesIO(decoded_payload)
save_path = "wav/example_server_results/demo_result.wav"
with wave.open(save_path, "wb") as wav_file:
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 32-bit audio
    wav_file.setframerate(24000)  # Sample rate (adjust as needed)
    wav_file.writeframes(decoded_payload)

print(f"playing {save_path}")

audio = AudioSegment.from_wav(save_path)

# Extract raw audio data as a NumPy array
audio_data = audio.get_array_of_samples()

# Play the audio using sounddevice
sd.play(audio_data, audio.frame_rate)
sd.wait()  # Wait until playback is finished
