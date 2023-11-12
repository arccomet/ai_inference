import pyaudio
import time
from pydub import AudioSegment
from multiprocessing import Process
import multiprocessing
import ctypes


class AudioRecorder:
    def __init__(self):
        self.frame_size = 2048  # Record in chunks of 2048 samples
        self.sample_format = pyaudio.paInt16  # 16 bits per sample
        self.channels = 1
        self.sample_rate = 16_000  # 44100  # Record at 44100 samples per second
        self.audio_buffer_duration = 30     # 30 seconds of audio
        self.shared_array = multiprocessing.Array(ctypes.c_char, self.audio_buffer_duration * 2 * self.sample_rate)

        self.record_process = Process(target=self.record_audio)
        self.record_process.start()  # Start the process

    def record_audio(self):
        print('>> Recording <<')

        p = pyaudio.PyAudio()
        stream = p.open(format=self.sample_format,
                        channels=self.channels,
                        rate=self.sample_rate,
                        frames_per_buffer=self.frame_size//2,
                        input=True)

        while True:
            data = stream.read(self.frame_size//2, exception_on_overflow=False)
            self.shared_array.value = self.shared_array[len(data):] + data

    def audio_sample_to_wav(self, sample_duration: int):
        print("Saving wav")
        buffer = self.shared_array.raw

        num_frames = self.sample_rate / self.frame_size  # per second
        buffer_length = len(buffer)
        sample_length = int(self.sample_rate * sample_duration * 2)
        if buffer_length >= sample_length:
            start_frame = int((self.audio_buffer_duration - sample_duration) * num_frames * self.frame_size)
            end_frame = int(self.audio_buffer_duration * num_frames * self.frame_size)
            sample_buffer = buffer[start_frame * 2:end_frame * 2]
        else:
            print(f"{sample_duration} secs of audio exceeds max sample duration, "
                  f"we don't store that long of an audio in the memory")
            sample_buffer = buffer

        audio_clip = AudioSegment(
            sample_buffer,
            sample_width=2,  # 2 bytes for a 16-bit sample
            frame_rate=self.sample_rate,
            channels=1  # Mono audio
        )
        filename = f"wav/temp{time.time()-sample_duration}to{time.time()}.wav"

        audio_clip.export(filename, format="wav")
        return filename


if __name__ == '__main__':
    a = AudioRecorder()
    while True:
        input()
        a.audio_sample_to_wav(5)


