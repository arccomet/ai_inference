import base64
import time
from queue import Queue, Empty

from agent_tts import XttsTTS, SAMPLE_RATE
from agent_hear import AgentHear
from agent_logic import VoiceAgentLogic
from pydub import AudioSegment


class SimpleTurnBaseVoiceAgent:
    def __init__(self, low_vram=False):
        if low_vram:
            import agent_hear
            import agent_tts
            agent_hear.LOW_VRAM = False
            agent_tts.LOW_VRAM = True

        self.agent_logic = VoiceAgentLogic(self)
        self.agent_tts = XttsTTS()
        self.agent_hear = AgentHear()

        # Every request recieved is a "job"
        self.job_queue = Queue()
        self.result_queue = Queue()

        # Text output
        self.response_stream_queue = Queue()

    def process_job(self, job_obj):
        decoded_payload = base64.b64decode(job_obj["payload"])

        # Speech to text
        text_msg = self.agent_hear.get_transcription(decoded_payload)
        print("Transcribe result", text_msg)

        # LLM logic
        reply_text = self.agent_logic.receive_message(text_msg)

        # Text to speech
        try:
            tts_prompt = self.agent_logic.clean_up_text_for_tts(reply_text)
            tts_result_generator = self.agent_tts.tts_stream(tts_prompt, output_dir="wav/outputs")

            print("TTS => ", tts_prompt)

            for tts_output in tts_result_generator:
                output_path = tts_output[0]
                if not tts_output[1]:  # If tts is still not finished
                    wav_file = AudioSegment.from_wav(output_path)
                    wav_data = wav_file.raw_data
                    encoded_wav = base64.b64encode(wav_data).decode('utf-8')
                    audio_depth = wav_file.sample_width

                    result = {"message": f"TTS Chunk",
                              "text": "audio chunk",
                              "payload": encoded_wav,
                              "sampleRate": SAMPLE_RATE,
                              "depth": audio_depth
                              }
                    self.result_queue.put_nowait(result)
                else:
                    wav_file = AudioSegment.from_wav(output_path)
                    wav_data = wav_file.raw_data
                    encoded_wav = base64.b64encode(wav_data).decode('utf-8')
                    audio_depth = wav_file.sample_width

                    result = {"message": f"<response end>",
                              "text": reply_text,
                              "payload": encoded_wav,
                              "sampleRate": SAMPLE_RATE,
                              "depth": audio_depth
                              }
                    self.result_queue.put_nowait(result)
        # If error, we handle it by sending message saying that TTS failed, but you still get the reply text
        except Exception as e:
            result = {"message": f"TTS Failed.",
                      "text": str(e),
                      "payload": "",
                      "sampleRate": SAMPLE_RATE,
                      "depth": 0
                      }
            print(e)
            self.result_queue.put_nowait(result)

    def put_new_job(self, job_obj):
        self.job_queue.put_nowait(job_obj)
        return True

    def try_get_result(self):
        try:
            result = self.result_queue.get_nowait()
            return result
        except Empty:
            print("Queue is empty. No new message received.")
            return False

    def run_agent(self):
        while True:
            job_obj = self.job_queue.get()
            time.sleep(0.001)
            self.process_job(job_obj)
