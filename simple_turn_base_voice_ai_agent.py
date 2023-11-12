import base64
import time
from queue import Queue, Empty

from agent_tts import XttsTTS
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
            tts_result_file_path = self.agent_tts.tts_full(self.agent_logic.clean_up_text_for_tts(reply_text),
                                                           output_dir="wav/outputs")
        except:
            tts_result_file_path = None

        if tts_result_file_path:
            wav_file = AudioSegment.from_wav(tts_result_file_path)
            wav_data = wav_file.raw_data
            encoded_wav = base64.b64encode(wav_data).decode('utf-8')

            result = {"message": f"Success.",
                      "text": reply_text,
                      "payload": encoded_wav,
                      "sampleRate": wav_file.frame_rate,
                      }
        else:
            result = {"message": f"TTS Failed.",
                      "text": reply_text,
                      "payload": "",
                      "sampleRate": 24000,
                      }

        self.result_queue.put_nowait(result)

    def put_new_job(self, job_obj):
        self.job_queue.put_nowait(job_obj)
        return True

    def try_get_result(self):
        print("Trying to get result")
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
