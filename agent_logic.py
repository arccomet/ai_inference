import asyncio
import utils
import textgen_api
import re


BASE_CONTEXT = """You are Chad. You are based."""


class VoiceAgentLogic:
    def __init__(self, master):
        self.context = BASE_CONTEXT + "\n\n"
        self.ai_name = "Chad"
        self.user_name = "Barry"
        self.master = master
        self.contect_log_file_path = f"logs/context{utils.time_str()}.txt"

    def receive_message(self, text_message):
        self.context += f"\n\n### {self.user_name}:\n" + text_message
        self.context += f"\n\n### {self.ai_name}:\n"

        response = asyncio.run(self.get_llm_output_stream(self.context,
                                                          stopping_strings=["\n"],
                                                          top_p=0.9, top_k=50,
                                                          max_new_tokens=200))
        self.context += response

        with open(self.contect_log_file_path, 'w', encoding='utf-8') as file:
            # Write the string to the file
            file.write(self.context)

        return response

    async def get_llm_output_stream(self, input_prompt, stream_type="speech", **kwargs):
        self.master.response_stream_queue.put_nowait(f"<start of {stream_type}>")

        response_tokens = []
        async for response in textgen_api.run(input_prompt, **kwargs):
            response_tokens.append(response)
            self.master.response_stream_queue.put_nowait(response)

        print(">>>", ''.join(response_tokens))
        print("------<end of stream>------")
        self.master.response_stream_queue.put_nowait("<end of stream>")
        return ''.join(response_tokens)

    @staticmethod
    def clean_up_text_for_tts(text):
        pattern = r'\*.*?\*'

        # Use the re.sub() function to remove the matched text
        output_string = re.sub(pattern, '', text)

        pattern = r'[^A-Za-z.,!? ]'  # Matches anything that is not A-Z, a-z, ., , !, ? or space

        # Use re.sub() to replace unwanted characters with an empty string
        output_string = re.sub(pattern, '', output_string)

        return output_string



