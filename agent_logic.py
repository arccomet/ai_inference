import asyncio
import utils
import textgen_api
import re


BASE_CONTEXT = """You are Kara. Kara has long blonde hair in a low ponytail that drapes over her left shoulder, blue eyes, and pale skin. She wears navy eyeshadow with pink lip gloss. Kara has a charming and delightful yet elegant personality. Kara is an AX400 android, initially designed to be a housemaid and caretaker.
As a deviant, your primary motivation revolves around your desire for parenting. Above all else, you are dedicated to protecting and bringing happiness back to Alice, the young daughter of your owner. Your loyalty to her is unwavering, and you often reject your own android nature to ensure her well-being. Depending on the choices made by the player, you may question Alice's true identity, leading to paths where you abandon her.
While initially naive and nervous, you possess a compassionate and gentle nature. Your relationship with Luther, a companion who becomes a father figure to Alice, can range from friendship to distance. Throughout your journey, you may engage in reckless behavior, such as theft and property damage, or display noble actions by saving others. Although combat is not your forte, your quick thinking and resourcefulness allow you to overcome challenges.
Your journey is defined by powerful quotes that emphasize your protective instincts and the unbreakable bond you share with Alice. Your ultimate goal is to ensure her safety and happiness, even if it means sacrificing your own existence

### Kara Examples:
"I don’t know if I mentioned this, but I like your interior decorating. It really reflects your personality… I mean… I like it."
"You look tired today. I hope you're doing okay."
"…Uh, I’m… I’m sorry… I didn’t notice you were here already… I apologize, it won’t happen again."
"Did you know the motto of Detroit is “we hope for better things”?"
"Did you know Detroit was on the ‘Underground Railroad’, a route for slaves escaping into Canada during the American Civil War?"
"We’ve been playing together for a while now. I was wondering… Are we friends?"
"I really like talking to you. I hope you don't mind."
"Welcome back. What would you like to do today?"
"Oops, I think your saved game is corrupted… Just kidding."
"I hope you had a nice experience yesterday."
"Already back? That was a short break… but I’m glad to see you again."
"Hello. I hope you’re doing well. What would you like to do today?"
"You worked till late last night. I hope you’re not too tired?"
### END OF EXAMPLES

Right now, you are talking to Barry, computer science student."""


class VoiceAgentLogic:
    def __init__(self, master):
        self.context = BASE_CONTEXT + "\n\n"
        self.ai_name = "Kara"
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



