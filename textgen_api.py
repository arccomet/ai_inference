import asyncio
import json
import sys
import websockets

# For local streaming, the websockets are hosted without ssl - ws://
HOST = 'localhost:5005'
URI = f'ws://{HOST}/api/v1/stream'


async def run(context, **kwargs):
    request = {
        'prompt': context,
        'max_new_tokens': kwargs.get('max_new_tokens', 250),
        'preset': kwargs.get('preset', 'None'),
        'do_sample': kwargs.get('do_sample', True),
        'temperature': kwargs.get('temperature', 0.7),
        'top_p': kwargs.get('top_p', 0.1),
        'typical_p': kwargs.get('typical_p', 1),
        'epsilon_cutoff': kwargs.get('epsilon_cutoff', 0),
        'eta_cutoff': kwargs.get('eta_cutoff', 0),
        'tfs': kwargs.get('tfs', 1),
        'top_a': kwargs.get('top_a', 0),
        'repetition_penalty': kwargs.get('repetition_penalty', 1.18),
        'repetition_penalty_range': kwargs.get('repetition_penalty_range', 0),
        'top_k': kwargs.get('top_k', 40),
        'min_length': kwargs.get('min_length', 0),
        'no_repeat_ngram_size': kwargs.get('no_repeat_ngram_size', 0),
        'num_beams': kwargs.get('num_beams', 1),
        'penalty_alpha': kwargs.get('penalty_alpha', 0),
        'length_penalty': kwargs.get('length_penalty', 1),
        'early_stopping': kwargs.get('early_stopping', False),
        'mirostat_mode': kwargs.get('mirostat_mode', 0),
        'mirostat_tau': kwargs.get('mirostat_tau', 5),
        'mirostat_eta': kwargs.get('mirostat_eta', 0.1),
        'seed': kwargs.get('seed', -1),
        'add_bos_token': kwargs.get('add_bos_token', True),
        'truncation_length': kwargs.get('truncation_length', 2048),
        'ban_eos_token': kwargs.get('ban_eos_token', False),
        'skip_special_tokens': kwargs.get('skip_special_tokens', True),
        'stopping_strings': kwargs.get('stopping_strings', [])
    }

    async with websockets.connect(URI, ping_interval=None) as websocket:
        await websocket.send(json.dumps(request))

        # Not returning context
        # yield context

        while True:
            incoming_data = await websocket.recv()
            incoming_data = json.loads(incoming_data)

            match incoming_data['event']:
                case 'text_stream':
                    yield incoming_data['text']
                case 'stream_end':
                    return


async def fake_run(input_prompt, **kwargs):
    # Your implementation of the run function
    # This function should yield the response tokens one by one

    # Example implementation:
    tokens = "Hello world. I'm a fake chatbot. I'm cool."
    for token in tokens:
        yield str(token)
        await asyncio.sleep(0.01)  # Simulating some delay


async def print_response_stream(input_prompt):
    response_tokens = []
    async for response in run(input_prompt):
        response_tokens.append(response)
        print(response, end='')
        sys.stdout.flush()  # If we don't flush, we won't see tokens in realtime.

    return ''.join(response_tokens)
