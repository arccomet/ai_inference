import utils
from flask import Flask, request, jsonify
import threading
from simple_turn_base_voice_ai_agent import SimpleTurnBaseVoiceAgent

app = Flask(__name__)


@app.route('/receive_audio', methods=['POST'])
def receive_audio():
    print(request.headers.get('Content-Type'))
    job_obj = request.json
    voice_agent.put_new_job(job_obj)

    return jsonify({"message": "Processing audio."}), 200


@app.route('/try_get_response', methods=['POST'])
def try_get_response():
    response = {"message": "Result not available"}

    text_msg = request.json["text"]
    if text_msg == "GET":
        result = voice_agent.try_get_result()
        if result:
            print(">>>>>>>>>>>>> valid result here")
            response = result
            response["message"] = "Success"

    print(response["message"])
    return jsonify(response), 200


if __name__ == '__main__':
    utils.init_directories()

    voice_agent = SimpleTurnBaseVoiceAgent(low_vram=False)

    voice_agent_thread = threading.Thread(target=voice_agent.run_agent, args=())
    voice_agent_thread.start()

    app.run(host='0.0.0.0', port=7865)
