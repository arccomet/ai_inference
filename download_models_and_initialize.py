import os

# By using XTTS you agree to CPML license https://coqui.ai/cpml
os.environ["COQUI_TOS_AGREED"] = "1"

import utils

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir


utils.init_directories()

LOW_VRAM = False
SAMPLE_RATE = 24000

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))

# This will trigger downloading model
print("Downloading if not downloaded Coqui XTTS V2")
from TTS.utils.manage import ModelManager
ModelManager().download_model(model_name)
print("XTTS downloaded")

import whisperx
import torch
import numpy as np

m = whisperx.load_model("large-v2", "cuda", compute_type="float16", language="en")
