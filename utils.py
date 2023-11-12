import os
import datetime


def init_directories():
    directories = ["wav", "wav/input_wavs", "wav/outputs", "logs"]
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)


def time_str():
    return datetime.datetime.now().strftime('%d%m%Y_%H_%M_%S')
