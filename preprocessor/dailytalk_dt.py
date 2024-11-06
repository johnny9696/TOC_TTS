import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text
from transformers import AutoTokenizer

import os
import json


def make_csv(path):
    with open(os.path.join(path, "metadata.json")) as json_file:
        json_data = json.load(json_file)
    dialogue = list(json_data.keys())
    data_path = os.path.join(path, "data")
    with open(os.path.join(path, "metadata.txt"), "w", encoding="UTF-8") as full_file:
        for d_num in dialogue:
            turns = list(json_data[d_num].keys())
            for turn_num in turns:
                turn = json_data[d_num][turn_num]["utterance_idx"]
                emotion = json_data[d_num][turn_num]["emotion"]
                speaker = json_data[d_num][turn_num]["speaker"]
                text = json_data[d_num][turn_num]["text"]
                wav_file = "_".join([str(turn_num), str(speaker), "d" + d_num]) + ".wav"
                full_wav_path = os.path.join(data_path, d_num, wav_file)
                # wav|speaker|emotion|text
                full_txt = "|".join([full_wav_path, str(speaker), emotion, text])
                full_file.write(full_txt + "\n")

def prepare_align(config):
    print("DailyTalk_dt")
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    make_csv(in_dir)
    print("Made finished")
    text_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    with open(os.path.join(in_dir, "metadata.txt"), encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            wav_path = parts[0]
            speaker = parts[1]
            emotion = parts[2]
            text = parts[3]
            text = _clean_text(text, cleaners)
            token = text_tokenizer.tokenize(text)
            text= " ".join(token)
            base_name = wav_path.split("/")[-1]
            base_name = base_name[:-4]
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sr = sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)