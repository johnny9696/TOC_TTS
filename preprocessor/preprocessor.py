import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
# BERT Model
from transformers import AutoTokenizer, BertModel
# W2V2
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# wav2vecSER Model
from speechbrain.inference.interfaces import foreign_class
# sentence transformer
from sentence_transformers import SentenceTransformer

import torch

from tqdm import tqdm

import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
                config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
                config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration_phone")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration_token")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "SBERT")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "BERT")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "W2VSER")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "CUTW2VSER")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "CUTTOKENW2VSER")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "W2V")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # load pretrained model
        sentence_embedder = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
        audio_embedder = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                                       pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
        text_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        model_name = "facebook/wav2vec2-base-960h"
        feature_extractor = Wav2Vec2Processor.from_pretrained(model_name)
        w2v_model = Wav2Vec2ForCTC.from_pretrained(model_name)
        processor_model = (sentence_embedder, audio_embedder, text_tokenizer, model, feature_extractor, w2v_model)

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename, processor_model)
                    if ret is None:
                        print(basename)
                        continue
                    else:
                        info, pitch, energy, n = ret
                    out.append(info)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        with open(os.path.join(self.out_dir, "full.txt"), "w", encoding="utf-8") as f:
            for m in out:
                f.write(m + "\n")
        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size:]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def match_t_p(self, pte, tte, bert_token, duration, duration_t):
        t_p_match_list = list()
        phone_list = list()
        duration_list = list()
        duration_token_list = list()
        time_set = list()
        time_set_token = list()
        start_indx = 0
        p_start_indx = 0
        for b_indx in range(len(bert_token)):
            token = bert_token[b_indx]
            token = token.replace("#", "")
            for d_t_indx in range(start_indx, len(tte)):
                t, ts, te = tte[d_t_indx]
                if token == t:
                    si = None
                    ei = None
                    for p_t_indx in range(p_start_indx, len(pte)):
                        p, pt, pe = pte[p_t_indx]
                        if ts == pt:
                            si = p_t_indx
                            time_set_token.append((ts, te))
                            duration_token_list.append(duration_t[d_t_indx])
                        if si is not None:
                            phone_list.append(p)
                            duration_list.append(duration[p_t_indx])
                            time_set.append((pt, pe))
                        if te == pe:
                            ei = p_t_indx
                            p_start_indx = p_t_indx + 1
                            break
                    t_p_match_list.append((t, str(ei - si + 1)))
                    start_indx = d_t_indx + 1
                    break

        return t_p_match_list, phone_list, duration_list, duration_token_list, time_set, time_set_token

    def process_utterance(self, speaker, basename, processor_model):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join("./raw_data/dailytalk", speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # split model
        (sentence_embedder, audio_embedder, text_tokenizer, model, w2v_feature, w2v_model) = processor_model

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end, pte = self.get_alignment(
            textgrid.get_tier_by_name("phones"), "phone"
        )
        token_t, duration_t, start_t, end_t, tte = self.get_alignment(
            textgrid.get_tier_by_name("words"), "word"
        )

        if start >= end:
            print(basename, start, end, phone, duration)
            return None


        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")
            raw_text = raw_text.replace(".", " ")
            raw_text = raw_text.replace(",", " ")
            raw_text = raw_text.replace("?", " ")
            raw_text = raw_text.replace("!", " ")
            raw_text = raw_text.replace("`", " ")
            raw_text = raw_text.replace("-", " ")
            raw_text = raw_text.replace("'", " ")
            raw_text = raw_text.replace('"', " ")
            raw_text = raw_text.replace(";", " ")
            raw_text = raw_text.replace(":", " ")
            raw_text = raw_text.replace("&", " ")
            raw_text = raw_text.replace("/", " ")
            raw_text = raw_text.replace("\\", " ")
            while "(" in raw_text:
                si = raw_text.index('(')
                et = raw_text.index(')')
                if si == 0:
                    raw_text = raw_text[et + 1:]
                else:
                    raw_text = raw_text[:si] + raw_text[et + 1:]

        # raw_text = " ".join(token_t)
        # text and audio pretrained model
        # BERT
        # btext = raw_text.replace(" ", "-")
        length = len(raw_text)
        # token = text_tokenizer(raw_text, return_tensors="pt")
        token = text_tokenizer(raw_text, return_tensors="pt")
        token_T = text_tokenizer.tokenize(raw_text)
        outputs = model(**token)
        text_embedding = outputs.last_hidden_state
        text_emb = text_embedding.squeeze(0).detach().numpy()
        text_emb = text_emb[1:-1, :]

        match_list, phone_list, duration, duration_token, time_set, time_set_token = self.match_t_p(pte, tte, token_T, duration, duration_t)
        start = time_set[0][0]
        end = time_set[-1][-1]
        # Read and trim wav files
        wav, sr = librosa.load(wav_path)
        wavcut = wav[
              int(self.sampling_rate * start): int(self.sampling_rate * end)
              ].astype(np.float32)

        length_list = []
        s_num = 0
        for indx, (t, time) in enumerate(match_list):
            if t != "[branketed]":
                length_list.append(time)

        target_length = " ".join(length_list)
        text = "{" + " ".join(phone_list) + "}"

        if len(length_list) != np.shape(text_emb)[0] or sum(list(map(int, length_list))) != len(phone_list):
            print('length is different {} {}'.format(len(length_list), np.shape(text_emb)[0]))
            print(raw_text)
            print(token_T, token_t, basename)
            print(sum(list(map(int, length_list))), len(phone_list))
            print(match_list)
            print(phone_list)
        """
        # SBERT
        s_emb = sentence_embedder.encode([text])
        s_emb = np.array(s_emb.squeeze(0))

        # WAV2VEC SER
        wav_e = torch.tensor(wavcut)
        wav_emb = audio_embedder.encode_batch(wav_e)
        wav_emb = np.array(wav_emb.squeeze(0))

        #cutted wav2vecser
        cut_wav_list = list()
        min_length = 0.03
        for s, e in time_set:
            if e-s <=min_length:
                m_time = e - s
                if s-(min_length - m_time)/2 <start and e +(min_length - m_time)/2<=end:
                    e = e + (min_length - m_time) - s + start
                    s=start
                elif s-(min_length - m_time)/2 >=start and e +(min_length - m_time)/2<=end:
                    s = s - (min_length - m_time)/2
                    e = e + (min_length - m_time)/2
                elif s-(min_length - m_time)/2 >=start and e +(min_length - m_time)/2>end:
                    s = s - (min_length-m_time) +(end - e)
                    e = end
                else:
                    s=start
                    e=end
                    print('wrong length cut'+basename)
            c_wav = wav[
              int(self.sampling_rate * s): int(self.sampling_rate * e)
              ].astype(np.float32)
            c_wav = torch.tensor(c_wav)
            try:
                wav_emb = audio_embedder.encode_batch(c_wav)
            except:
                print('wrong length cut'+basename, start, end, s, e)
                raise
            cut_wav_list.append(wav_emb)
        cut_wav_emb = torch.cat(cut_wav_list, dim = 0).detach().numpy()
        """
        """
        #cutted wav2vecser_token
        cut_wav_token_list = list()
        min_length = 0.03
        for s, e in time_set_token:
            if e-s <=min_length:
                m_time = e - s
                if s-(min_length - m_time)/2 <start and e +(min_length - m_time)/2<=end:
                    e = e + (min_length - m_time) - s + start
                    s=start
                elif s-(min_length - m_time)/2 >=start and e +(min_length - m_time)/2<=end:
                    s = s - (min_length - m_time)/2
                    e = e + (min_length - m_time)/2
                elif s-(min_length - m_time)/2 >=start and e +(min_length - m_time)/2>end:
                    s = s - (min_length-m_time) +(end - e)
                    e = end
                else:
                    s=start
                    e=end
                    print('wrong length cut'+basename)
            ct_wav = wav[
              int(self.sampling_rate * s): int(self.sampling_rate * e)
              ].astype(np.float32)
            ct_wav = torch.tensor(ct_wav)
            try:
                wav_emb = audio_embedder.encode_batch(ct_wav)
            except:
                print('wrong length cut'+basename, start, end, s, e)
                raise
            cut_wav_token_list.append(wav_emb)
        cut_wav_token_emb = torch.cat(cut_wav_token_list, dim = 0).detach().numpy()
        """
        """
        # wav2vec
        resample = librosa.resample(wavcut, orig_sr=sr, target_sr=16000)
        feature = w2v_feature(resample, sampling_rate=16000, return_tensors="pt")
        w2v_e = w2v_model(feature.input_values, output_hidden_states=True).hidden_states
        w2v_e = w2v_e[-1]
        w2v_emb = w2v_e.squeeze(0).detach().numpy()
        """
        # Compute fundamental frequency
        pitch, t = pw.dio(
            wavcut.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wavcut.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]

        if np.sum(pitch != 0) <= 1:
            print(basename)
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wavcut, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos: pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos: pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]
        """
        # Save files
        dur_filename = "duration-phone-{}.npy".format(basename)
        np.save(os.path.join(self.out_dir, "duration_phone", dur_filename), duration)

        dur_filename = "duration-token-{}.npy".format(basename)
        np.save(os.path.join(self.out_dir, "duration_token", dur_filename), duration_token)
        """
        pitch_filename = "pitch-{}.npy".format(basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "energy-{}.npy".format(basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "mel-{}.npy".format(basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )
        """
        bert_filename = "bert-{}.npy".format(basename)
        np.save(
            os.path.join(self.out_dir, "BERT", bert_filename),
            text_emb
        )
        
        sbert_filename = "sbert-{}.npy".format(basename)
        np.save(
            os.path.join(self.out_dir, "SBERT", sbert_filename),
            s_emb
        )
        
        w2vser_filename = "w2vser-{}.npy".format(basename)
        np.save(
            os.path.join(self.out_dir, "W2VSER", w2vser_filename),
            wav_emb
        )
        
        w2vser_filename = "cutw2vser-{}.npy".format(basename)
        np.save(
            os.path.join(self.out_dir, "CUTW2VSER", w2vser_filename),
            cut_wav_emb
        )
        """
        """
        w2vser_token_filename = "cuttokenw2vser-{}.npy".format(basename)
        np.save(
            os.path.join(self.out_dir, "CUTTOKENW2VSER", w2vser_token_filename),
            cut_wav_token_emb
        )
        """
        """
        w2v_filename = "w2v-{}.npy".format(basename)
        np.save(os.path.join(self.out_dir, "W2V", w2v_filename),
                w2v_emb
                )
        """
        return (
            "|".join([basename, speaker, text, raw_text, target_length]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier, level):
        phones = []
        durations = []
        start_time = 0
        time_phone_list = []
        end_time = 0
        end_idx = 0
        if level =="phone":
            sil_phones =["sil, sp, spn"]
        elif level =="word":
            sil_phones =["[branket]"]
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)

            time_phone_list.append((p, s, e))

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time, time_phone_list

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value