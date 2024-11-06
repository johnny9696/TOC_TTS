import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dialogue_dataset(Dataset):
    def __init__(self, text_path, model_config, preprocess_config):
        self.bert_in = model_config["dialogue_predictor"]["bert_in"]
        self.sbert_in = model_config["dialogue_predictor"]["sbert_in"]
        self.history_length = model_config["dialogue_predictor"]["history_length"]
        self.n_mels = model_config["dialogue_predictor"]["n_mels"]
        self.n_speakers = model_config["n_speaker"]
        self.processed_path = preprocess_config["path"]["preprocessed_path"]
        self.data_path = self.load(os.path.join(preprocess_config["path"]["preprocessed_path"], text_path))
        path = os.path.join(preprocess_config["path"]["preprocessed_path"],"fullbase.txt")
        with open(path, 'r', encoding="UTF-8") as f:
            self.full_name = f.read().split("\n")
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

    def load(self, path):
        with open(path, 'r', encoding='UTF-8') as f:
            data = f.read().split("\n")
        data.pop()
        return data

    def get_path_and_load(self, basename, time):
        if time == "cur":
            bert_path = os.path.join(self.processed_path, "BERT/bert-{}.npy".format(basename))
            sbert_path = os.path.join(self.processed_path, "SBERT/sbert-{}.npy".format(basename))
            p_duration_path = os.path.join(self.processed_path,"duration_phone/duration-phone-{}.npy".format(basename))
            t_duration_path = os.path.join(self.processed_path,"duration_token/duration-token-{}.npy".format(basename))
            pitch_path = os.path.join(self.processed_path,"pitch/pitch-{}.npy".format(basename))
            energy_path = os.path.join(self.processed_path,"energy/energy-{}.npy".format(basename))
            mel_path = os.path.join(self.processed_path,"mel/mel-{}.npy".format(basename))
            #cut_w2vser_path = os.path.join(self.processed_path,"CUTW2VSER/cutw2vser-{}.npy".format(basename))
            cut_token_w2vser_path = os.path.join(self.processed_path,"CUTTOKENW2VSER/cuttokenw2vser-{}.npy".format(basename))
            #w2vser_path = os.path.join(self.processed_path,'W2VSER/w2vser-{}.npy'.format(basename))
            bert = np.load(bert_path)
            sbert = np.load(sbert_path)
            p_duration = np.load(p_duration_path)
            t_duration = np.load(t_duration_path)
            pitch = np.load(pitch_path)
            energy = np.load(energy_path)
            mel = np.load(mel_path)
            #cut_w2vser = np.load(cut_w2vser_path)
            cut_w2vser = np.load(cut_token_w2vser_path)
            #w2vser = np.load(w2vser_path)
            return (bert, sbert, p_duration, t_duration, pitch, energy, mel, cut_w2vser)

        elif time == "history":
            bert_path = os.path.join(self.processed_path, "BERT/bert-{}.npy".format(basename))
            sbert_path = os.path.join(self.processed_path, "SBERT/sbert-{}.npy".format(basename))
            bert = np.load(bert_path)
            sbert = np.load(sbert_path)
            return (bert, sbert)
    def cutmel(self, mel, duration):
        max_dur = max(duration)
        cutmel = np.zeros((len(duration), max_dur, self.n_mels))
        sf = 0
        for indx, length in enumerate(duration):
            cutmel[indx, :length, :] = mel[sf:sf+length, :]
            sf = sf + length
        return cutmel, max_dur

    def get_cur(self, basename, phone, token_dur):
        (bert, sbert, p_duration, t_duration, pitch, energy, mel, cut_w2vser) = self.get_path_and_load(basename, time = "cur")
        bert_length = np.shape(bert)[0]
        phone = np.array(text_to_sequence(phone, self.cleaners))
        text_length = np.shape(phone)[0]
        mel_length = np.shape(mel)[0]
        cutmel, cutmel_dur = self.cutmel(mel, t_duration)
        token_dur = list(map(int, token_dur.split(" ")))
        return (phone, text_length, bert, bert_length, sbert, p_duration, t_duration, pitch, energy, mel, mel_length, token_dur, cutmel, cutmel_dur, cut_w2vser)

    def get_hist(self, basename):
        history_basename = list()
        h_speaker = list()
        #turn, speaker, dialoguenum
        c_turn, c_speaker, c_dialogue_num = basename.split("_")
        if int(c_turn) < self.history_length:
            s_turn = 0
            end_turn = int(c_turn)
            for i in range(self.history_length - end_turn):
                history_basename.append(None)
                h_speaker.append(self.n_speakers)
        else:
            s_turn = int(c_turn) - self.history_length
            end_turn = int(c_turn)
        for turn in range(s_turn, end_turn):
            for speaker in range(self.n_speakers):
                h_base = "_".join([str(turn), str(speaker), c_dialogue_num])
                if h_base in self.full_name:
                    history_basename.append(h_base)
                    h_speaker.append(speaker)

        bert_list = list()
        sbert_list = list()
        bert_length = list()
        for name in history_basename:
            if name is not None:
                (bert, sbert) = self.get_path_and_load(name ,"history")
                bert_list.append(bert)
                sbert_list.append(sbert)
                bert_length.append(np.shape(bert)[0])
            else:
                bert_list.append(np.zeros((1, self.bert_in)))
                sbert_list.append(np.zeros(self.sbert_in))
                bert_length.append(0)

        pad_bert = np.zeros((self.history_length, max(bert_length), self.bert_in))
        pad_sbert = np.zeros((self.history_length, self.sbert_in))

        for i in range(self.history_length):
            if history_basename[i] is not None:
                pad_bert[i, :bert_length[i], :] = bert_list[i]
                pad_sbert[i, :] = sbert_list[i]
        return (pad_bert, max(bert_length), pad_sbert, h_speaker)

    def get_all(self, data):
        basename, speaker, phone, raw_text, token_dur = data.split("|")
        (text, text_length, bert, bert_length, sbert, p_duration, t_duration, pitch, energy, mel, mel_length, token_dur, cutmel, cutmel_dur, cut_w2vser) = self.get_cur(basename, phone, token_dur)
        (h_bert, h_bert_length, h_sbert, h_speaker) = self.get_hist(basename)
        return (text, text_length, bert, bert_length, sbert, h_bert, h_bert_length, h_sbert, speaker, h_speaker, mel, mel_length, p_duration, t_duration, token_dur, pitch, energy, cutmel, cutmel_dur, cut_w2vser)

    def __len__(self):
        return len(self.data_path)
    def __getitem__(self, indx):
        return self.get_all(self.data_path[indx])
    def collate_fn(self, batch):
        """(text, text_length, bert, bert_length, sbert,
        h_bert, h_bert_length, h_sbert, speaker, h_speaker,
        mel, mel_length, p_duration, t_duration, token_dur,
        pitch, energy, cutmel, cutmel_dur, cut_w2vser)"""
        batch_size = len(batch)
        # get length
        text_length = [batch[i][1] for i in range(batch_size)]
        bert_length = [batch[i][3] for i in range(batch_size)]
        h_bert_length = [batch[i][6] for i in range(batch_size)]
        mel_length = [batch[i][11] for i in range(batch_size)]
        cutmel_length = [batch[i][18] for i in range(batch_size)]
        max_text_length = max(text_length)
        max_bert_length = max(max(bert_length), max(h_bert_length))
        max_mel_length = max(mel_length)
        max_cutmel_length = max(cutmel_length)

        output = list()
        for i in range(batch_size):
            t_length = torch.tensor(text_length[i], dtype=torch.int).unsqueeze(0)
            b_length = torch.tensor(bert_length[i], dtype = torch.int).unsqueeze(0)
            speaker = torch.tensor(int(batch[i][8]), dtype = torch.long).unsqueeze(0)
            h_speaker = torch.tensor(batch[i][9], dtype = torch.long).unsqueeze(0)
            m_length = torch.tensor(mel_length[i], dtype = torch.int).unsqueeze(0)
            text = torch.zeros((1, max_text_length), dtype= torch.long)
            bert = torch.zeros((1, max_bert_length, self.bert_in), dtype = torch.float)
            sbert = torch.zeros((1, self.sbert_in), dtype = torch.float)
            h_bert = torch.zeros((1, self.history_length, max_bert_length, self.bert_in))
            h_sbert = torch.zeros((1, self.history_length, self.sbert_in), dtype = torch.float)
            mel = torch.zeros((1, max_mel_length, self.n_mels), dtype = torch.float)
            cutmel = torch.zeros((1, max_bert_length, max_cutmel_length, self.n_mels), dtype = torch.float)
            p_duration = torch.zeros((1, max_text_length), dtype = torch.float)
            t_duration = torch.zeros((1, max_text_length), dtype = torch.float)
            token_dur = torch.zeros((1, max_bert_length), dtype = torch.float)
            pitch = torch.zeros((1, max_text_length), dtype = torch.float)
            energy = torch.zeros((1, max_text_length), dtype = torch.float)
            cut_w2v = torch.zeros((1, max_bert_length, 768), dtype = torch.float)
            #w2vser = torch.zeros((1,768), dtype = torch.float)

            text[0, :text_length[i]] = torch.from_numpy(batch[i][0])
            bert[0, :bert_length[i], :] = torch.from_numpy(batch[i][2])
            sbert[0, :] = torch.from_numpy(batch[i][4])
            h_bert[0, :, :h_bert_length[i], :] = torch.tensor(batch[i][5], dtype = torch.float)
            h_sbert[0, :, :] = torch.tensor(batch[i][7], dtype = torch.float)
            mel[0, :mel_length[i], :] = torch.tensor(batch[i][10], dtype = torch.float)
            cutmel[0, :bert_length[i], :cutmel_length[i], :] = torch.tensor(batch[i][17])
            p_duration[0, :text_length[i]] = torch.tensor(batch[i][12], dtype = torch.float)
            t_duration[0, :bert_length[i]] = torch.tensor(batch[i][13], dtype = torch.float)
            token_dur[0, :bert_length[i]] = torch.tensor(batch[i][14], dtype = torch.float)
            pitch[0, :text_length[i]] = torch.tensor(batch[i][15], dtype = torch.float)
            energy[0, :text_length[i]] = torch.tensor(batch[i][16], dtype = torch.float)
            cut_w2v[0,:bert_length[i],:] = torch.tensor(batch[i][19], dtype = torch.float)
            #w2vser[0, : ]= torch.tensor(batch[i][19], dtype = torch.float)
            output.append((text, t_length, torch.tensor(max_text_length), bert, b_length,
                           torch.tensor(max_bert_length), sbert, h_bert, h_sbert, speaker,
                           h_speaker, mel, m_length, torch.tensor(max_mel_length), p_duration,
                           t_duration, token_dur, pitch, energy, cutmel,
                           cut_w2v))
        return output
class Dialogue_dataset_eval(Dataset):
    def __init__(self, text_path, model_config, preprocess_config):
        self.bert_in = model_config["dialogue_predictor"]["bert_in"]
        self.sbert_in = model_config["dialogue_predictor"]["sbert_in"]
        self.history_length = model_config["dialogue_predictor"]["history_length"]
        self.n_mels = model_config["dialogue_predictor"]["n_mels"]
        self.n_speakers = model_config["n_speaker"]
        self.processed_path = preprocess_config["path"]["preprocessed_path"]
        self.data_path = self.load(os.path.join(preprocess_config["path"]["preprocessed_path"], text_path))
        path = os.path.join(preprocess_config["path"]["preprocessed_path"],"fullbase.txt")
        with open(path, 'r', encoding="UTF-8") as f:
            self.full_name = f.read().split("\n")
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

    def load(self, path):
        with open(path, 'r', encoding='UTF-8') as f:
            data = f.read().split("\n")
        data.pop()
        return data

    def get_path_and_load(self, basename, time):
        if time == "cur":
            bert_path = os.path.join(self.processed_path, "BERT/bert-{}.npy".format(basename))
            sbert_path = os.path.join(self.processed_path, "SBERT/sbert-{}.npy".format(basename))
            bert = np.load(bert_path)
            sbert = np.load(sbert_path)
            return (bert, sbert)

        elif time == "history":
            bert_path = os.path.join(self.processed_path, "BERT/bert-{}.npy".format(basename))
            sbert_path = os.path.join(self.processed_path, "SBERT/sbert-{}.npy".format(basename))
            bert = np.load(bert_path)
            sbert = np.load(sbert_path)
            return (bert, sbert)

    def get_cur(self, basename, phone, token_dur):
        (bert, sbert) = self.get_path_and_load(basename, time = "cur")
        bert_length = np.shape(bert)[0]
        phone = np.array(text_to_sequence(phone, self.cleaners))
        text_length = np.shape(phone)[0]
        return (phone, text_length, bert, bert_length, sbert)

    def get_hist(self, basename):
        history_basename = list()
        h_speaker = list()
        #turn, speaker, dialoguenum
        c_turn, c_speaker, c_dialogue_num = basename.split("_")
        if int(c_turn) < self.history_length:
            s_turn = 0
            end_turn = int(c_turn)
            for i in range(self.history_length - end_turn):
                history_basename.append(None)
                h_speaker.append(self.n_speakers)
        else:
            s_turn = int(c_turn) - self.history_length
            end_turn = int(c_turn)
        for turn in range(s_turn, end_turn):
            for speaker in range(self.n_speakers):
                h_base = "_".join([str(turn), str(speaker), c_dialogue_num])
                if h_base in self.full_name:
                    history_basename.append(h_base)
                    h_speaker.append(speaker)

        bert_list = list()
        sbert_list = list()
        bert_length = list()
        for name in history_basename:
            if name is not None:
                (bert, sbert) = self.get_path_and_load(name ,"history")
                bert_list.append(bert)
                sbert_list.append(sbert)
                bert_length.append(np.shape(bert)[0])
            else:
                bert_list.append(np.zeros((1, self.bert_in)))
                sbert_list.append(np.zeros(self.sbert_in))
                bert_length.append(0)

        pad_bert = np.zeros((self.history_length, max(bert_length), self.bert_in))
        pad_sbert = np.zeros((self.history_length, self.sbert_in))

        for i in range(self.history_length):
            if history_basename[i] is not None:
                pad_bert[i, :bert_length[i], :] = bert_list[i]
                pad_sbert[i, :] = sbert_list[i]
        return (pad_bert, max(bert_length), pad_sbert, h_speaker)

    def get_all(self, data):
        basename, speaker, phone, raw_text, token_dur = data.split("|")
        (text, text_length, bert, bert_length, sbert) = self.get_cur(basename, phone, token_dur)
        (h_bert, h_bert_length, h_sbert, h_speaker) = self.get_hist(basename)
        return (text, text_length, bert, bert_length, sbert, h_bert, h_bert_length, h_sbert, speaker, h_speaker, basename )

    def __len__(self):
        return len(self.data_path)
    def __getitem__(self, indx):
        return self.get_all(self.data_path[indx])
    def collate_fn(self, batch):
        """(text, text_length, bert, bert_length, sbert, h_bert, h_bert_length, h_sbert, speaker, h_speaker, basename )"""
        batch_size = len(batch)
        # get length
        text_length = [batch[i][1] for i in range(batch_size)]
        bert_length = [batch[i][3] for i in range(batch_size)]
        h_bert_length = [batch[i][6] for i in range(batch_size)]
        max_text_length = max(text_length)
        max_bert_length = max(max(bert_length), max(h_bert_length))
        basename = [batch[i][10] for i in range(batch_size)]

        output = list()
        for i in range(batch_size):
            t_length = torch.tensor(text_length[i], dtype=torch.int).unsqueeze(0)
            b_length = torch.tensor(bert_length[i], dtype = torch.int).unsqueeze(0)
            speaker = torch.tensor(int(batch[i][8]), dtype = torch.long).unsqueeze(0)
            h_speaker = torch.tensor(batch[i][9], dtype = torch.long).unsqueeze(0)
            text = torch.zeros((1, max_text_length), dtype= torch.long)
            bert = torch.zeros((1, max_bert_length, self.bert_in), dtype = torch.float)
            sbert = torch.zeros((1, self.sbert_in), dtype = torch.float)
            h_bert = torch.zeros((1, self.history_length, max_bert_length, self.bert_in))
            h_sbert = torch.zeros((1, self.history_length, self.sbert_in), dtype = torch.float)

            text[0, :text_length[i]] = torch.from_numpy(batch[i][0])
            bert[0, :bert_length[i], :] = torch.from_numpy(batch[i][2])
            sbert[0, :] = torch.from_numpy(batch[i][4])
            h_bert[0, :, :h_bert_length[i], :] = torch.tensor(batch[i][5], dtype = torch.float)
            h_sbert[0, :, :] = torch.tensor(batch[i][7], dtype = torch.float)

            output.append((text, t_length, torch.tensor(max_text_length), bert, b_length,
                           torch.tensor(max_bert_length), sbert, h_bert, h_sbert, speaker,
                           h_speaker, None, None, None, None,
                           None, None, None, None, None,
                           None))
        return output, basename

class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        data_size = len(data)
        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
