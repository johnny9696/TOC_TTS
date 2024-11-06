import torch
from torch import nn
from transformer.SubLayers import MultiHeadAttention
from transformer.Models import get_sinusoid_encoding_table
from .modules import VariancePredictor, LengthRegulator

class Layernorm(nn.Module):
    def __init__(self, hidden_dim, eps = 1e-5):
        super(Layernorm, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
    def forward(self, x):
        """
        [b, i, hidden_dim]
        """
        x_mean = torch.mean(x, -1, keepdim = True)
        x_variance = torch.mean((x-x_mean)**2, -1, keepdim = True)
        x = (x - x_mean) * torch.rsqrt(x_variance + self.eps)
        n_dims = len(x.shape)
        shape = [1, 1] + [-1] * (n_dims - 2)
        x = x * self.gamma.view(shape) + self.bias.view(shape)
        return x

class mel_prosody(nn.Module):
    def __init__(self, model_config):
        super(mel_prosody, self).__init__()
        n_mels = model_config["dialogue_predictor"]["n_mels"]
        kernel = model_config["dialogue_predictor"]["2d_kernel"]
        filter_channel = model_config["dialogue_predictor"]["filter_channel"]
        hidden_channel = model_config["transformer"]["encoder_hidden"]
        n_layers = model_config["dialogue_predictor"]["melgru_layers"]
        dropout = model_config["dialogue_predictor"]["dropout"]

        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(n_mels, filter_channel, kernel, padding=kernel//2),
            nn.BatchNorm2d(filter_channel),
            nn.ReLU()
        )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(filter_channel, hidden_channel, kernel, padding= kernel//2),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU()
        )
        self.mean_gru = nn.GRU(
            input_size=hidden_channel,
            hidden_size=hidden_channel,
            batch_first=True,
            bidirectional=False,
            num_layers= 1,
        )
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(hidden_channel,hidden_channel, kernel, padding= kernel//2),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU()
        )
        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(hidden_channel,hidden_channel, kernel, padding= kernel//2),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU()
        )

        self.out_linear = nn.Conv1d(hidden_channel, hidden_channel, 1)
        self.speaker_linear= nn.Linear(hidden_channel,hidden_channel)
    def forward(self, cutmel, speaker, x_mask):
        """
        cutmel [b, text_length, frame, n_mels]
        x_mask [b, text_length]
        """
        b, t, f, c = cutmel.size()
        cutmel = cutmel.permute(0, 3, 1, 2) #[b, n_mels, text_length, frame]
        cutmel = self.conv2d_1(cutmel)
        cutmel = self.conv2d_2(cutmel)
        #cutmel = torch.mean(cutmel, dim=-1)
        cutmel = cutmel.permute(0, 2, 3, 1).view(b*t, f, -1)
        cutmel, _ = self.mean_gru(cutmel)
        cutmel = cutmel[:,-1, :].view(b, t, -1).transpose(-1,-2)
        #print(cutmel.size())
        speaker = self.speaker_linear(speaker)
        cutmel = cutmel + speaker.unsqueeze(-1).expand(-1, -1, cutmel.size(-1)) #+ output.transpose(-1,-2)
        cutmel_ = self.conv1d_1(cutmel) #v17
        cutmel_ = self.conv1d_2(cutmel_ + cutmel) #v18
        hidden = self.out_linear(cutmel_).transpose(-1,-2).masked_fill(x_mask.unsqueeze(-1), 0)

        return hidden


class conv_mixer(nn.Module):
    def __init__(self, model_config):
        super(conv_mixer, self).__init__()
        kernel_size = model_config["dialogue_predictor"]["1d_kernel"]
        in_channel = model_config["transformer"]["encoder_hidden"]
        filter_channel = model_config["dialogue_predictor"]["filter_channel"]
        hidden_channel =model_config["transformer"]["encoder_hidden"]
        dropout = model_config["dialogue_predictor"]["dropout"]
        self.conv1d_0 = nn.Sequential(
            nn.Conv1d(in_channel, filter_channel, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(filter_channel, hidden_channel, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(in_channel, filter_channel, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(filter_channel, hidden_channel, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.out_linear = nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1, stride=1)
    def forward(self, fusion, x_mask):
        """
        fusion [b, length, hidden]
        """

        fusion = fusion.transpose(-1, -2)
        fusion_ = self.conv1d_0(fusion)
        fusion_ = self.conv1d_1(fusion+fusion_) #V18
        fusion = self.out_linear((fusion + fusion_)).transpose(-1,-2).masked_fill(x_mask.unsqueeze(-1), 0)
        return fusion

class emotion_predictor(nn.Module):
    def __init__(self, model_config):
        super(emotion_predictor, self).__init__()
        wav_dim = model_config["dialogue_predictor"]["wav_in"]
        hidden_dim = model_config["transformer"]["encoder_hidden"]
        dropout = model_config["dialogue_predictor"]["dropout"]

        self.slf_attn = MultiHeadAttention(1, hidden_dim, hidden_dim, hidden_dim, dropout)
        self.layernorm1 = Layernorm(hidden_dim)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), #v18
            nn.ReLU(), #v18
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU()
        )
        self.layernorm2 = Layernorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, atten_input, sbert, speaker_emb, history_emotion, bert_mask, gen = False):
        if history_emotion is not None:
            fusion = atten_input + sbert.unsqueeze(1).expand(-1, atten_input.size(1), -1) + history_emotion + speaker_emb.unsqueeze(1).expand(-1, atten_input.size(1), -1)
        else:
            fusion = atten_input + sbert.unsqueeze(1).expand(-1, atten_input.size(1), -1) + speaker_emb.unsqueeze(1).expand(-1, atten_input.size(1), -1)
        attn_mask = bert_mask.unsqueeze(1)*bert_mask.unsqueeze(-1)
        fusion_, _ = self.slf_attn(fusion, fusion, fusion, attn_mask)
        fusion = self.layernorm1(fusion+fusion_)
        fusion_ = self.linear(fusion)
        fusion = self.layernorm2(fusion + fusion_)
        history_emotion = self.out_linear(fusion)


        return history_emotion
class context_predictor(nn.Module):
    def __init__(self, model_config):
        super(context_predictor, self).__init__()
        hidden_dim = model_config["transformer"]["encoder_hidden"]
        dropout = model_config["dialogue_predictor"]["dropout"]
        self.slf_attention = MultiHeadAttention(1, hidden_dim, hidden_dim, hidden_dim, dropout = dropout)
        self.layernorm1 = Layernorm(hidden_dim)
        self.linear= nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU()
        )
        self.layernorm2 = Layernorm(hidden_dim)

    def forward(self, bert, history_context, emotion_context, bert_mask):
        if emotion_context is not None:
            bert = bert + emotion_context.expand(-1, bert.size(1), -1)
        if history_context is not None:
            history_context = bert + history_context
        else:
            history_context = bert
        attn_mask = bert_mask.unsqueeze(1) * bert_mask.unsqueeze(-1)
        history_context_, _ = self.slf_attention(history_context,history_context, history_context, attn_mask)
        history_context = self.layernorm1(history_context + history_context_)
        history_context_ = self.linear(history_context)
        history_context = self.layernorm2(history_context + history_context_)
        return history_context

class history_dialogue_predictor(nn.Module):
    def __init__(self, model_config):
        super(history_dialogue_predictor, self).__init__()

        text_in=model_config["dialogue_predictor"]["bert_in"]
        sbert_in=model_config["dialogue_predictor"]["sbert_in"]
        hidden_dim = model_config["transformer"]["encoder_hidden"]
        self.hist_length = model_config["dialogue_predictor"]["history_length"]
        self.n_speaker = model_config["n_speaker"]
        self.bert_linear = nn.Linear(text_in, hidden_dim)
        self.sbert_linear = nn.Linear(sbert_in, hidden_dim)
        self.speaker_linear = nn.Linear(hidden_dim, hidden_dim)
        self.context_predictor = nn.ModuleList()
        self.emotion_predictor = nn.ModuleList()
        for i in range(self.hist_length+1):
            self.emotion_predictor.append(emotion_predictor(model_config))
            self.context_predictor.append(context_predictor(model_config))
        self.token_duration = VariancePredictor(model_config)
        self.mixer = conv_mixer(model_config)

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(1000, hidden_dim).unsqueeze(0),
            requires_grad=False,
        )

        self.cross = MultiHeadAttention(1, hidden_dim, hidden_dim, hidden_dim)
    def forward(self, h_bert, h_sbert, bert, sbert, bert_mask, h_speaker,
                h_speaker_emb, n_speaker, n_speaker_emb, token_target = None, gen = False):
        #shape matching
        h_bert = self.bert_linear(h_bert)
        h_sbert = self.sbert_linear(h_sbert)
        h_speaker_emb = self.speaker_linear(h_speaker_emb)

        bert = self.bert_linear(bert)
        sbert =self.sbert_linear(sbert)
        n_speaker_emb= self.speaker_linear(n_speaker_emb)

        emotion_emb = None
        context_emb0 = None
        context_emb1 = None
        #get context_emb
        for i in range(self.hist_length):
            c_speaker = h_speaker[0][i]
            if c_speaker == 0 :
                context_emb0 = self.context_predictor[i](h_bert[:, i], context_emb0, emotion_emb, bert_mask)
                emotion_emb= self.emotion_predictor[i](context_emb0, h_sbert[:, i], h_speaker_emb[:, i], emotion_emb,
                                                                bert_mask, gen=gen)
            elif c_speaker == 1:
                context_emb1 = self.context_predictor[i](h_bert[:, i], context_emb1, emotion_emb, bert_mask)
                emotion_emb= self.emotion_predictor[i](context_emb1, h_sbert[:, i], h_speaker_emb[:, i], emotion_emb,
                                                           bert_mask, gen=gen)
        if n_speaker[0] == 0:
            context_emb = self.context_predictor[self.hist_length](bert, context_emb0, emotion_emb, bert_mask)
        elif n_speaker[0] == 1:
            context_emb = self.context_predictor[self.hist_length](bert, context_emb1, emotion_emb, bert_mask)
        emotion_emb= self.emotion_predictor[self.hist_length](context_emb, sbert, n_speaker_emb, emotion_emb, bert_mask, gen = gen)
        context_emb = context_emb + emotion_emb + bert #v18

        #length predictor
        log_token_duration = self.token_duration(bert, bert_mask)
        if token_target is not None:
            t_round = torch.clamp(
                (torch.round(torch.exp(log_token_duration) - 1)),
                min=0,
            )
        else:
            t_round = token_target

        fusion = self.mixer(context_emb, bert_mask)

        return fusion, log_token_duration, t_round