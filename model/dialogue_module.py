import torch
from torch import nn
from transformer.SubLayers import MultiHeadAttention
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

class LayerNormT(nn.Module):
  def __init__(self, channels, eps=1e-4):
      super().__init__()
      self.channels = channels
      self.eps = eps

      self.gamma = nn.Parameter(torch.ones(channels))
      self.beta = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    n_dims = len(x.shape)
    mean = torch.mean(x, 1, keepdim=True)
    variance = torch.mean((x -mean)**2, 1, keepdim=True)

    x = (x - mean) * torch.rsqrt(variance + self.eps)

    shape = [1, -1] + [1] * (n_dims - 2)
    x = x * self.gamma.view(*shape) + self.beta.view(*shape)
    return x

class mel_prosody(nn.Module):
    def __init__(self, model_config):
        super(mel_prosody, self).__init__()
        n_mels = model_config["dialogue_predictor"]["n_mels"]
        filter_channel = model_config["dialogue_predictor"]["filter_channel"]
        kernel2d = model_config["dialogue_predictor"]["2d_kernel"]
        kernel1d =model_config["dialogue_predictor"]["1d_kernel"]
        dropout = model_config["dialogue_predictor"]["dropout"]
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(n_mels, filter_channel//2, kernel1d, padding=kernel1d//2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(filter_channel//2, filter_channel, kernel1d, padding=kernel1d//2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.batchnorm1 = nn.BatchNorm2d(filter_channel//2)
        self.batchnorm2 = nn.BatchNorm2d(filter_channel)

        self.bigru = nn.GRU(
            input_size=filter_channel,
            hidden_size=filter_channel//2,
            bidirectional=True,
            batch_first=True,
            num_layers= 1,
        )

        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(filter_channel*2, filter_channel, kernel1d, padding = kernel1d//2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(filter_channel, filter_channel, kernel1d, padding=kernel1d // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.batchnorm3 = nn.BatchNorm1d(filter_channel)
        self.batchnorm4 = nn.BatchNorm1d(filter_channel)

    def forward(self, cutmel, speaker, x_mask):
        """
        cutmel [b, text_length, frame, n_mels]
        x_mask [b, length]
        """
        b, t, f, m = cutmel.size()
        cutmel = cutmel.permute(0, 3, 1, 2)
        cutmel = self.conv2d_1(cutmel)
        cutmel = self.batchnorm1(cutmel)
        cutmel = self.conv2d_2(cutmel)
        cutmel = self.batchnorm2(cutmel)

        cutmel = cutmel.permute(0, 2, 3, 1).view(b*t, f, -1)
        cutmel, _ = self.bigru(cutmel)
        forward = cutmel[:, -1, :]
        backward = cutmel[:, 0, :]
        hidden = torch.cat([forward, backward], dim = -1).view(b, t, -1).transpose(-1,-2)

        hidden = self.conv1d_1(hidden)
        hidden = self.batchnorm3(hidden)
        hidden = hidden + speaker.unsqueeze(-1).expand(-1,-1, t)
        hidden_ = self.conv1d_2(hidden)
        hidden = self.batchnorm4(hidden + hidden_).transpose(-1,-2).masked_fill(x_mask.unsqueeze(-1), 0)
        return hidden

class global_context(nn.Module):
    def __init__(self, model_config):
        super(global_context, self).__init__()
        hidden_dim = model_config["transformer"]["encoder_hidden"]
        self.l_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            bidirectional=False,
            batch_first=True,
        )
        self.h_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        #self.layernorm = nn.BatchNorm1d(hidden_dim)

    def forward(self, sbert, l_context, g_past):
        """
        sbert b, hidden
        l_context b, l, hidden
        g_past b, hidden
        """
        l_context, _ = self.l_gru(l_context)
        if g_past is not None:
            g_past = l_context[:,-1,:] + g_past
        else:
            g_past = l_context[:, -1, :]
        g_past = self.out_linear(g_past)
        if g_past is not None:
            sbert = sbert + g_past
        g_present = self.out_linear(sbert)
        return g_present

class local_context(nn.Module):
    def __init__(self, model_config):
        super(local_context, self).__init__()
        n_head = model_config["dialogue_predictor"]["n_heads"]
        hidden_dim = model_config["transformer"]["encoder_hidden"]
        kernel1d = model_config["dialogue_predictor"]["1d_kernel"]
        dropout = model_config["dialogue_predictor"]["dropout"]
        self.slf_attention = MultiHeadAttention(n_head, hidden_dim, hidden_dim, hidden_dim, dropout)
        self.layernorm1 = Layernorm(hidden_dim)
        self.layernorm2 = LayerNormT(hidden_dim)
        self.conv1d = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel1d, padding = kernel1d//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, bert, speaker, g_past, bert_mask):
        if g_past is None:
            bert = bert + speaker.unsqueeze(1).expand(-1, bert.size(1), -1)
        else:
            bert = bert + speaker.unsqueeze(1).expand(-1, bert.size(1), -1) + g_past.unsqueeze(1).expand(-1, bert.size(1), -1)
        attn_mask = bert_mask.unsqueeze(1) * bert_mask.unsqueeze(-1)
        bert_, _ = self.slf_attention(bert, bert, bert, attn_mask)
        bert = self.layernorm1(bert + bert_).transpose(-1,-2)
        bert_ = self.conv1d(bert).masked_fill(bert_mask.unsqueeze(1), 0)
        bert = self.layernorm2(bert + bert_).transpose(-1,-2)
        return bert

class dialogue_context(nn.Module):
    def __init__(self, model_config):
        super(dialogue_context, self).__init__()
        self.history_length = model_config["dialogue_predictor"]["history_length"]
        bert_in = model_config["dialogue_predictor"]["bert_in"]
        sbert_in = model_config["dialogue_predictor"]["sbert_in"]
        hidden_dim = model_config["transformer"]["encoder_hidden"]
        kernel1d = model_config["dialogue_predictor"]["1d_kernel"]
        dropout = model_config["dialogue_predictor"]["dropout"]

        self.bert_linear = nn.Linear(bert_in, hidden_dim)
        self.sbert_linear = nn.Linear(sbert_in, hidden_dim)
        self.token_duration = VariancePredictor(model_config)


        self.local_context = nn.ModuleList()
        self.global_context = nn.ModuleList()
        for i in range( self.history_length + 1 ):
            self.local_context.append(local_context(model_config))
            self.global_context.append(global_context(model_config))

        self.cont1d_1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel1d, padding= kernel1d//2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cont1d_2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel1d, padding= kernel1d//2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layernorm1 = LayerNormT(hidden_dim)
        self.layernorm2 = LayerNormT(hidden_dim)

        self.proj_m = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.proj_logs = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, h_bert, h_sbert, h_speaker, bert, sbert, speaker, bert_mask, t_duration_target):
        h_bert = self.bert_linear(h_bert)
        bert = self.bert_linear(bert)

        # length predictor
        log_token_duration = self.token_duration(bert, bert_mask)
        if t_duration_target is not None:
            t_round = torch.clamp(
                (torch.round(torch.exp(log_token_duration) - 1)),
                min=0,
            )
        else:
            t_round = t_duration_target

        h_sbert = self.sbert_linear(h_sbert)
        sbert = self.sbert_linear(sbert)
        g_context = None
        for i in range(self.history_length):
            l_context = self.local_context[i](h_bert[:,i, :,:], h_speaker[:,i,:], g_context, bert_mask)
            g_context = self.global_context[i](h_sbert[:,i,:], l_context, g_context)

        l_context = self.local_context[self.history_length](bert, speaker, g_context, bert_mask)
        g_context = self.global_context[self.history_length](sbert, l_context, g_context)

        d_context = (l_context + g_context.unsqueeze(1).expand(-1, l_context.size(1), -1)).transpose(-1,-2)

        d_context_ = self.cont1d_1(d_context).masked_fill(bert_mask.unsqueeze(1), 0)
        d_context = self.layernorm1(d_context + d_context_)
        d_context_ = self.cont1d_1(d_context).masked_fill(bert_mask.unsqueeze(1), 0)
        d_context = self.layernorm2(d_context + d_context_)
        mu = self.proj_m(d_context).transpose(-1,-2).masked_fill(bert_mask.unsqueeze(-1), 0)
        logs = self.proj_logs(d_context).transpose(-1,-2).masked_fill(bert_mask.unsqueeze(-1), 0)
        return mu, logs, log_token_duration, t_round

class sampler(nn.Module):
    def __init__(self, model_config):
        super(sampler, self).__init__()
        n_mel = model_config["dialogue_predictor"]["n_mels"]
        kernel_size = model_config["flowdecoder"]["kernel_size"]
        dropout = model_config["flowdecoder"]["dropout"]

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_mel, n_mel, kernel_size, padding = kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_mel, n_mel, kernel_size, padding = kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layernorm1 = LayerNormT(n_mel)
        self.layernorm2 = LayerNormT(n_mel)
        self.proj_m = nn.Conv1d(n_mel, n_mel, 1)
        self.proj_logs = nn.Conv1d(n_mel, n_mel, 1)

    def forward(self, x, x_mask):
        x = x.transpose(-1,-2)
        x_ = self.conv1(x).masked_fill(x_mask.unsqueeze(1), 0)
        x = self.layernorm1(x+x_)
        x_ = self.conv2(x).masked_fill(x_mask.unsqueeze(1), 0)
        x = self.layernorm2(x + x_)
        mu = self.proj_m(x).masked_fill(x_mask.unsqueeze(1), 0)
        logs = self.proj_logs(x).masked_fill(x_mask.unsqueeze(1), 0)

        return mu, logs