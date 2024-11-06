#adopt Glow TTS Flow based spectrogram decoder
import torch
from torch import nn
from torch.nn import functional as F

class ActNorm(nn.Module):
    def __init__(self, model_config):
        super(ActNorm,self).__init__()
        self.channels = model_config["dialogue_predictor"]["n_mels"] * model_config["flowdecoder"]["n_sqz"]

        self.logs = nn.Parameter(torch.zeros(1, self.channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, self.channels, 1))
        self.initialized = False

    def forward(self, x, x_mask=None, speaker=None, gen=False):
        if x_mask is None:
            x_mask = torch.zeros(x.size(0), 1, x.size(2)).to(device=x.device, dtype = torch.bool)
        x_len = torch.sum(~x_mask, [1,2])
        if not self.initialized:
            self.initialize(x,x_mask)
            self.initialized = True
        if gen:
            z =((x- self.bias) * torch.exp(-self.logs)).masked_fill(x_mask, 0)
            logdet = None
        else:
            z = (self.bias + torch.exp(self.logs) * x).masked_fill(x_mask, 0)
            logdet = torch.sum(self.logs) * x_len
        return z, logdet
    def store_inverse(self):
            pass

    def initialize(self, x, x_mask):
        with torch.no_grad():
            denom = torch.sum(~x_mask,[0,2])
            m = torch.sum(x.masked_fill(x_mask, 0), [0, 2])/denom
            m_sq = torch.sum((x*x).masked_fill(x_mask, 0),[0,2])/denom
            v = m_sq - (m **2)
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

            bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape).to(dtype = self.bias.dtype)
            logs_init = (-logs).view(*self.logs.shape).to(dtype = self.logs.dtype)

            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)

class InvConvNear(nn.Module):
    def __init__(self, model_config):
        super(InvConvNear, self).__init__()
        self.channel = model_config["dialogue_predictor"]["n_mels"] * model_config["flowdecoder"]["n_sqz"]
        self.n_split = model_config["flowdecoder"]["n_split"]
        self.no_jacobian = False
        assert(self.n_split % 2 == 0)

        w_init = torch.qr(torch.FloatTensor(self.n_split, self.n_split).normal_())[0]
        if torch.det(w_init) <0:
            w_init[:,0] = -1 * w_init[:,0]
            self.weight = nn.Parameter(w_init)
    def forward(self, x, x_mask = None, speaker = None, gen = False):
        b, c, t = x.size()
        assert(c % self.n_split == 0)
        if x_mask is None:
            x_mask = 0
            x_len = torch.ones((b,), dtype= x.dtype, device = x.device) * t
        else:
            x_len = torch.sum(~x_mask,[1,2])

        x = x.view(b, 2, c//self.n_split, self.n_split//2, t)
        x = x.permute(0,1,3,2,4).contiguous().view(b, self.n_split, c//self.n_split, t)

        if gen:
            if hasattr(self, "weight_inv"):
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float()).to(dtype = self.weight.dtype)
            logdet = None
        else:
            weight = self.weight
            if self.no_jacobian :
                logdet = 0
            else:
                logdet = torch.logdet(self.weight) * (c/self.n_split) * x_len
        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = F.conv2d(x, weight)

        z = z.view(b, 2, self.n_split//2, c//self.n_split, t)
        z = z.permute(0,1,3,2,4).contiguous().view(b, c, t).masked_fill(x_mask, 0)
        return z, logdet

    def store_inverse(self):
        self.weight_int = torch.inverse(self.weight.float()).to(dtype = self.weight.dtype)

class WN(nn.Module):
    def __init__(self, model_config):
        super(WN, self).__init__()
        speaker_dim = model_config["transformer"]["encoder_hidden"]
        self.hidden_dim = model_config["flowdecoder"]["filter_channel"]
        dilation_rate= model_config["flowdecoder"]["dilation_rate"]
        self.n_layers = model_config["flowdecoder"]["n_layers"]
        kernel_size = model_config["flowdecoder"]["kernel_size"]
        dropout = model_config["flowdecoder"]["dropout"]

        self.in_layers = nn.ModuleList()
        self.res_skip_layers= nn.ModuleList()
        self.drop = nn.Dropout(dropout)

        cond_layer = nn.Conv1d(speaker_dim, 2*self.hidden_dim*self.n_layers, 1)
        self.cond_layer = nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(self.n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) /2)
            in_layer = nn.Conv1d(self.hidden_dim, 2* self.hidden_dim, kernel_size, dilation = dilation, padding=padding)
            in_layer = nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            if i<self.n_layers-1:
                res_skip_channels = 2 * self.hidden_dim
            else:
                res_skip_channels = self.hidden_dim

            res_skip_layer = nn.Conv1d(self.hidden_dim, res_skip_channels , 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name = 'weight')
            self.res_skip_layers.append(res_skip_layer)
    def forward(self,x, x_mask = None, speaker = None, gen = False):
        output=  torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_dim])

        if speaker is not None:
            speaker = self.cond_layer(speaker.unsqueeze(-1))
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = self.drop(x_in)
            if speaker is not None:
                cond_offset = i * 2 * self.hidden_dim
                speaker_l = speaker[:, cond_offset:cond_offset+2*self.hidden_dim,:]
            else:
                speaker_l = torch.zeros_like(x_in)
            acts = self.fused_add_tanh_sigmoid_multiply(x_in, speaker_l, n_channels_tensor)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i <self.n_layers -1:
                x = (x+res_skip_acts[:,:self.hidden_dim,:]).masked_fill(x_mask, 0)
                output = output + res_skip_acts[:,self.hidden_dim:, :]
            else:
                output = output + res_skip_acts
            return output.masked_fill(x_mask, 0)
    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            nn.utils.remove_weight_norm(l)

    def fused_add_tanh_sigmoid_multiply(self, x ,speaker, n_tensor):
        n_channels = n_tensor[0]
        in_act = x + speaker
        t_act = torch.tanh(in_act[:,:n_channels,:])
        s_act = torch.sigmoid(in_act[:,n_channels:,:])
        act = t_act * s_act
        return act

class CouplingBlock(nn.Module):
    def __init__(self, model_config):
        super(CouplingBlock, self).__init__()
        self.inchannel = model_config["dialogue_predictor"]["n_mels"] * model_config["flowdecoder"]["n_sqz"]
        hidden_channel = model_config["flowdecoder"]["filter_channel"]
        self.sigmoid_scale = False

        start = nn.Conv1d(self.inchannel//2, hidden_channel, 1)
        start = nn.utils.weight_norm(start)
        self.start = start
        end = nn.Conv1d(hidden_channel, self.inchannel, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        self.wn = WN(model_config)
    def forward(self, x, x_mask=None, speaker=None, gen=False):
        b, c, t = x.size()
        if x_mask is None:
            x_mask = 0
        x_0, x_1 = x[:,:self.inchannel//2], x[:,self.inchannel//2:]

        x = self.start(x_0).masked_fill(x_mask, 0)
        x = self.wn(x, x_mask, speaker)
        out = self.end(x)

        z_0 = x_0
        m = out[:, : self.inchannel//2, :]
        logs = out[:, self.inchannel//2:, :]

        if self.sigmoid_scale:
            logs = torch.log(1e-6 + torch.sigmoid(logs + 2))
        if gen:
            z_1 = ((x_1 -m)* torch.exp(-logs)).masked_fill(x_mask, 0)
            logdet = None
        else:
            z_1 = (m + x_1 *torch.exp(logs)).masked_fill(x_mask,0)
            logdet = torch.sum(logs.masked_fill(x_mask, 0),[1,2])

        z= torch.cat([z_0, z_1],1)
        return z, logdet
    def store_inverse(self):
        self.wn.remove_weight_norm()

class FlowSpecDecoder(nn.Module):
    def __init__(self, model_config):
        super(FlowSpecDecoder, self).__init__()
        n_blocks = model_config["flowdecoder"]["n_blocks"]
        self.n_sqz = model_config["flowdecoder"]["n_sqz"]

        self.flows = nn.ModuleList()
        for b in range(n_blocks):
            self.flows.append(ActNorm(model_config))
            self.flows.append(InvConvNear(model_config))
            self.flows.append(CouplingBlock(model_config))

    def forward(self, x, x_mask, speaker, gen = False):
        """
        x [b, hidden, length]
        x_mask [b, length]
        speaker [b, hidden]
        """

        if gen:
            flows = reversed(self.flows)
            logdet_tot = None
        else:
            flows = self.flows
            logdet_tot = 0

        x, x_mask = self.squeeze(x, x_mask , sqz= self.n_sqz)
        for f in flows:
            if not gen:
                x, logdet = f(x, x_mask, speaker = speaker, gen = gen)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, speaker = speaker, gen = gen)
        x, x_mask = self.unsqueeze(x, x_mask, sqz= self.n_sqz)

        return x, logdet_tot, x_mask

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()

    def squeeze(self,x, x_mask=None, sqz = 2):
        b, c, t = x.size()
        t = (t//sqz) * sqz
        x = x[:,:,:t]
        x_sqz = x.view(b, c, t//sqz, sqz)
        x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c*sqz, t//sqz)
        if x_mask is not None:
            x_mask = x_mask.unsqueeze(-2)
            x_mask = x_mask[:,:,sqz-1::sqz]
        else:
            x_mask = torch.zeros(b, 1, t//sqz).to(device = x.device, dtype = torch.bool)
        return x_sqz.masked_fill(x_mask, 0), x_mask

    def unsqueeze(self, x, x_mask, sqz = 2):
        b, c, t = x.size()
        x_unsqz = x.view(b, sqz, c//sqz, t)
        x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c//sqz, t*sqz)

        if x_mask is not None:
            x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, sqz).view(b, 1, t*sqz)
        else:
            x_mask = torch.zeros(b, 1, t*sqz).to(device = x.device, dtype = torch.bool)
        return x_unsqz.masked_fill(x_mask, 0), x_mask.squeeze(1)