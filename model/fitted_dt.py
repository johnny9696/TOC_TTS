import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from transformer.Models import get_sinusoid_encoding_table
from transformer.SubLayers import MultiHeadAttention
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
from .dialogue_nmodule import mel_prosody, history_dialogue_predictor
from .flowdecoder import FlowSpecDecoder
from .modules import LengthRegulator, VariancePredictor

class Fitted_DT(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config, t_step):
        super(Fitted_DT, self).__init__()
        self.model_config = model_config
        self.n_mels = model_config["dialogue_predictor"]["n_mels"]
        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)  # version19
        self.history_context = history_dialogue_predictor(model_config) #version 18_flow
        self.mel_emb = mel_prosody(model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()
        self.postnet_decoder = FlowSpecDecoder(model_config)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            n_speaker = model_config["n_speaker"]+1
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
        self.t_step = t_step
        self.LR = LengthRegulator()
        self.length_regulator = LengthRegulator()
        #if retraining remove duration predictor
        self.duration_predictor = VariancePredictor(model_config)

        #if we use without token duration predictor token phoneme cross attention
        self.cross_atten = MultiHeadAttention(2, model_config["transformer"]["encoder_hidden"], model_config["transformer"]["encoder_hidden"], model_config["transformer"]["encoder_hidden"])




    def forward(
        self,
        texts,
        src_lens,
        max_src_len,
        bert,
        bert_length,
        max_bert_length,
        sbert,
        h_bert,
        h_sbert,
        speakers,
        h_speaker,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        pd_targets=None,
        td_targets=None,
        tdd_targets=None,
        p_targets=None,
        e_targets=None,
        cutmel=None,
        cutw2v=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        temperature = 1.0,
        gen = False,
        steps = 0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        bert_masks = get_mask_from_lengths(bert_length, max_bert_length)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        if gen:
            mels = None
            mel_lens = None
            max_mel_len = None
            pd_targets = None
            td_targets = None
            tdd_targets = None
            p_targets = None
            e_targets = None
            cutmel = None
            cutw2v = None


        output = self.encoder(texts, src_masks)

        h_speaker_emb = self.speaker_emb(h_speaker)
        speaker_emb = self.speaker_emb(speakers)

        phone_prosody, log_t_predictions, t_round = self.history_context(h_bert, h_sbert, bert, sbert,
                                                                         bert_masks, h_speaker,h_speaker_emb,
                                                                         speakers, speaker_emb, tdd_targets, gen = gen)
        if cutmel is not None:
            mel_prosody = self.mel_emb(cutmel, speaker_emb, bert_masks)
        else:
            mel_prosody = None

        if not gen:
            context, _ = self.LR(mel_prosody, t_round, max_src_len)
            #output = output + context

            # without token duration predictor
            atten_mask = src_masks.unsqueeze(1) * src_masks.unsqueeze(-1)
            context, _ = self.cross_atten(output, context, context, atten_mask)
            output = output + context


        else:
            context, _ = self.LR(phone_prosody, t_round, max_src_len)
            #output = output + context


            # without token duration predictor
            atten_mask = src_masks.unsqueeze(1) * src_masks.unsqueeze(-1)
            context, _ = self.cross_atten(output, context, context, atten_mask)
            output = output + context


        if self.speaker_emb is not None:
            output = output + speaker_emb.unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            pd_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)
        postnet_output = self.postnet(output) + output

        if not gen:
            postnet_output = postnet_output.transpose(-1,-2)
            z, logdet, mel_masks = self.postnet_decoder(mels.transpose(-1,-2), mel_masks, postnet_output, gen = False)
            mel_gen = None
        else:
            z = torch.normal(0.0, temperature,[mel_masks.shape[0], self.n_mels, mel_masks.shape[1]]).to(mel_masks.device)
            mel_gen, logdet, mel_masks = self.postnet_decoder(z, mel_masks, postnet_output.transpose(-1,-2), gen=True)


        return (
            (z, logdet),
            mel_gen,
            p_predictions,
            e_predictions,
            log_d_predictions,
            log_t_predictions,
            d_rounded,
            src_masks,
            bert_masks,
            mel_masks,
            src_lens,
            mel_lens,
            mel_prosody,
            phone_prosody,
            context,
            output,
            postnet_output
        )