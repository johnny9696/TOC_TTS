import torch
import torch.nn as nn
import math

def MLELoss_det(z, logdet = None, x_mask = None):
    z_logs = torch.zeros_like(z*~x_mask)
    l = torch.sum(z_logs) + 0.5 * torch.sum(z**2 * torch.exp(-2 * z_logs))
    if logdet is not None:
        l = l - torch.sum(logdet)
    if x_mask == None:
        l = l / torch.sum(torch.ones_like(z))
    else:
        l = l / torch.sum(torch.ones_like(z).masked_fill(x_mask, 0))
    l = l + 0.5 * math.log(2 * math.pi)
    return l

def MLELoss_det_z(z, z_mu, z_logs, logdet = None, x_mask = None):
    l = torch.sum(z_logs) + 0.5 * torch.sum((z-z_mu)**2 * torch.exp(-2 * z_logs))
    if logdet is not None:
        l = l - torch.sum(logdet)
    if x_mask == None:
        l = l / torch.sum(torch.ones_like(z))
    else:
        l = l / torch.sum(torch.ones_like(z))
    l = l + 0.5 * math.log(2 * math.pi)
    return l

class FittedLoss(nn.Module):
    """ FittedLoss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(FittedLoss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]

        self.warmupstep = train_config["optimizer"]["warm_up_step"]

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.KLD_loss = nn.KLDivLoss()

    def forward(self, inputs, predictions, step):
        (
            mel_targets,
            _,
            _,
            duration_targets,
            t_duration_targets,
            token_dur_targets,
            pitch_targets,
            energy_targets,
            _,
            phone_emotion_targets,
        ) = inputs[11:]
        (
            (z, logdet),
            mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            log_token_duration_predictions,
            _,
            src_masks,
            bert_masks,
            mel_masks,
            _,
            _,
            mel_prosody,
            phone_prosody,
            phone_emotion_predict,
            prenet_mel_prediction,
            postnet_mel_prediction,
        ) = predictions
        src_masks = ~src_masks
        bert_masks = ~bert_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_masks = mel_masks[:, :mel_masks.shape[1]]
        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        phone_emotion_targets.requires_grad = False
        mel_targets.requires_grad = False


        log_token_duration_targets = torch.log(token_dur_targets.float() + 1)
        log_token_duration_targets.requires_grad = False
        mel_prosody_detach = mel_prosody.detach()

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        pre_mel_loss = self.mae_loss(mel_targets, prenet_mel_prediction)
        postnet_mel_loss = self.mae_loss(mel_targets, postnet_mel_prediction.transpose(-1,-2))
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets) #version 19
        energy_loss = self.mse_loss(energy_predictions, energy_targets) #version 19
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        token_duration_loss = self.mse_loss(log_token_duration_predictions, log_token_duration_targets)

        mel_loss = MLELoss_det(z, logdet, ~mel_masks.unsqueeze(1))
        condition_loss = self.mse_loss(mel_prosody_detach, phone_prosody)

        if step <=self.warmupstep:
            condition_loss = condition_loss * 0
            pitch_loss = pitch_loss * 0
            energy_loss = energy_loss * 0

        total_loss = (
                mel_loss + pre_mel_loss + postnet_mel_loss +
                duration_loss +
                pitch_loss + energy_loss +
                token_duration_loss + condition_loss
        )

        return (
            total_loss,
            pre_mel_loss,
            postnet_mel_loss,
            mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            token_duration_loss,
            condition_loss,
        )



class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
