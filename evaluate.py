import argparse
import os
import numpy as np

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample, to_device_eval, synth_samples
from model import FittedLoss
from dataset import Dialogue_dataset, Dialogue_dataset_neval

from scipy.io import wavfile
from utils.model import vocoder_infer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dialogue_dataset_neval(
        "dialogue_val.txt", model_config, preprocess_config )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FittedLoss(preprocess_config, model_config, train_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(9)]
    for batchs, basename_list in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward

                output = model(*(batch), gen=False, steps=step)

                # Cal Loss
                losses = Loss(batch, output, step)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Post Mel Loss: {:.4f},  Mel MLE Loss: {:.4f},  Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Token Duration Loss: {:.4f}, Condition Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    with torch.no_grad():
        output1 = model(*(batch), gen=True, steps=step)
    if logger is not None:
        fig, fig0, wav_reconstruction, wav_prediction= synth_one_sample(
            batch,
            output1,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/Mel_spec_Prosody",
            step = step
        )
        log(
            logger,
            fig=fig0,
            tag="Validation/Condition",
            step = step
        )

        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/reconstructed",
            step = step
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/synthesized_prosody",
            step = step
        )

    if step % train_config["step"]["save_step"] == 0:
        with torch.no_grad():
            for batchs, basename_list in loader:
                for indx, batch in enumerate(batchs):
                    batch = to_device_eval(batch, device)
                    try :
                        with torch.no_grad():
                            # Forward
                            output = model(*(batch), gen=True, steps=step)
                            wav_predictions = vocoder_infer(
                                output[1], vocoder, model_config, preprocess_config )
                            wavfile.write(os.path.join(train_config["path"]["result_path"], "{}.wav".format(basename_list[indx])), 22050, np.array(wav_predictions))
                    except:
                        print(basename_list[indx])


    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)