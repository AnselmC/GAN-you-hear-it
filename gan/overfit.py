# stdlib
import logging
import os
import datetime
import sys
import argparse
# thirdparty
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
import librosa
import numpy as np
# own
from gann import ConvolutionalGenerator, LinearGenerator, Discriminator
from dataset import AudioSnippetDataset
from helpers import visualize_sample

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Training")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

def overfit(sample, generator_type, lr, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if generator_type == "conv":
        generator = ConvolutionalGenerator(
            device, entropy_size=128, lr=lr)
    else:
        generator = LinearGenerator(
            device, entropy_size=10, lr=lr)

    target = sample.to(device)
    visualize_sample(target.cpu().unsqueeze(0))
    input("Press any button to continue")
    for epoch in range(epochs):
        logger.debug("Epoch: {} of {}".format(epoch, epochs))
        generator.optim.zero_grad()
        fake_data = generator.generate_data(1, device, train=True)
        gen_loss = torch.norm(torch.sub(fake_data, target))
        gen_loss.backward()
        generator.optim.step()
        logger.debug("Gen loss: {}".format(gen_loss.item()))
        visualize_sample(fake_data.cpu())

    return generator

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", dest="lr", type=float, default=0.01)
    parser.add_argument("-e", dest="epochs", type=int, default=500)
    parser.add_argument("-t", dest="type", type=str, choices={"conv", "linear"}, default="conv")
    parser.add_argument("-s", dest="sample", type=str, default="training_data_new/arr_123.npy")
    args = parser.parse_args()
    sample = np.load(args.sample)
    real = np.real(sample)
    imag = np.imag(sample)
    sample = np.array([real, imag])
    sample = Variable(torch.Tensor(sample))
    gen = overfit(sample, args.type, args.lr, args.epochs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fn_gen = "overfit_sample.wav"
    fn_gt = "groundtruth_sample.wav"
    sr = 22050
    sample = sample.numpy()
    sample_complex = sample[0] + 1j*sample[1]
    time_out_sample = librosa.istft(sample_complex)
    print(time_out_sample.min())
    print(time_out_sample.max())
    generated = gen.generate_data(1, device)
    out_gen = generated.squeeze(0).cpu().detach().numpy()
    out_gen_complex = out_gen[0] + 1j*out_gen[1]
    time_out = librosa.istft(out_gen_complex)
    print(time_out.min())
    print(time_out.max())
    librosa.output.write_wav(fn_gen, time_out, sr)
    librosa.output.write_wav(fn_gt, time_out_sample, sr)
