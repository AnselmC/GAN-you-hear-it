# stdlib
import logging
import os
import sys
import argparse
# thirdparty
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import librosa
import numpy as np
# own
from gann import Generator, Discriminator
from dataset import AudioSnippetDataset
from helpers import visualize_sample

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("Training")

__description__ = r"""
              ____    ______  __  __      __    __
             /\  _`\ /\  _  \/\ \/\ \    /\ \  /\ \
             \ \ \L\_\ \ \L\ \ \ `\\ \   \ `\`\\/'/ ___   __  __
              \ \ \L_L\ \  __ \ \ , ` \   `\ `\ /' / __`\/\ \/\ \
               \ \ \/, \ \ \/\ \ \ \`\ \    `\ \ \/\ \L\ \ \ \_\ \
                \ \____/\ \_\ \_\ \_\ \_\     \ \_\ \____/\ \____/
                 \/___/  \/_/\/_/\/_/\/_/      \/_/\/___/  \/___/


           __  __                              ______   __       _
          /\ \/\ \                            /\__  _\ /\ \__  /'_`\
          \ \ \_\ \     __     __     _ __    \/_/\ \/ \ \ ,_\/\_\/\`\
           \ \  _  \  /'__`\ /'__`\  /\`'__\     \ \ \  \ \ \/\/_//'/'
            \ \ \ \ \/\  __//\ \L\.\_\ \ \/       \_\ \__\ \ \_  /\_\
             \ \_\ \_\ \____\ \__/.\_\\ \_\       /\_____\\ \__\ \/\_\
              \/_/\/_/\/____/\/__/\/_/ \/_/       \/_____/ \/__/  \/_/

A Generative Adversarial Network for generating music samples.
"""


def train(data_loader, epochs, entropy_size, models, visual):
    logger.debug("Initializing training...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = data_loader.batch_size
    
    discriminator = Discriminator(device, model=models[0] if models else None, lr=0.0001)

    generator = Generator(device, model=models[1] if models else None, entropy_size=entropy_size, lr=0.0001)
    L1_lambda = 100 # 0.0001

    dim4d = lambda a, b, c, d: a*b*c*d

    losses = []
    fake_losses = []
    gen_losses = []
    logger.debug("Init done")
    logger.debug("Starting training with {} samples and {} epochs".format(len(train_data), epochs))
    for epoch in range(epochs):
        logger.debug("Epoch: {} of {}".format(epoch, epochs))

        running_loss = 0.0
        running_fake_loss = 0.0
        logger.debug(
            "Training discriminator with {} batches".format(len(data_loader)))
        discriminator.train()
        generator.eval()
        for step, data in enumerate(data_loader):
            data = data.to(device)
            logger.debug("Batch: {}/{}".format(step, len(data_loader)))
            if visual:
                visualize_sample(data.cpu())
            discriminator.optim.zero_grad()
            out = discriminator(data)
            label = Variable(0.7 + 0.3 * torch.rand(len(out), 1)).to(device) # make labels noisy (real: 0.7-1.2)
            #acc = len(out[out>=0.5])/len(out)
            loss = discriminator.loss(out, label).to(device)
            loss.backward()
            fake_data = generator.generate_data(batch_size, device, train=True)
            out = discriminator(fake_data)
            if visual:
                visualize_sample(fake_data.cpu())
            #fake_acc = len(out[out<0.5])/len(out)
            label = Variable(0.3 * torch.rand(len(out), 1)).to(device) # make labels noisy (fake: 0-0.3)
            fake_loss = discriminator.loss(out, label).to(device)
            fake_loss.backward()
            discriminator.optim.step()
            logger.debug("Loss: {}".format(loss.item()))
            #logger.debug("Acc: {}".format(acc))
            logger.debug("Fake Loss: {}".format(fake_loss.item()))
            #logger.debug("Fake acc: {}".format(acc))
            running_loss += loss.item()
            running_fake_loss += fake_loss.item()

        losses.append(running_loss/len(data_loader))
        fake_losses.append(running_fake_loss/len(data_loader))
        logger.debug("Running loss: {}".format(losses[-1]))
        logger.debug("Running fake loss: {}".format(fake_losses[-1]))

        running_gen_loss = 0.0
        logger.debug("Training generator...")
        discriminator.eval()
        generator.train()
        for step in range(len(data_loader)):
            generator.optim.zero_grad()
            logger.debug("Batch: {}/{}".format(step, len(data_loader)))
            fake_data = generator.generate_data(batch_size, device, train=True)
            if visual:
                visualize_sample(fake_data.cpu())
            out = discriminator(fake_data)
            #acc = len(out[out>=0.5])/len(out)
            # TODO: norm calculation is wrong
            reg_loss = torch.norm(fake_data, p=1)/dim4d(*fake_data.shape)
            gen_loss = generator.loss(out, Variable(torch.ones([batch_size, 1])).to(device))
            gen_loss += L1_lambda * reg_loss
            gen_loss.backward()
            generator.optim.step()
            logger.debug("Gen loss: {}".format(gen_loss.item()))
            #logger.debug("Gen Acc: {}".format(acc))
            running_gen_loss += gen_loss.item()
        gen_losses.append(running_gen_loss/len(data_loader))
        logger.debug("Running generator loss: {}".format(gen_losses[-1]))

    return generator, discriminator


if __name__ == "__main__":
    print(__description__)
    parser = argparse.ArgumentParser("Trains a GANN on audio signals")
    parser.add_argument("--input_data", dest="input_data", required=True,
                        help="The npz archive with training data created with the `preprocessing.py` script")
    parser.add_argument("--subset_size", dest="subset_size", type=int,
                        help="The subset size to use from the input folder. Entire folder is used if not given or subset size is larger than folder size")
    parser.add_argument("--epochs", dest="epochs", type=int,
                        help="The number of epochs to train for (default=10)", default=10)
    parser.add_argument("--batch_size", dest="batch_size", type=int,
                        help="The batch size used during training (default is 4)", default=4)
    parser.add_argument("--entropy_size", dest="entropy_size", type=int,
                        help="The size of the entropy vector used as input for the generator (default is 10)", default=10)
    parser.add_argument("--models", dest="models", type=str, nargs=2,
                        help="Pretrained models to use for further training; Discriminator, then Generator (default None)")
    parser.add_argument("--visual", dest="visual", action="store_true",
                        help="Whether to show visual representations of the training samples and generated results as images")
    args = parser.parse_args()
    train_data = AudioSnippetDataset(args.input_data, subset_size=args.subset_size)
    data_loader = DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=4)
    gen, dis = train(data_loader=data_loader, entropy_size=args.entropy_size, epochs=args.epochs, models=args.models, visual=args.visual)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generated = gen.generate_data(1, device)
    out = generated.squeeze(0).cpu().detach().numpy()
    out_complex = out[0] + 1j*out[1]
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    wav_dir = os.path.join(results_dir, "audio")
    if not os.path.exists(wav_dir):
        os.mkdir(wav_dir)
    model_dir = os.path.join(results_dir, "models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    fn_prefix = str(args.epochs) +  "_" + str(args.batch_size) + "_" + str(args.entropy_size) + "_" + str(len(train_data))
    fn_wav = os.path.join(wav_dir, fn_prefix + ".wav")
    fn_gen_model = os.path.join(model_dir, fn_prefix + "_gen.model")
    fn_dis_model = os.path.join(model_dir, fn_prefix + "_dis.model")
    time_out = librosa.istft(out_complex)
    sr = 22050
    librosa.output.write_wav(fn_wav, time_out, sr)
    torch.save(gen.state_dict(), fn_gen_model)
    torch.save(dis.state_dict(), fn_dis_model)
