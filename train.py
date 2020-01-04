# stdlib
import logging
import os
import sys
import argparse
# thirdparty
import torch
from torch import optim, nn
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


def train(data_loader, epochs, entropy_size, models, visual):
    logger.debug("Initializing training...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = data_loader.batch_size
    
    discriminator = Discriminator()
    if models:
        discriminator.load_state_dict(torch.load(models[0]))
    discriminator.to(device)
    optimizer_dis = optim.SGD(
        discriminator.parameters(), lr=0.0001, momentum=0.90)
    loss_dis = nn.BCELoss()

    generator = Generator(entropy_size=entropy_size)
    if models:
        generator.load_state_dict(torch.load(models[1]))
    generator.to(device)
    optimizer_gen = optim.Adam(generator.parameters(), lr=0.0001)
    loss_gen = nn.BCELoss()
    L1_lambda = 0.0001

    losses = []
    fake_losses = []
    gen_losses = []
    logger.debug("Init done")
    logger.debug("Starting training with {} epochs".format(epochs))
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
            optimizer_dis.zero_grad()
            out = discriminator(data)
            label = Variable(0.7 + 0.3 * torch.rand(len(out))).to(device) # make labels noisy (real: 0.7-1.2)
            loss = loss_dis(out, label).to(device)
            loss.backward()
            fake_data = generator.generate_data(batch_size, device)
            out = discriminator(fake_data)
            label = Variable(0.3 * torch.rand(len(out))).to(device) # make labels noisy (fake: 0-0.3)
            fake_loss = loss_dis(out, label).to(device)
            fake_loss.backward()
            optimizer_dis.step()
            logger.debug("Loss: {}".format(loss.item()))
            logger.debug("Fake Loss: {}".format(fake_loss.item()))
            running_loss += loss.item()
            running_fake_loss += fake_loss.item()

        losses.append(running_loss/len(data_loader))
        fake_losses.append(running_fake_loss/len(data_loader))
        logger.debug("Running loss: {}".format(losses[-1]))
        logger.debug("Running fake loss: {}".format(fake_losses[-1]))

        running_gen_loss = 0.0
        logging.debug("Training generator...")
        discriminator.eval()
        generator.train()
        for step in range(len(data_loader)):
            optimizer_gen.zero_grad()
            logger.debug("Batch: {}/{}".format(step, len(data_loader)))
            fake_data = generator.generate_data(batch_size, device, train=True)
            fake_data = fake_data.to(device)
            if visual:
                visualize_sample(fake_data.cpu())
            out = discriminator(fake_data)
            reg_loss = torch.norm(fake_data, p=1)/len(fake_data)
            gen_loss = loss_gen(out, Variable(torch.ones([batch_size, 1])).to(device))
            gen_loss += L1_lambda * reg_loss
            gen_loss.backward()
            optimizer_gen.step()
            logger.debug("Gen loss: {}".format(gen_loss.item()))
            running_gen_loss += gen_loss.item()
        gen_losses.append(running_gen_loss/len(data_loader))
        logger.debug("Running generator loss: {}".format(gen_losses[-1]))

    return generator, discriminator


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Trains a GANN on audio signals")
    parser.add_argument("--input_data", dest="input_data", required=True,
                        help="The npz archive with training data created with the `preprocessing.py` script")
    parser.add_argument("--epochs", dest="epochs", type=int,
                        help="The number of epochs to train for (default=10)", default=10)
    parser.add_argument("--batch_size", dest="batch_size", type=int,
                        help="The batch size used during training (default is 4)", default=4)
    parser.add_argument("--entropy_size", dest="entropy_size", type=int,
                        help="The size of the entropy vector used as input for the generator (default is 1024)", default=1024)
    parser.add_argument("--models", dest="models", type=str, nargs=2,
                        help="Pretrained models to use for further training; Discriminator, then Generator (default None)")
    parser.add_argument("--visual", dest="visual", action="store_true",
                        help="Whether to show visual representations of the training samples and generated results as images")
    args = parser.parse_args()
    train_data = AudioSnippetDataset(args.input_data)
    data_loader = DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=4)
    gen, dis = train(data_loader=data_loader, entropy_size=args.entropy_size, epochs=args.epochs, models=args.models, visual=args.visual)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generated = gen.generate_data(1, device).squeeze(0)
    out = generated.cpu().detach().numpy()
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
    fn_prefix = str(args.input_data) + "_" + str(args.epochs) +  "_" + str(args.batch_size) + "_" + str(args.entropy_size)
    fn_wav = os.path.join(wav_dir, fn_prefix + ".wav")
    fn_gen_model = os.path.join(model_dir, fn_prefix + "_gen.model")
    fn_dis_model = os.path.join(model_dir, fn_prefix + "_dis.model")
    time_out = librosa.istft(out_complex)
    sr = 22050
    librosa.output.write_wav(fn_wav, time_out, sr)
    torch.save(gen.state_dict(), fn_gen_model)
    torch.save(dis.state_dict(), fn_dis_model)
