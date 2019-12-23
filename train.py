# stdlib
import logging
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

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("Training")


def train(data_loader, epochs, entropy_size):
    logger.debug("Initializing training...")
    batch_size = data_loader.batch_size
    generator = Generator(entropy_size=entropy_size)
    optimizer_gen = optim.SGD(generator.parameters(), lr=0.0001, momentum=0.95)
    loss_gen = nn.BCELoss()
    discriminator = Discriminator()
    optimizer_dis = optim.SGD(
        discriminator.parameters(), lr=0.0001, momentum=0.95)
    loss_dis = nn.BCELoss()

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
            "Training discriminator with {} batch".format(len(data_loader)))
        for i, data in enumerate(data_loader):
            logger.debug("Batch: {}/{}".format(i, len(data_loader)))
            optimizer_dis.zero_grad()
            out = discriminator(data)
            loss = loss_dis(out, Variable(torch.ones([batch_size, 1])))
            loss.backward()
            fake_data = generator.generate_data(batch_size)
            out = discriminator(fake_data)
            fake_loss = loss_dis(out, Variable(torch.zeros([batch_size, 1])))
            fake_loss.backward()
            optimizer_dis.step()
            logger.debug("Loss: {}".format(loss.item()))
            logger.debug("Fake Loss: {}".format(fake_loss.item()))
            running_loss += loss.item()
            running_fake_loss += fake_loss.item()

        losses.append(running_loss/len(train_data))
        fake_losses.append(running_fake_loss/len(train_data))
        logger.debug("Running loss: {}".format(losses[-1]))
        logger.debug("Running fake loss: {}".format(fake_losses[-1]))

        running_gen_loss = 0.0
        logging.debug("Training generator...")
        for step in range(len(data_loader)):
            optimizer_gen.zero_grad()
            fake_data = generator.generate_data(batch_size, train=True)
            out = discriminator(fake_data)
            gen_loss = loss_gen(out, Variable(torch.ones([batch_size, 1])))
            gen_loss.backward()
            optimizer_gen.step()
            logger.debug("Gen loss: {}".format(gen_loss.item()))
            running_gen_loss += gen_loss.item()
        gen_losses.append(running_gen_loss/len(train_data))
        logger.debug("Running generator loss: {}".format(gen_losses[-1]))

    return generator, discriminator


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Trains a GANN on audio signals")
    parser.add_argument("-i, --input_data", dest="input_data", required=True,
                        help="The npz archive with training data created with the `preprocessing.py` script")
    parser.add_argument("-e, --epochs", dest="epochs", type=int,
                        help="The number of epochs to train for (default=10)", default=10)
    parser.add_argument("-bs, --batch_size", dest="batch_size", type=int,
                        help="The batch size used during training (default is 4)", default=4)
    parser.add_argument("-en, --entropy_size", dest="entropy_size", type=int,
                        help="The size of the entropy vector used as input for the generator (default is 1024)", default=1024)
    args = parser.parse_args()
    train_data = AudioSnippetDataset(args.input_data)
    data_loader = DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=4)
    gen, dis = train(data_loader=data_loader, entropy_size=args.entropy_size, epochs=args.epochs)

    generated = gen.generate_data(1)
    out = generated.detach().numpy()
    out_complex = out[0] + 1j*out[1]
    time_out = librosa.istft(out_complex)
    sr = 22050
    librosa.output.write_wav("out.wav", time_out, sr)
