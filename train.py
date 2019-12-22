# stdlib
import logging
import sys
import argparse
# thirdparty
import torch
from torch import optim, nn
from torch.autograd import Variable
import librosa
import numpy as np
# own
from gann import Generator, Discriminator

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("Training")


def train(train_data, epochs=100):
    logger.debug("Initializing training...")
    generator = Generator()
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
        logger.debug("Training discriminator with {} samples".format(len(train_data)))
        for step in range(len(train_data)):
            logger.debug("Sample no {} of {}".format(step, len(train_data)))
            optimizer_dis.zero_grad()
            # take real and imaginary part separately 
            real_part = np.real(train_data[step])
            ima_part = np.imag(train_data[step])
            data = np.array([real_part, ima_part])
            data = Variable(torch.Tensor(data.reshape(1, *data.shape)))
            out = discriminator(data)
            loss = loss_dis(out, Variable(torch.ones([1, 1])))
            loss.backward()
            gen_input = np.random.normal(0, 1, 1024)
            fake_data = generator(Variable(torch.Tensor(gen_input))).detach()
            out = discriminator(fake_data.view(1, *fake_data.shape))
            fake_loss = loss_dis(out, Variable(torch.zeros([1, 1])))
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
        logging.debug("Training discriminator...")
        for step in range(len(train_data)):
            optimizer_gen.zero_grad()
            gen_input = np.random.normal(0, 1, 1024)
            fake_data = generator(Variable(torch.Tensor(gen_input)))
            out = discriminator(fake_data.view(1, *fake_data.shape))
            gen_loss = loss_gen(out, Variable(torch.ones([1, 1])))
            gen_loss.backward()
            optimizer_gen.step()
            running_gen_loss += gen_loss.item()
        gen_losses.append(running_gen_loss/len(train_data))
        logger.debug("Running generator loss: {}".format(gen_losses[-1]))

    return generator, discriminator


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Trains a GANN on audio signals")
    parser.add_argument("-i, --input_data", dest="input_data", required=True,
                        help="The npz archive with training data created with the `preprocessing.py` script")
    args = parser.parse_args()
    npz_data = np.load(args.input_data)
    train_data = [elem for elem in npz_data.values() if elem.shape==(65, 6891)]
    gen, dis = train(train_data[:9])

    sample_data = np.random(0, 1, 1024)
    out = gen(sample_data)
    time_out = librosa.istft(out.detach().numpy())
    sr = 22050
    librosa.output.write_wav("out.wav", time_out, sr)
