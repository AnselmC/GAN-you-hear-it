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
import librosa
import numpy as np
# own
from gann import ConvolutionalGenerator, LinearGenerator, Discriminator
from dataset import AudioSnippetDataset
from helpers import Progress, CustomWriter, visualize_sample, convert_sample_to_time_signal

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


def train(generator_type, data_loader, epochs, entropy_size, models, lrs,
          reg_strength, visual):
    timestamp = datetime.datetime.now().strftime(format="%d-%m-%Y-%H%M%S")
    if visual:
        logs_path = "logs/"
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)
        logging.basicConfig(filename=os.path.join(logs_path,
                                                  timestamp + ".log"),
                            level=logging.DEBUG)
        progress = Progress(epochs, len(data_loader), True)
        progress.init_print()
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger("Training")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logger.debug("Initializing training...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = data_loader.batch_size
    # TODO: train data in namespace, but should be clearly defined
    shape = train_data[0].shape
    sample_h, sample_w = shape[1], shape[2]

    discriminator = Discriminator(device,
                                  input_h=sample_h,
                                  input_w=sample_w,
                                  model=models[0] if models else None,
                                  lr=lrs[0])

    writer = CustomWriter(
        os.path.join("runs", str(epochs), str(batch_size),
                     str(sample_h) + "_" + str(sample_w), str(timestamp)),
        comment="conv" if generator_type == "conv" else "linear")
    if generator_type == "conv":
        generator = ConvolutionalGenerator(device,
                                           output_h=sample_h,
                                           output_w=sample_w,
                                           model=models[1] if models else None,
                                           entropy_size=entropy_size,
                                           lr=lrs[1])
        writer.add_graph(generator,
                         torch.randn(1, entropy_size, 1, 1).to(device), False)
    else:
        generator = LinearGenerator(device,
                                    output_h=sample_h,
                                    output_w=sample_w,
                                    model=models[1] if models else None,
                                    entropy_size=entropy_size,
                                    lr=lrs[1])
        writer.add_graph(generator,
                         torch.randn(1, entropy_size).to(device), False)

    writer.add_graph(discriminator,
                     generator.generate_data(1, device, train=False), False)

    L1_lambda = reg_strength

    def dim4d(a, b, c, d):
        return a * b * c * d

    losses = []
    fake_losses = []
    gen_losses = []
    logger.debug("Init done")
    logger.debug("Starting training with {} samples and {} epochs".format(
        len(train_data), epochs))
    for epoch in range(epochs):
        #L1_lambda /= 2
        if visual:
            progress.update_epoch()
        logger.debug("Epoch: {} of {}".format(epoch, epochs))

        running_loss = 0.0
        running_fake_loss = 0.0
        logger.debug("Training discriminator with {} batches".format(
            len(data_loader)))
        discriminator.train()
        generator.eval()
        for step, data in enumerate(data_loader):
            data = data.to(device)
            logger.debug("Batch: {}/{}".format(step, len(data_loader)))
            if visual:
                visualize_sample(data.cpu(), plot=True)
            discriminator.optim.zero_grad()
            out = discriminator(data)
            # make labels noisy (real: 0.7-1.2)
            label = Variable(0.7 + 0.3 * torch.rand(len(out), 1)).to(device)
            #acc = len(out[out>=0.5])/len(out)
            loss = discriminator.loss(out, label).to(device)
            loss.backward()
            fake_data = generator.generate_data(batch_size,
                                                device,
                                                train=False)
            if data_loader.dataset.transform:
                fake_data = data_loader.dataset.transform(
                    fake_data)  # non-generated data is transformed
            out = discriminator(fake_data)
            if visual:
                visualize_sample(fake_data.cpu(), plot=True)
            #fake_acc = len(out[out<0.5])/len(out)
            # make labels noisy (fake: 0-0.3)
            label = Variable(0.3 * torch.rand(len(out), 1)).to(device)
            fake_loss = discriminator.loss(out, label).to(device)
            fake_loss.backward()
            if visual:
                progress.update_batch(loss.item(), fake_loss.item())
            discriminator.optim.step()
            logger.debug("Loss: {}".format(loss.item()))
            #logger.debug("Acc: {}".format(acc))
            logger.debug("Fake Loss: {}".format(fake_loss.item()))
            #logger.debug("Fake acc: {}".format(acc))
            running_loss += loss.item()
            writer.write_dis_loss(loss.item(), epoch * len(data_loader) + step)
            running_fake_loss += fake_loss.item()
            writer.write_dis_fake_loss(fake_loss.item(),
                                       epoch * len(data_loader) + step)

        losses.append(running_loss / len(data_loader))
        fake_losses.append(running_fake_loss / len(data_loader))
        logger.debug("Running loss: {}".format(losses[-1]))
        logger.debug("Running fake loss: {}".format(fake_losses[-1]))

        running_gen_loss = 0.0
        logger.debug("Training generator...")
        discriminator.eval()
        generator.train()
        if visual:
            progress.switch_to_generator()
        for step in range(len(data_loader)):
            generator.optim.zero_grad()
            logger.debug("Batch: {}/{}".format(step, len(data_loader)))
            fake_data = generator.generate_data(batch_size, device, train=True)
            if data_loader.dataset.transform:
                fake_data_transformed = data_loader.dataset.transform(
                    fake_data)  # non-generated data is transformed
            if (epoch * len(data_loader) + step % 100 == 0):
                logger.debug("Writing generated sample...")
                writer.write_image(
                    visualize_sample(fake_data_transformed.cpu(), plot=visual),
                    epoch * len(data_loader) + step)
                writer.write_audio(fake_data_transformed.cpu()[0],
                                   epoch * len(data_loader) + step)
            out = discriminator(fake_data_transformed)
            #acc = len(out[out>=0.5])/len(out)
            # TODO: norm calculation is wrong
            reg_loss = torch.norm(fake_data, p=1) / dim4d(*fake_data.shape)
            writer.write_gen_reg_loss(L1_lambda * reg_loss.item(),
                                      epoch * len(data_loader) + step)
            logger.debug("Reg loss: {}".format(L1_lambda * reg_loss))
            # flip labels when training generator
            gen_loss = generator.loss(
                1-out,
                Variable(torch.ones([batch_size, 0])).to(device))
            gen_loss += L1_lambda * reg_loss
            gen_loss.backward()
            generator.optim.step()
            writer.write_gen_loss(gen_loss.item(),
                                  epoch * len(data_loader) + step)
            if visual:
                progress.update_batch(gen_loss.item())
            logger.debug("Gen loss: {}".format(gen_loss.item()))
            #logger.debug("Gen Acc: {}".format(acc))
            running_gen_loss += gen_loss.item()
        gen_losses.append(running_gen_loss / len(data_loader))
        logger.debug("Running generator loss: {}".format(gen_losses[-1]))

    writer.close()
    return generator, discriminator


if __name__ == "__main__":
    print(__description__)
    parser = argparse.ArgumentParser("Trains a GANN on audio signals")
    parser.add_argument(
        "--input_data",
        dest="input_data",
        required=True,
        help=
        "The npz archive with training data created with the `preprocessing.py` script"
    )
    parser.add_argument(
        "--subset_size",
        dest="subset_size",
        type=int,
        help=
        "The subset size to use from the input folder. Entire folder is used if not given or subset size is larger than folder size"
    )
    parser.add_argument("--epochs",
                        dest="epochs",
                        type=int,
                        help="The number of epochs to train for (default=10)",
                        default=10)
    parser.add_argument(
        "--reg_strength",
        dest="reg_strength",
        type=float,
        help=
        "L1 regularization strength for output of generator (enforces sparseness). Default: 10",
        default=10.)
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        help="The batch size used during training (default is 4)",
        default=4)
    parser.add_argument(
        "--entropy_size",
        dest="entropy_size",
        type=int,
        help=
        "The size of the entropy vector used as input for the generator (default is 10)",
        default=10)
    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        nargs=2,
        help=
        "Learning rates to use for discriminator and generator, respectively (defaults are 0.0002 and 0.0001)",
        default=[0.0002, 0.0001])
    parser.add_argument(
        "--models",
        dest="models",
        type=str,
        nargs=2,
        help=
        "Pretrained models to use for further training; Discriminator, then Generator (default None)"
    )
    parser.add_argument(
        "--generator",
        dest="gen",
        type=str,
        default="conv",
        choices={"conv", "linear"},
        help=
        "Which generator to use. Either \"linear\" or \"conv\" (default: \"conv\""
    )
    parser.add_argument(
        "--visual",
        dest="visual",
        action="store_true",
        help=
        "Whether to show visual representations of the training samples and generated results as images"
    )
    args = parser.parse_args()

    train_data = AudioSnippetDataset(args.input_data,
                                     subset_size=args.subset_size)
    data_loader = DataLoader(train_data,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=4)
    gen, dis = train(generator_type=args.gen,
                     data_loader=data_loader,
                     entropy_size=args.entropy_size,
                     epochs=args.epochs,
                     models=args.models,
                     lrs=args.lr,
                     reg_strength=args.reg_strength,
                     visual=args.visual)

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    wav_dir = os.path.join(results_dir, "audio")
    if not os.path.exists(wav_dir):
        os.mkdir(wav_dir)
    model_dir = os.path.join(results_dir, "models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    fn_prefix = str(args.epochs) + "_" + str(args.batch_size) + \
        "_" + str(args.entropy_size) + "_" + str(len(train_data))
    fn_wav = os.path.join(wav_dir, fn_prefix + ".wav")
    fn_gen_model = os.path.join(model_dir, fn_prefix + "_gen.model")
    fn_dis_model = os.path.join(model_dir, fn_prefix + "_dis.model")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generated = gen.generate_data(1, device)
    time_out = convert_sample_to_time_signal(generated)
    sr = 22050
    librosa.output.write_wav(fn_wav, time_out, sr)
    torch.save(gen.state_dict(), fn_gen_model)
    torch.save(dis.state_dict(), fn_dis_model)
