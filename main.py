import argparse
import os
import torch
import torch.nn as nn
import pickle
from data_load import get_ori_data, scale_data
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from trainer import StepByStep
import pandas as pd
from models.TCNGAN import TCNGAN


def initialize_checkpoint(checkpoints_directory_name, epochs, model,  generator_optimizer, discriminator_optimizer):
    last_checkpoint_path = \
        sorted(os.listdir(checkpoints_directory_name),
               key=lambda fileName: int(fileName.split('.')[0].split('_')[1]),
               reverse=True)[0]
    checkpoint = torch.load(f'{checkpoints_directory_name}{last_checkpoint_path}')
    checkpoint_epoch = checkpoint['epoch']
    assert epochs > checkpoint_epoch, f'There is already an experiment with the same parameterisation and trained up to {checkpoint_epoch} epochs, skipping as it is a waste of time.'
    print(
        f'A previous experiment with the same parameterisation found and trained up to {checkpoint_epoch} epochs. Resuming the training for the remaining {epochs - checkpoint_epoch} epochs.')
    model.load_state_dict(checkpoint['model_state_dict'])
    generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
    initial_epoch = checkpoint_epoch
    initial_generator_losses = checkpoint['generator_loss']
    initial_discriminator_fake_losses = checkpoint['discriminator_fake_loss']
    initial_discriminator_real_losses = checkpoint['discriminator_real_loss']
    return initial_epoch, initial_generator_losses, initial_discriminator_fake_losses, initial_discriminator_real_losses


def recover_checkpoint_data(experiment_root_directory_name):
    with open(f'{experiment_root_directory_name}/scaler.pickle', "rb") as fb:
        scaler = pickle.load(fb)
    with open(f'{experiment_root_directory_name}/x_train_tensor.pickle', "rb") as fb:
        x_train_tensor = pickle.load(fb)
    with open(f'{experiment_root_directory_name}/y_train_tensor.pickle', "rb") as fb:
        y_train_tensor = pickle.load(fb)
    scaled_x_train, scaler, _ = scale_data(x_train_tensor, scaler=scaler)
    scaled_y_train, _, _ = scale_data(y_train_tensor, scaler=scaler)
    scaled_x_train_tensor = torch.as_tensor(scaled_x_train)
    scaled_y_train_tensor = torch.as_tensor(scaled_y_train)
    return scaled_x_train_tensor, scaled_y_train_tensor


def recover_ori_data(experiment_root_directory_name, seq_len, scaling_method, trace):
    x, y = get_ori_data(sequence_length=seq_len, stride=1, shuffle=True, seed=13, trace=trace)
    x_train_tensor = torch.as_tensor(x)
    y_train_tensor = torch.as_tensor(y)
    scaled_x_train, scaler, _ = scale_data(x_train_tensor, scaling_method=scaling_method)
    scaled_x_train_tensor = torch.as_tensor(scaled_x_train)
    with open(f"{experiment_root_directory_name}/scaler.pickle", "wb") as fb:
        pickle.dump(scaler, fb)
    with open(f"{experiment_root_directory_name}/x_train_tensor.pickle", "wb") as fb:
        pickle.dump(x_train_tensor, fb)
    with open(f"{experiment_root_directory_name}/y_train_tensor.pickle", "wb") as fb:
        pickle.dump(y_train_tensor, fb)
    return scaled_x_train_tensor, y_train_tensor


def main(args):
    args.device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    if args.device == "cuda":
        torch.cuda.empty_cache()
    print(args)
    seq_len = args.seq_len
    batch_size = args.batch_size
    generator_kernel_size = args.generator_kernel_size
    generator_channels = args.generator_channels
    discriminator_kernel_size = args.discriminator_kernel_size
    discriminator_channels = args.discriminator_channels
    lr = args.lr
    epochs = args.epochs
    trace = args.trace
    dropout = args.dropout
    scaling_method = args.scaling_method
    n_clip = args.n_clip
    params = vars(args)

    experiment_root_directory_name = f'experiments/tcngan_trace-{trace}_gen_kernel-{generator_kernel_size}_gen_channel-{"-".join([str(num) for num in generator_channels])}_dis_kernel-{discriminator_kernel_size}_dis_channel_{"-".join([str(num) for num in discriminator_channels])}_dropout-{dropout}_lr-{lr}_scaling-{scaling_method}/'
    tensorboard_model = f'tcngan_trace-{trace}_gen_kernel-{generator_kernel_size}_gen_channel-{"-".join([str(num) for num in generator_channels])}_dis_kernel-{discriminator_kernel_size}_dis_channel_{"-".join([str(num) for num in discriminator_channels])}_dropout-{dropout}_lr-{lr}_scaling-{scaling_method}'
    checkpoints_directory_name = f'{experiment_root_directory_name}checkpoints/'
    checkpoint_available = os.path.exists(checkpoints_directory_name) and len(
        os.listdir(checkpoints_directory_name)) > 0
    data_available = os.path.exists(f'{experiment_root_directory_name}/scaler.pickle') and os.path.exists(
        f'{experiment_root_directory_name}/x_train_tensor.pickle') and os.path.exists(
        f'{experiment_root_directory_name}/y_train_tensor.pickle')
    if checkpoint_available and data_available:
        scaled_x_train_tensor, scaled_y_train_tensor = recover_checkpoint_data(experiment_root_directory_name=experiment_root_directory_name)
    else:
        os.makedirs(experiment_root_directory_name, exist_ok=True)
        with open(experiment_root_directory_name + "parameters.txt", "w") as parameters_text_file:
            parameters_text_file.write(repr(args))
        scaled_x_train_tensor, scaled_y_train_tensor = recover_ori_data(experiment_root_directory_name=experiment_root_directory_name, seq_len=args.seq_len, scaling_method=scaling_method, trace=trace)

    train_data = TensorDataset(scaled_x_train_tensor.float(), scaled_y_train_tensor.float())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    discriminator_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    torch.manual_seed(43)

    n_features = scaled_x_train_tensor.shape[2]  # Batch first
    model = TCNGAN(num_features=n_features, seq_len=seq_len, batch_size=batch_size, generator_channels=generator_channels, discriminator_channels=discriminator_channels, generator_kernel_size=generator_kernel_size, discriminator_kernel_size=discriminator_kernel_size, dropout=dropout)
    model.to(args.device)
    loss = nn.BCEWithLogitsLoss()
    generator_optimizer = optim.Adam(model.generator.parameters(), lr=lr)
    discriminator_optimizer = optim.Adam(model.discriminator.parameters(), lr=lr)

    initial_epoch = 0
    initial_generator_losses = []
    initial_discriminator_fake_losses = []
    initial_discriminator_real_losses = []

    # Check if there is a folder with that parameterisation. If so, check if there is a checkpoints folder, and, if so, load the last checkpoint and check that the number of epochs requested is greater than the checkpoint epoch
    checkpoint_available = os.path.exists(checkpoints_directory_name) and len(
        os.listdir(checkpoints_directory_name)) > 0
    if checkpoint_available == True:
        # format: epoch_X
        initial_epoch, initial_generator_losses, initial_discriminator_fake_losses, initial_discriminator_real_losses = initialize_checkpoint(
            checkpoints_directory_name=checkpoints_directory_name, epochs=epochs, model=model, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer)

    trainer = StepByStep(model=model, loss_fn=loss, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer,
                         save_checkpoints=True, checkpoints_directory=checkpoints_directory_name, checkpoint_context=params, initial_epoch=initial_epoch,
                         initial_generator_losses=initial_generator_losses, initial_discriminator_fake_losses=initial_discriminator_fake_losses,
                         initial_discriminator_real_losses=initial_discriminator_real_losses, device=args.device)
    trainer.set_loaders(train_loader, discriminator_loader)
    trainer.set_tensorboard(tensorboard_model, folder='experiments/tensorboards')
    trainer.train(epochs, n_clip=n_clip)

    metrics_text_file = open(experiment_root_directory_name + "metrics.txt", "w")
    metrics_text_file.write('no hay métricas')
    losses = pd.DataFrame({'generator_losses': trainer.generator_losses, 'discriminator_fake_losses': trainer.discriminator_fake_losses, 'discriminator_real_losses': trainer.discriminator_real_losses})
    losses.to_csv(experiment_root_directory_name + "losses.csv")


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trace',
        choices=['azure_v2', 'google2019', 'alibaba2018'],
        default='azure_v2',
        type=str)
    parser.add_argument(
        '--epochs',
        default=1,
        type=int)
    parser.add_argument(
        '--seq_len',
        default=10,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int)
    parser.add_argument(
        '--generator_kernel_size',
        default=7,
        type=int)
    parser.add_argument(
        '--generator_channels',
        default=[25, 25, 25],
        # this must have as many elements as num_layers-1, as the last one will be added to match the number of input features. In this example, the number of layers will be 4
        type=lambda string_list: [int(num_channel_str) for num_channel_str in string_list.split(',')])
    parser.add_argument(
        '--discriminator_kernel_size',
        default=3,
        type=int)
    parser.add_argument(
        '--discriminator_channels',
        default=[16, 32, 64],  # same length as the number of layers of the discriminator
        type=lambda string_list: [int(num_channel_str) for num_channel_str in string_list.split(',')])
    parser.add_argument(
        '--lr',
        default=0.001,
        type=float)
    parser.add_argument(
        '--n_clip',
        default=5,
        type=int)
    parser.add_argument(
        '--dropout',
        default=0,
        type=float)
    parser.add_argument(
        '--scaling_method',
        default='standard',
        type=str)
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cpu',
        type=str)
    args = parser.parse_args()
    main(args)
