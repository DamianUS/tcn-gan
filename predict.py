from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import os
import pickle
from helpers import index_splitter
from data_load import get_ori_data, scale_data
from tqdm import tqdm, trange
from trainer import StepByStep
import numpy as np
import argparse
from distutils import util
import torch.nn as nn
from models.TCNGAN import TCNGAN

def prepare_model(n_features, checkpoint):
    print(args)
    checkpoint_context = checkpoint['model_params']
    seq_len = checkpoint_context["seq_len"]
    lr = checkpoint_context["lr"]
    dropout = checkpoint_context["dropout"]
    batch_size = checkpoint_context["batch_size"]
    generator_channels = checkpoint_context["generator_channels"]
    discriminator_channels = checkpoint_context["discriminator_channels"]
    generator_kernel_size = checkpoint_context["generator_kernel_size"]
    discriminator_kernel_size = checkpoint_context["discriminator_kernel_size"]
    model = TCNGAN(num_features=n_features, seq_len=seq_len, batch_size=batch_size,
                   generator_channels=generator_channels[:-1], discriminator_channels=discriminator_channels,
                   generator_kernel_size=generator_kernel_size, discriminator_kernel_size=discriminator_kernel_size,
                   dropout=dropout)
    model.to(args.device)
    generator_optimizer = optim.Adam(model.generator.parameters(), lr=lr)
    discriminator_optimizer = optim.Adam(model.discriminator.parameters(), lr=lr)
    model.load_state_dict(checkpoint['model_state_dict'])
    generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
    return model, generator_optimizer, discriminator_optimizer, checkpoint

def export_checkpoint(experiment_dir, checkpoint_pth_file, args):
    checkpoint = torch.load(checkpoint_pth_file)
    checkpoint_context = checkpoint['model_params']
    seq_len = checkpoint_context["seq_len"]
    batch_size = checkpoint_context["batch_size"]
    experiment_root_directory_name = args.experiment_directory_path
    scaler = None
    scaling_method = 'standard'
    data_available = os.path.exists(f'{experiment_root_directory_name}/scaler.pickle') and os.path.exists(
        f'{experiment_root_directory_name}/x_train_tensor.pickle') and os.path.exists(
        f'{experiment_root_directory_name}/y_train_tensor.pickle')
    if data_available:
        with open(f'{experiment_root_directory_name}/scaler.pickle', "rb") as fb:
            scaler = pickle.load(fb)
        with open(f'{experiment_root_directory_name}/x_train_tensor.pickle', "rb") as fb:
            x_train_tensor = pickle.load(fb)
        with open(f'{experiment_root_directory_name}/y_train_tensor.pickle', "rb") as fb:
            y_train_tensor = pickle.load(fb)
    else:
        scaling_method = checkpoint_context["scaling_method"] if "scaling_method" in checkpoint_context else 'standard'
        ori_data_filename = checkpoint_context["ori_data_filename"] if "ori_data_filename" in checkpoint_context else 'azure_v2'
        trace = checkpoint_context[
            "trace"] if "trace" in checkpoint_context else None
        x, y = get_ori_data(sequence_length=seq_len, stride=1, shuffle=True, seed=13, trace=trace)
        x_train_tensor = torch.as_tensor(x)
        y_train_tensor = torch.as_tensor(y)
    if scaler is not None:
        scaled_x_train, _, _ = scale_data(x_train_tensor, scaler=scaler)
    else:
        scaled_x_train, scaler, _ = scale_data(x_train_tensor, scaling_method=scaling_method)
    scaled_x_train_tensor = torch.as_tensor(scaled_x_train)
    model, generator_optimizer, discriminator_optimizer, checkpoint = prepare_model(scaled_x_train_tensor.shape[2], checkpoint)
    epoch = checkpoint['epoch']
    experiment_root_directory_name = f'{experiment_dir}/epoch_{epoch}/'
    torch.manual_seed(43)
    sbs_transf = StepByStep(model, None, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, device=args.device)
    generated_data_directory_name = experiment_root_directory_name + "generated_data/"
    os.makedirs(generated_data_directory_name, exist_ok=True)
    n_samples_export = args.n_samples_export
    for i in trange(n_samples_export, leave=False, colour='green'):
        predicted_sequence = sbs_transf.predict(scaled_x_train_tensor[0,:,:])[0]
        rescaled_sequence = np.reshape(scaler.inverse_transform(predicted_sequence.reshape(-1, 1)),
                                       predicted_sequence.shape)
        np.savetxt(f'{generated_data_directory_name}/sample_{i}.csv', rescaled_sequence, delimiter=",")

def main(args):
    print(args)
    experiment_directories = []
    if args.recursive == True:
        root_dir = args.experiment_directory_path
        experiment_directories = []
        for subdir, dirs, files in os.walk(root_dir):
            if 'checkpoints' in dirs:
                experiment_directories.append(subdir)
    else:
        experiment_directories.append(args.experiment_directory_path)

    progress_bar = tqdm(experiment_directories, colour='red')
    for experiment_dir in progress_bar:
        progress_bar.set_description(f'Creating samples for {experiment_dir}')
        epoch = args.epoch
        if epoch == -1:
            checkpoints_dir = f'{experiment_dir}/checkpoints/'
            assert os.path.exists(checkpoints_dir) and len(os.listdir(checkpoints_dir)) > 0, f'{experiment_dir}checkpoints/ does not exist or is empty'
            checkpoint_paths = [f'{checkpoints_dir}{checkpoint_name}' for checkpoint_name in sorted(os.listdir(checkpoints_dir), key=lambda fileName: int(fileName.split('.')[0].split('_')[1]), reverse=True)]
        else:
            checkpoint_pth_file = f'{experiment_dir}/checkpoints/epoch_{epoch}.pth'
            checkpoint_paths = [checkpoint_pth_file]
        for checkpoint_path in tqdm(checkpoint_paths, leave=False, colour='yellow'):
            assert os.path.exists(checkpoint_path), f'{checkpoint_path} does not exist'
            export_checkpoint(experiment_dir, checkpoint_path, args)

if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_directory_path',
        type=str)
    parser.add_argument(
        '--epoch',
        default=-1,
        type=int)
    parser.add_argument(
        '--n_samples_export',
        default=10,
        type=int)
    parser.add_argument(
        '--recursive',
        default=False,
        type=lambda x: bool(util.strtobool(str(x))))
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        type=str)
    args = parser.parse_args()
    main(args)