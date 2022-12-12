import argparse
import os
import torch
import torch.nn as nn
import pickle
from data_load import get_ori_data, scale_data
from helpers import index_splitter
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from trainer import StepByStep
import pandas as pd
from models.Discriminator import Discriminator

def initialize_checkpoint(checkpoints_directory_name, epochs, model, optimizer):
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
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    initial_epoch = checkpoint_epoch
    initial_losses = checkpoint['loss']
    initial_val_losses = checkpoint['val_loss']
    return initial_epoch, initial_losses, initial_val_losses

def recover_checkpoint_data(experiment_root_directory_name):
    with open(f'{experiment_root_directory_name}/scaler.pickle', "rb") as fb:
        scaler = pickle.load(fb)
    with open(f'{experiment_root_directory_name}/x_train_tensor.pickle', "rb") as fb:
        x_train_tensor = pickle.load(fb)
    with open(f'{experiment_root_directory_name}/y_train_tensor.pickle', "rb") as fb:
        y_train_tensor = pickle.load(fb)
    with open(f'{experiment_root_directory_name}/x_val_tensor.pickle', "rb") as fb:
        x_val_tensor = pickle.load(fb)
    with open(f'{experiment_root_directory_name}/y_val_tensor.pickle', "rb") as fb:
        y_val_tensor = pickle.load(fb)
    scaled_x_train, scaler, _ = scale_data(x_train_tensor, scaler=scaler)
    scaled_y_train, _, _ = scale_data(y_train_tensor, scaler=scaler)
    scaled_x_val, _, _ = scale_data(x_val_tensor, scaler=scaler)
    scaled_y_val, _, _ = scale_data(y_val_tensor, scaler=scaler)
    scaled_x_train_tensor = torch.as_tensor(scaled_x_train)
    scaled_y_train_tensor = torch.as_tensor(scaled_y_train)
    scaled_x_val_tensor = torch.as_tensor(scaled_x_val)
    scaled_y_val_tensor = torch.as_tensor(scaled_y_val)
    return scaled_x_train_tensor, scaled_y_train_tensor, scaled_x_val_tensor, scaled_y_val_tensor


def recover_ori_data(experiment_root_directory_name, synthetic_data_filename, seq_len, scaling_method, trace):
    x, y = get_ori_data(sequence_length=seq_len, stride=1, shuffle=True, seed=13,
                        synthetic_data_filename=synthetic_data_filename, trace=trace)
    train_idx, val_idx = index_splitter(len(x), [75, 25])
    x_tensor = torch.as_tensor(x)
    y_tensor = torch.as_tensor(y)
    x_train_tensor = x_tensor[train_idx]
    x_val_tensor = x_tensor[val_idx]
    y_train_tensor = y_tensor[train_idx]
    y_val_tensor = y_tensor[val_idx]
    scaled_x_train, scaler, _ = scale_data(x_train_tensor, scaling_method=scaling_method)
    scaled_x_val, _, _ = scale_data(x_val_tensor, scaler=scaler)
    scaled_x_train_tensor = torch.as_tensor(scaled_x_train)
    scaled_x_val_tensor = torch.as_tensor(scaled_x_val)
    with open(f"{experiment_root_directory_name}/scaler.pickle", "wb") as fb:
        pickle.dump(scaler, fb)
    with open(f"{experiment_root_directory_name}/x_train_tensor.pickle", "wb") as fb:
        pickle.dump(x_train_tensor, fb)
    with open(f"{experiment_root_directory_name}/y_train_tensor.pickle", "wb") as fb:
        pickle.dump(y_train_tensor, fb)
    with open(f"{experiment_root_directory_name}/x_val_tensor.pickle", "wb") as fb:
        pickle.dump(x_val_tensor, fb)
    with open(f"{experiment_root_directory_name}/y_val_tensor.pickle", "wb") as fb:
        pickle.dump(y_val_tensor, fb)
    return scaled_x_train_tensor, y_train_tensor, scaled_x_val_tensor, y_val_tensor


def main(args):
    args.device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    if args.device == "cuda":
        torch.cuda.empty_cache()
    print(args)
    synthetic_data_filename = args.synthetic_data_filename
    batch_size = args.batch_size
    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    teacher_forcing = args.teacher_forcing
    lr = args.lr
    epochs = args.epochs
    trace = args.trace
    dropout = args.dropout
    if args.rnn_module == "RNN":
        rnn_layer_module = nn.RNN
    elif args.rnn_module == "GRU":
        rnn_layer_module = nn.GRU
    else:
        rnn_layer_module = nn.LSTM
    input_output_ratio = args.input_output_ratio
    scaling_method = args.scaling_method
    encoder_decoder_model = args.encoder_decoder_model
    normalization = args.normalization
    narrow_attn_heads = args.narrow_attn_heads

    params = vars(args)

    # if encoder_decoder_model == "EncoderDecoder":
    #     experiment_root_directory_name = f'experiments/model_{encoder_decoder_model}_{args.rnn_module}_{trace}_layers-{num_layers}_hidden-{hidden_dim}_dropout-{dropout}_norm-{normalization}_attn_{narrow_attn_heads}_lr-{lr}_batch-{batch_size}_seq-{seq_len}_scale-{scaling_method}/'
    #     tensorboard_model = f'model_{encoder_decoder_model}_{args.rnn_module}_{trace}_layers-{num_layers}_hidden-{hidden_dim}_dropout-{dropout}_norm-{normalization}_lr-{lr}_batch-{batch_size}_attn_{narrow_attn_heads}_seq-{seq_len}_scale-{scaling_method}'
    # elif encoder_decoder_model == 'TCN':
    #     experiment_root_directory_name = f'experiments/model_{encoder_decoder_model}_{trace}_layers-{num_layers}_channels-{args.num_channels}_kernel-{args.kernel_size}_dropout-{dropout}_lr-{lr}_batch-{batch_size}_seq-{seq_len}_scale-{scaling_method}/'
    #     tensorboard_model = f'model_{encoder_decoder_model}_layers-{num_layers}_channels-{args.num_channels}_kernel-{args.kernel_size}_dropout-{dropout}_lr-{lr}_batch-{batch_size}_seq-{seq_len}_scale-{scaling_method}'
    # elif encoder_decoder_model == "Transformer":
    #     experiment_root_directory_name = f'experiments/model_{encoder_decoder_model}_{trace}_layers-{num_layers}_hidden-{hidden_dim}_dropout-{dropout}_attn_{narrow_attn_heads}_lr-{lr}_batch-{batch_size}_seq-{seq_len}_scale-{scaling_method}/'
    #     tensorboard_model = f'model_{encoder_decoder_model}_{trace}_layers-{num_layers}_hidden-{hidden_dim}_dropout-{dropout}_attn_{narrow_attn_heads}_lr-{lr}_batch-{batch_size}_seq-{seq_len}_scale-{scaling_method}'
    experiment_root_directory_name = f'experiments/discriminator/'
    tensorboard_model = f'discriminator'
    checkpoints_directory_name = f'{experiment_root_directory_name}checkpoints/'
    checkpoint_available = os.path.exists(checkpoints_directory_name) and len(
        os.listdir(checkpoints_directory_name)) > 0
    data_available = os.path.exists(f'{experiment_root_directory_name}/scaler.pickle') and os.path.exists(
        f'{experiment_root_directory_name}/x_train_tensor.pickle') and os.path.exists(
        f'{experiment_root_directory_name}/y_train_tensor.pickle') and os.path.exists(
        f'{experiment_root_directory_name}/x_val_tensor.pickle') and os.path.exists(
        f'{experiment_root_directory_name}/y_val_tensor.pickle')
    if checkpoint_available and data_available:
        scaled_x_train_tensor, scaled_y_train_tensor, scaled_x_val_tensor, scaled_y_val_tensor = recover_checkpoint_data(
            experiment_root_directory_name=experiment_root_directory_name)
    else:
        os.makedirs(experiment_root_directory_name, exist_ok=True)
        parameters_text_file = open(experiment_root_directory_name + "parameters.txt", "w")
        parameters_text_file.write(repr(args))
        scaled_x_train_tensor, scaled_y_train_tensor, scaled_x_val_tensor, scaled_y_val_tensor = recover_ori_data(
            experiment_root_directory_name=experiment_root_directory_name, synthetic_data_filename=synthetic_data_filename,
            seq_len=args.seq_len, scaling_method=scaling_method, trace=trace)

    train_data = TensorDataset(scaled_x_train_tensor.float(), scaled_y_train_tensor.float())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = TensorDataset(scaled_x_val_tensor.float(), scaled_y_val_tensor.float())
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    torch.manual_seed(43)
    torch.manual_seed(43)

    n_features = scaled_x_train_tensor.shape[2]  # Batch first
    channels = [16, 32, 64]
    model = Discriminator(num_inputs=n_features, num_channels = channels, kernel_size=3, stride=1, dilation=1, padding=1, seq_len=args.seq_len)
    model.to(args.device)
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    initial_epoch = 0
    initial_losses = []
    initial_val_losses = []

    # Check if there is a folder with that parameterisation. If so, check if there is a checkpoints folder, and, if so, load the last checkpoint and check that the number of epochs requested is greater than the checkpoint epoch
    checkpoint_available = os.path.exists(checkpoints_directory_name) and len(
        os.listdir(checkpoints_directory_name)) > 0
    if checkpoint_available == True:
        # format: epoch_X
        initial_epoch, initial_losses, initial_val_losses = initialize_checkpoint(checkpoints_directory_name=checkpoints_directory_name, epochs=epochs, model=model, optimizer=optimizer)

    trainer = StepByStep(model, loss, optimizer, save_checkpoints=True,
                                  checkpoints_directory=checkpoints_directory_name,
                                  checkpoint_context=params, initial_epoch=initial_epoch, initial_losses=initial_losses,
                                  intial_val_losses=initial_val_losses, device=args.device)
    trainer.set_loaders(train_loader, test_loader)
    trainer.set_tensorboard(tensorboard_model, folder='experiments/tensorboards')
    trainer.train(epochs)

    metrics_text_file = open(experiment_root_directory_name + "metrics.txt", "w")
    metrics_text_file.write('no hay métricas')
    losses = pd.DataFrame({'train_loss': trainer.losses, 'val_loss': trainer.val_losses})
    losses.to_csv(experiment_root_directory_name + "losses.csv")


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--synthetic_data_filename',
        default=None,
        type=str)
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
        '--hidden_dim',
        default=2,
        type=int)
    parser.add_argument(
        '--dropout',
        default=0,
        type=float)
    parser.add_argument(
        '--teacher_forcing',
        default=0.5,
        type=float)
    parser.add_argument(
        '--wide_attn_heads',
        default=1,
        type=int)
    parser.add_argument(
        '--lr',
        default=0.001,
        type=float)
    parser.add_argument(
        '--ff_units',
        default=2,
        type=int)
    parser.add_argument(
        '--rnn_module',
        choices=['RNN', 'GRU', 'LSTM'],
        default='GRU',
        type=str)
    parser.add_argument(
        '--num_layers',
        default=1,
        type=int)
    parser.add_argument(
        '--input_output_ratio',
        default=0.5,
        type=float)
    parser.add_argument(
        '--scaling_method',
        default='standard',
        type=str)
    parser.add_argument(
        '--encoder_decoder_model',
        choices=['EncoderDecoder', 'TCN', 'Transformer'],
        default='EncoderDecoder',
        type=str)
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cpu',
        type=str)
    parser.add_argument(
        '--normalization',
        choices=['BatchNormalization', 'LayerNormalization'],
        default=None,
        type=str)
    parser.add_argument(
        '--narrow_attn_heads',
        default=0,
        type=int)
    parser.add_argument(
        '--kernel_size',
        default=7,
        type=int)
    parser.add_argument(
        '--num_channels',
        default=25,
        type=int)
    args = parser.parse_args()
    main(args)
