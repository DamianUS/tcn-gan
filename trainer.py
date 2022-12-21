import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
#from torchvision.transforms import Normalize
from torch.optim.lr_scheduler import LambdaLR
import os
from tqdm import tqdm, trange
import datetime

plt.style.use('fivethirtyeight')

def make_lr_fn(start_lr, end_lr, num_iter, step_mode='exp'):
    if step_mode == 'linear':
        factor = (end_lr / start_lr - 1) / num_iter
        def lr_fn(iteration):
            return 1 + iteration * factor
    else:
        factor = (np.log(end_lr) - np.log(start_lr)) / num_iter    
        def lr_fn(iteration):
            return np.exp(factor)**iteration    
    return lr_fn

class StepByStep(object):
    def __init__(self, model, loss_fn, generator_optimizer, discriminator_optimizer, save_checkpoints=False, checkpoints_directory='', checkpoint_context={}, initial_epoch=0, initial_generator_losses=[], initial_discriminator_fake_losses=[], initial_discriminator_real_losses=[], device=None):
        # Here we define the attributes of our class
        
        # We start by storing the arguments as attributes 
        # to use them later
        self.model = model
        self.loss_fn = loss_fn
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        # Let's send the model to the specified device right away
        self.model.to(self.device)

        # These attributes are defined here, but since they are
        # not informed at the moment of creation, we keep them None
        self.train_loader = None
        self.discriminator_loader = None
        self.writer = None
        self.scheduler = None
        self.is_batch_lr_scheduler = False
        self.clipping = None
        
        # These attributes are going to be computed internally
        self.generator_losses = initial_generator_losses
        self.discriminator_fake_losses = initial_discriminator_fake_losses
        self.discriminator_real_losses = initial_discriminator_real_losses
        self.learning_rates = []
        self.total_epochs = initial_epoch

        self.visualization = {}
        self.handles = {}

        # Creates the train_step function for our model, 
        # loss function and optimizer
        # Note: there are NO ARGS there! It makes use of the class
        # attributes directly
        self.train_step_fn = self._make_train_step_fn()
        # Defines wether the checkpoints should be saved and where
        self.save_checkpoints = save_checkpoints
        self.checkpoints_directory = checkpoints_directory
        self.checkpoint_context = checkpoint_context

    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader, discriminator_loader=None):
        # This method allows the user to define which train_loader (and val_loader, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        self.train_loader = train_loader
        self.discriminator_loader = discriminator_loader

    def set_tensorboard(self, name, folder='runs'):
        # This method allows the user to define a SummaryWriter to interface with TensorBoard
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def _make_train_step_fn(self):
        # This method does not need ARGS... it can refer to
        # the attributes: self.model, self.loss_fn and self.optimizer
        
        # Builds function that performs a step in the train loop
        def perform_train_step_fn(x, y):
            # Sets model to TRAIN mode
            self.model.train()

            # training generator
            x_hat = self.model(x, obj='generator')
            self.generator_optimizer.zero_grad()
            pred_fake = self.model(x_hat, obj='discriminator')
            loss_generator = self.loss_fn(pred_fake, torch.ones_like(pred_fake))
            loss_generator.backward()
            self.generator_optimizer.step()

            # training discriminator
            for i in range(self.n_clip):
                if i == 0:
                    x_discriminator, y_discriminator = x, y
                else:
                    discriminator_iterator = iter(self.discriminator_loader)
                    try:
                        x_discriminator, y_discriminator = next(discriminator_iterator)
                    except StopIteration:
                        discriminator_iterator = iter(self.discriminator_loader)
                        x_discriminator, y_discriminator = next(discriminator_iterator)
                self.discriminator_optimizer.zero_grad()
                x_hat = self.model(x, obj='generator')
                pred_real = self.model(x_discriminator.to(self.device), obj='discriminator')
                pred_fake = self.model(x_hat.detach(), obj='discriminator')
                loss_discriminator_real = self.loss_fn(pred_real, torch.ones_like(pred_real))
                loss_discriminator_fake = self.loss_fn(pred_fake, torch.zeros_like(pred_fake))
                loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) * 0.5
                loss_discriminator.backward()
                self.discriminator_optimizer.step()
            # Returns the loss
            return loss_generator.item(), loss_discriminator_fake.item(), loss_discriminator_real.item()
        # Returns the function that will be called inside the train loop
        return perform_train_step_fn
            
    def _mini_batch(self, validation=False):
        # The mini-batch can be used with both loaders
        # The argument `validation`defines which loader and 
        # corresponding step function is going to be used
        data_loader = self.train_loader
        step_fn = self.train_step_fn

        if data_loader is None:
            return None
            
        n_batches = len(data_loader)
        # Once the data loader and step function, this is the same
        # mini-batch loop we had before
        mini_batch_generator_losses = []
        mini_batch_discriminator_fake_losses = []
        mini_batch_discriminator_real_losses = []
        mini_batch_progress_bar = tqdm(data_loader, desc='MiniBatch', colour='yellow', leave=False)

        for i, (x_batch, y_batch) in enumerate(mini_batch_progress_bar):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            loss_generator, loss_discriminator_fake, loss_discriminator_real = step_fn(x_batch, y_batch)
            mini_batch_generator_losses.append(loss_generator)
            mini_batch_discriminator_fake_losses.append(loss_discriminator_fake)
            mini_batch_discriminator_real_losses.append(loss_discriminator_real)
            mini_batch_progress_bar.set_description(
                f'Minibatch losses: Generator: {loss_generator:.4f}. Discriminator fake: {loss_discriminator_fake:.4f}. Discriminator real: {loss_discriminator_real:.4f}'
            )

            if not validation:
                self._mini_batch_schedulers(i / n_batches)

        return np.mean(mini_batch_generator_losses), np.mean(mini_batch_discriminator_fake_losses), np.mean(mini_batch_discriminator_real_losses)

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass
        
    def train(self, n_epochs, seed=42, n_clip=5):
        # To ensure reproducibility of the training process
        self.set_seed(seed)
        self.n_clip = n_clip
        for epoch in tqdm(range(self.total_epochs, n_epochs)):
            # Keeps track of the numbers of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1

            # inner loop
            # Performs training using mini-batches
            loss_generator, loss_discriminator_fake, loss_discriminator_real = self._mini_batch(validation=False)
            self.generator_losses.append(loss_generator)
            self.discriminator_fake_losses.append(loss_discriminator_fake)
            self.discriminator_real_losses.append(loss_discriminator_real)
                        
            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'generator': loss_generator, 'discriminator_fake': loss_discriminator_fake, 'discriminator_real': loss_discriminator_real}

                # Records both losses for each epoch under the main tag "loss"
                self.writer.add_scalars(main_tag='losses',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)
            #print(f'Finished epoch {self.total_epochs}/{n_epochs}. Train loss: {loss}. Validation loss: {val_loss}.')
            if self.save_checkpoints == True:
                os.makedirs(self.checkpoints_directory, exist_ok=True)
                self.save_checkpoint(f'{self.checkpoints_directory}epoch_{self.total_epochs}.pth')
        if self.writer:
            # Closes the writer
            self.writer.close()


    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
                      'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
                      'generator_loss': self.generator_losses,
                      'discriminator_fake_loss': self.discriminator_fake_losses,
                      'discriminator_real_loss': self.discriminator_real_losses,
                      'model_params': self.checkpoint_context}
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint = torch.load(filename)
        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.total_epochs = checkpoint['epoch']
        self.generator_losses = checkpoint['generator_loss']
        self.discriminator_fake_losses = checkpoint['discriminator_fake_loss']
        self.discriminator_real_losses = checkpoint['discriminator_real_loss']
        self.model.train()  # always use TRAIN for resuming training

    def predict(self, x):
        # Set is to evaluation mode for predictions
        self.model.eval() 
        # Takes aNumpy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and uses model for prediction
        x_hat_tensor = self.model(x_tensor.to(self.device), obj='generator')
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        return x_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.generator_losses, label='Generator Loss', c='b')
        plt.plot(self.discriminator_fake_losses, label='Discriminator fake Loss', c='b')
        plt.plot(self.discriminator_real_losses, label='Discriminator real Loss', c='g')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def add_graph(self):
        # Fetches a single mini-batch so we can use add_graph
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    @staticmethod
    def _visualize_tensors(axs, x, y=None, yhat=None, layer_name='', title=None):
        # The number of images is the number of subplots in a row
        n_images = len(axs)
        # Gets max and min values for scaling the grayscale
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images])
        # For each image
        for j, image in enumerate(x[:n_images]):
            ax = axs[j]
            # Sets title, labels, and removes ticks
            if title is not None:
                ax.set_title('{} #{}'.format(title, j), fontsize=12)
            ax.set_ylabel(
                '{}\n{}x{}'.format(layer_name, *np.atleast_2d(image).shape), 
                rotation=0, labelpad=40
            )
            xlabel1 = '' if y is None else '\nLabel: {}'.format(y[j])
            xlabel2 = '' if yhat is None else '\nPredicted: {}'.format(yhat[j])
            xlabel = '{}{}'.format(xlabel1, xlabel2)
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Plots weight as an image
            ax.imshow(
                np.atleast_2d(image.squeeze()),
                cmap='gray', 
                vmin=minv, 
                vmax=maxv
            )
        return

    def visualize_filters(self, layer_name, **kwargs):
        try:
            # Gets the layer object from the model
            layer = self.model
            for name in layer_name.split('.'):
                layer = getattr(layer, name)
            # We are only looking at filters for 2D convolutions
            if isinstance(layer, nn.Conv2d):
                # Takes the weight information
                weights = layer.weight.data.cpu().numpy()
                # The weights have channels_out (filter), channels_in, H, W shape
                n_filters, n_channels, _, _ = weights.shape

                # Builds a figure
                size = (2 * n_channels + 2, 2 * n_filters)
                fig, axes = plt.subplots(n_filters, n_channels, figsize=size)
                axes = np.atleast_2d(axes).reshape(n_filters, n_channels)
                # For each channel_out (filter)
                for i in range(n_filters):    
                    StepByStep._visualize_tensors(
                        axes[i, :], 
                        weights[i], 
                        layer_name='Filter #{}'.format(i), 
                        title='Channel' if (i == 0) else None
                    )

                for ax in axes.flat:
                    ax.label_outer()

                fig.tight_layout()
                return fig
        except AttributeError:
            return
    
    def attach_hooks(self, layers_to_hook, hook_fn=None):
        # Clear any previous values
        self.visualization = {}
        # Creates the dictionary to map layer objects to their names
        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules[1:]}
        
        if hook_fn is None:
            # Hook function to be attached to the forward pass
            def hook_fn(layer, inputs, outputs):
                # Gets the layer name
                name = layer_names[layer]
                # Detaches outputs
                values = outputs.detach().cpu().numpy()
                # Since the hook function may be called multiple times
                # for example, if we make predictions for multiple mini-batches
                # it concatenates the results
                if self.visualization[name] is None:
                    self.visualization[name] = values
                else:
                    self.visualization[name] = np.concatenate([self.visualization[name], values])

        for name, layer in modules:
            # If the layer is in our list
            if name in layers_to_hook:
                # Initializes the corresponding key in the dictionary
                self.visualization[name] = None
                # Register the forward hook and keep the handle in another dict
                self.handles[name] = layer.register_forward_hook(hook_fn)

    def remove_hooks(self):
        # Loops through all hooks and removes them
        for handle in self.handles.values():
            handle.remove()
        # Clear the dict, as all hooks have been removed
        self.handles = {}

    def visualize_outputs(self, layers, n_images=10, y=None, yhat=None):
        layers = list(filter(lambda l: l in self.visualization.keys(), layers))
        shapes = [self.visualization[layer].shape for layer in layers]
        n_rows = [shape[1] if len(shape) == 4 else 1 for shape in shapes]
        total_rows = np.sum(n_rows)

        fig, axes = plt.subplots(total_rows, n_images, figsize=(1.5*n_images, 1.5*total_rows))
        axes = np.atleast_2d(axes).reshape(total_rows, n_images)

        # Loops through the layers, one layer per row of subplots
        row = 0
        for i, layer in enumerate(layers):
            start_row = row
            # Takes the produced feature maps for that layer
            output = self.visualization[layer]

            is_vector = len(output.shape) == 2

            for j in range(n_rows[i]):
                StepByStep._visualize_tensors(
                    axes[row, :],
                    output if is_vector else output[:, j].squeeze(),
                    y, 
                    yhat, 
                    layer_name=layers[i] if is_vector else '{}\nfil#{}'.format(layers[i], row-start_row),
                    title='Image' if (row == 0) else None
                )
                row += 1

        for ax in axes.flat:
            ax.label_outer()

        plt.tight_layout()
        return fig

    def correct(self, x, y, threshold=.5):
        self.model.eval()
        yhat = self.model(x.to(self.device))
        y = y.to(self.device)
        self.model.train()

        # We get the size of the batch and the number of classes 
        # (only 1, if it is binary)
        n_samples, n_dims = yhat.shape
        if n_dims > 1:        
            # In a multiclass classification, the biggest logit
            # always wins, so we don't bother getting probabilities

            # This is PyTorch's version of argmax, 
            # but it returns a tuple: (max value, index of max value)
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims += 1
            # In binary classification, we NEED to check if the
            # last layer is a sigmoid (and then it produces probs)
            if isinstance(self.model, nn.Sequential) and \
               isinstance(self.model[-1], nn.Sigmoid):
                predicted = (yhat > threshold).long()
            # or something else (logits), which we need to convert
            # using a sigmoid
            else:
                predicted = (torch.sigmoid(yhat) > threshold).long()

        # How many samples got classified correctly for each class
        result = []
        for c in range(n_dims):
            n_class = (y == c).sum().item()
            n_correct = (predicted[y == c] == c).sum().item()
            result.append((n_correct, n_class))
        return torch.tensor(result)
    
    @staticmethod
    def loader_apply(loader, func, reduce='sum'):
        results = [func(x, y) for i, (x, y) in enumerate(loader)]
        results = torch.stack(results, axis=0)

        if reduce == 'sum':
            results = results.sum(axis=0)
        elif reduce == 'mean':
            results = results.float().mean(axis=0)

        return results

    @staticmethod
    def statistics_per_channel(images, labels):
        # NCHW
        n_samples, n_channels, n_height, n_weight = images.size()
        # Flatten HW into a single dimension
        flatten_per_channel = images.reshape(n_samples, n_channels, -1)

        # Computes statistics of each image per channel
        # Average pixel value per channel 
        # (n_samples, n_channels)
        means = flatten_per_channel.mean(axis=2)
        # Standard deviation of pixel values per channel
        # (n_samples, n_channels)
        stds = flatten_per_channel.std(axis=2)

        # Adds up statistics of all images in a mini-batch
        # (1, n_channels)
        sum_means = means.sum(axis=0)
        sum_stds = stds.sum(axis=0)
        # Makes a tensor of shape (1, n_channels)
        # with the number of samples in the mini-batch
        n_samples = torch.tensor([n_samples]*n_channels).float()

        # Stack the three tensors on top of one another
        # (3, n_channels)
        return torch.stack([n_samples, sum_means, sum_stds], axis=0)

    # @staticmethod
    # def make_normalizer(loader):
    #     total_samples, total_means, total_stds = StepByStep.loader_apply(loader, StepByStep.statistics_per_channel)
    #     norm_mean = total_means / total_samples
    #     norm_std = total_stds / total_samples
    #     return Normalize(mean=norm_mean, std=norm_std)
                        
    def lr_range_test(self, data_loader, end_lr, num_iter=100, step_mode='exp', alpha=0.05, ax=None):
        # Since the test updates both model and optimizer we need to store
        # their initial states to restore them in the end
        previous_states = {'model': deepcopy(self.model.state_dict()), 
                           'optimizer': deepcopy(self.optimizer.state_dict())}
        # Retrieves the learning rate set in the optimizer
        start_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        # Builds a custom function and corresponding scheduler
        lr_fn = make_lr_fn(start_lr, end_lr, num_iter)
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)

        # Variables for tracking results and iterations
        tracking = {'loss': [], 'lr': []}
        iteration = 0

        # If there are more iterations than mini-batches in the data loader,
        # it will have to loop over it more than once
        while (iteration < num_iter):
            # That's the typical mini-batch inner loop
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # Step 1
                yhat = self.model(x_batch)
                # Step 2
                loss = self.loss_fn(yhat, y_batch)
                # Step 3
                loss.backward()

                # Here we keep track of the losses (smoothed)
                # and the learning rates
                tracking['lr'].append(scheduler.get_last_lr()[0])
                if iteration == 0:
                    tracking['loss'].append(loss.item())
                else:
                    prev_loss = tracking['loss'][-1]
                    smoothed_loss = alpha * loss.item() + (1-alpha) * prev_loss
                    tracking['loss'].append(smoothed_loss)

                iteration += 1
                # Number of iterations reached
                if iteration == num_iter:
                    break

                # Step 4
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

        # Restores the original states
        self.optimizer.load_state_dict(previous_states['optimizer'])
        self.model.load_state_dict(previous_states['model'])

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.get_figure()
        ax.plot(tracking['lr'], tracking['loss'])
        if step_mode == 'exp':
            ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        fig.tight_layout()
        return tracking, fig
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def capture_gradients(self, layers_to_hook):
        if not isinstance(layers_to_hook, list):
            layers_to_hook = [layers_to_hook]

        modules = list(self.model.named_modules())
        self._gradients = {}

        def make_log_fn(name, parm_id):
            def log_fn(grad):
                self._gradients[name][parm_id].append(grad.tolist())
                return
            return log_fn

        for name, layer in self.model.named_modules():
            if name in layers_to_hook:
                self._gradients.update({name: {}})
                for parm_id, p in layer.named_parameters():
                    if p.requires_grad:
                        self._gradients[name].update({parm_id: []})
                        log_fn = make_log_fn(name, parm_id)
                        self.handles[f'{name}.{parm_id}.grad'] = p.register_hook(log_fn)
        return

    def capture_parameters(self, layers_to_hook):
        if not isinstance(layers_to_hook, list):
            layers_to_hook = [layers_to_hook]

        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules}

        self._parameters = {}

        for name, layer in modules:
            if name in layers_to_hook:
                self._parameters.update({name: {}})
                for parm_id, p in layer.named_parameters():
                    self._parameters[name].update({parm_id: []})

        def fw_hook_fn(layer, inputs, outputs):
            name = layer_names[layer]
            for parm_id, parameter in layer.named_parameters():
                self._parameters[name][parm_id].append(parameter.tolist())

        self.attach_hooks(layers_to_hook, fw_hook_fn)
        return
    
    def set_lr_scheduler(self, scheduler):
        # Makes sure the scheduler in the argument is assigned to the
        # optimizer we're using in this class
        if scheduler.optimizer == self.optimizer:
            self.scheduler = scheduler
            if (isinstance(scheduler, optim.lr_scheduler.CyclicLR) or
                isinstance(scheduler, optim.lr_scheduler.OneCycleLR) or
                isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts)):
                self.is_batch_lr_scheduler = True
            else:
                self.is_batch_lr_scheduler = False

    def _epoch_schedulers(self, val_loss):
        if self.scheduler:
            if not self.is_batch_lr_scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                current_lr = list(map(lambda d: d['lr'], self.scheduler.optimizer.state_dict()['param_groups']))
                self.learning_rates.append(current_lr)
        
    def _mini_batch_schedulers(self, frac_epoch):
        if self.scheduler:
            if self.is_batch_lr_scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    self.scheduler.step(self.total_epochs + frac_epoch)
                else:
                    self.scheduler.step()

                current_lr = list(map(lambda d: d['lr'], self.scheduler.optimizer.state_dict()['param_groups']))
                self.learning_rates.append(current_lr)

    def set_clip_grad_value(self, clip_value):
        self.clipping = lambda: nn.utils.clip_grad_value_(self.model.parameters(), clip_value=clip_value)

    def set_clip_grad_norm(self, max_norm, norm_type=2):
        self.clipping = lambda: nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm, norm_type=norm_type)

    def set_clip_backprop(self, clip_value):
        if self.clipping is None:
            self.clipping = []
        for p in self.model.parameters():
            if p.requires_grad:
                func = lambda grad: torch.clamp(grad, -clip_value, clip_value)
                handle = p.register_hook(func)
                self.clipping.append(handle)

    def remove_clip(self):
        if isinstance(self.clipping, list):
            for handle in self.clipping:
                handle.remove()
        self.clipping = None
