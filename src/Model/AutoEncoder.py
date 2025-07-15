from itertools import chain
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Encoder(nn.Module):
    """
    A class that represents an encoder module for a AutoEncoder network.
    """
    def __init__(self, input_dim=40, hidden_neus=27, latent_dim=7):
        """
        Initialize the Encoder class.

        Parameters:
        - input_dim (int): The dimensionality of the input data (default: 40).
        - hidden_neurons (int): The number of neurons in the hidden layer (default: 27).
        - latent_dim (int): The dimensionality of the latent space (default: 7).

        Returns:
            None
        """
        
        super(Encoder, self).__init__()
        encoder_network = []
        encoder_network.append(nn.Linear(input_dim, hidden_neus, bias=True))
        encoder_network.append(nn.ReLU())
        encoder_network.append(nn.Linear(hidden_neus, latent_dim, bias=True))
        # encoder_network.append(nn.Identity())
        self.encoder_network = nn.Sequential(*encoder_network)
        self.init_params()

    def init_params(self):
        """
        Initialize the parameters of the neural network using Xavier initialization.

        Returns:
            None
        """
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                bound = 1 / np.sqrt(layer.in_features)
                layer.weight.data.uniform_(-bound, bound)
                layer.bias.data.zero_()

    def forward(self, inputs):
        """
        Pass the inputs through the encoder network and return the output.

        Parameters:
        - inputs (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the encoder network.
        """

        return self.encoder_network(inputs)

class Decoder(nn.Module):
    """
    A decoder module that takes a latent vector as input and produces an output vector.
    """
    
    def __init__(self, latent_dim=7, hidden_neus=27, output_dim=40):
        """
        Initialize the Decoder class with the specified dimensions for 
        the latent space, hidden neurons, and output space.

        Parameters:
        - latent_dim (int): The dimension of the latent space.
        - hidden_neurons (int): The number of hidden neurons in the decoder network.
        - output_dim (int): The dimension of the output space.

        Returns:
            None
        """
        
        super(Decoder, self).__init__()
        decoder_network = []
        decoder_network.append(nn.Linear(latent_dim, hidden_neus, bias=True))
        decoder_network.append(nn.ReLU())
        decoder_network.append(nn.Linear(hidden_neus, output_dim, bias=True))
        # decoder_network.append(nn.Tanh())
        self.decoder_network = nn.Sequential(*decoder_network)
        self.init_params()

    def init_params(self):
        """
        Initialize the parameters of the neural network using Xavier initialization.

        Returns:
            None
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                bound = 1 / np.sqrt(layer.in_features)
                layer.weight.data.uniform_(-bound, bound)
                layer.bias.data.zero_()

    def forward(self, latent):
        return self.decoder_network(latent)


class Autoencoder(nn.Module):
    """
    Autoencoder class
    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_dim=40, output_dim=40, 
                hidden_neus=27, latent_dim=7):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_neus, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_neus, output_dim)

    def paramaeters(self):
        return chain(self.encoder.parameters(), self.decoder.parameters())

    def recon_loss(self, input, output):
        """
        Calculate the shrink loss.

        Parameters:
            - self: The instance of the class.
            - input_vector (torch.tensor): Description of the input vector.
            - output_vector (torch.tensor): Description of the output vector.
            - latent_vector (torch.tensor): Description of the latent vector.

        Returns:
            float: Description of the shrink loss.
        """
        
        batch_loss = nn.MSELoss(reduction='mean')(input, output)
        return batch_loss
    
    def forward(self, input):
        latent = self.encoder(input)
        output = self.decoder(latent)
        loss = self.recon_loss(input, output)
        return latent, output, loss


    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()
