""" Implementation of autoencoders.
"""


import torch
from torch import nn
from collections import namedtuple
from pathlib import Path
from torch.utils.data import DataLoader


def load_optimizer(config, params):
    """ Loads Adam optimizer for autoencoders.
    
    Args:
        config: configuration of autoencoders.
        params: parameters for Adam optimizer.
    
    Returns:
        A torch.optim.Adam optimizer.
    """

    kwargs = dict(config.get("optim", {}))
    assert kwargs.pop("optimizer", "Adam") == "Adam"
    optim = torch.optim.Adam(params, **kwargs)
    return optim


def load_networks(config, **kwargs):
    """ Loads the network structure of autoencoders.
    
    Args:
        config: configuration of autoencoders.
        **kwargs: dict to load autoencoders.
    
    Returns:
        An instance of the required autoencoder.
    """

    kwargs = kwargs.copy()
    kwargs.update(dict(config.get("model", {})))
    name = kwargs.pop("name")

    if name == "scgen":
        model = AutoEncoder

    elif name == "cae":
        model = ConditionalAutoEncoder

    else:
        raise ValueError

    return model(**kwargs)


def load_autoencoder_model(config, restore=None, **kwargs):
    """ Loads an autoencoder.
    
    Args:
        config: configuration of autoencoders.
        restore: file path from which a previous state is restored. Default is None.
        **kwargs: dict to load autoencoders.
    
    Returns:
        1. An instance of the required autoencoder.
        2. A torch.optim.Adam optimizer.
    """

    # Loads network structure and optimizer.
    model = load_networks(config, **kwargs)
    optim = load_optimizer(config, model.parameters())

    # Restores a previous state if required.
    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optim_state"])
        if config.model.name == "scgen" and "code_means" in ckpt:
            model.code_means = ckpt["code_means"]

    return model, optim


def dnn(
    dinput,
    doutput,
    hidden_units=(16, 16),
    activation="ReLU",
    dropout=0.0,
    batch_norm=False,
    net_fn=nn.Sequential,
    **kwargs
):
    """ Loads a deep neural network.
    
    Args:
        dinput: int, input dimension.
        doutput: int, output dimension.
        hidden_units: tuple, number of units on each hidden layer. Default is (16, 16).
        activation: activation function. Default is "ReLU".
        dropout: dropout rate. Default is 0.0.
        batch_norm: batch normalization. Default is False.
        **kwargs: additional parameters.
    
    Returns:
        A torch.nn DNN.
    """
    
    # Lists number of units in the hidden layers.
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]
    hidden_units = list(hidden_units)

    # Size for each layer of weights.
    layer_sizes = zip([dinput] + hidden_units[:-1], hidden_units)

    # Sets activation function.
    if isinstance(activation, str):
        Activation = getattr(nn, activation)
    else:
        Activation = activation

    # Loads layers.
    layers = list()
    for indim, outdim in layer_sizes:
        layers.append(nn.Linear(indim, outdim, **kwargs))

        if batch_norm:
            layers.append(nn.BatchNorm1d(outdim))

        layers.append(Activation())

        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(hidden_units[-1], doutput))
    # Loads DNN.
    net = nn.Sequential(*layers)
    return net


class DNN(nn.module):
    """ Implementation of dense neural networks. """

    def __init__(
        self,
        dinput,
        doutput,
        hidden_units=(16, 16),
        activation="ReLU",
        dropout=0.0,
        batch_norm=False,
        net_fn=nn.Sequential,
        **kwargs
    ):
        super(DNN,self).__init__()
        # Lists number of units in the hidden layers.
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        hidden_units = list(hidden_units)

        # Size for each layer of weights.
        layer_sizes = zip([dinput] + hidden_units[:-1], hidden_units)

        # Sets activation function.
        if isinstance(activation, str):
            Activation = getattr(nn, activation)
        else:
            Activation = activation

        # Loads layers.
        layers = list()
        for indim, outdim in layer_sizes:
            layers.append(nn.Linear(indim, outdim, **kwargs))

            if batch_norm:
                layers.append(nn.BatchNorm1d(outdim))

            layers.append(Activation())

            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_units[-1], doutput))
        # Loads DNN.
        self.net = nn.Sequential(*layers)
        return


class AutoEncoder(nn.Module):
    """ Implementation of autoencoders inherited from nn.Module.

    Attributes:
        beta: regularization coefficient.
        latent_dim: latent dimension of the embedding.
        hidden_units: list of number of hidden units for encoder and decoder. 
        encoder_net: encoder module of the autoencoder.
        decoder_net: decoder module of the autoencoder.
        mse: mean square loss for the autoencoder.
    """

    LossComps = namedtuple("AELoss", "mse reg")
    Outputs = namedtuple("AEOutputs", "recon code")

    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_net=None,
        decoder_net=None,
        hidden_units=None,
        beta=0,
        dropout=0,
        mse=None,
        **kwargs
    ):
        """ Initializes."""
        super(AutoEncoder, self).__init__(**kwargs)

        # Loads encoder.
        if encoder_net is None:
            assert hidden_units is not None
            encoder_net = self.build_encoder(
                input_dim, latent_dim, hidden_units, dropout=dropout
            )

        # Loads decoder.
        if decoder_net is None:
            assert hidden_units is not None
            decoder_net = self.build_decoder(
                input_dim, latent_dim, hidden_units, dropout=dropout
            )

        self.beta = beta
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net

        if mse is None:
            mse = nn.MSELoss(reduction="none")

        self.mse = mse

        return

    def build_encoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        """ Loads an encoder module for the autoencoder."""
        encoder = DNN(
            dinput=input_dim,
            doutput=latent_dim,
            hidden_units=hidden_units,
            **kwargs
        )
        return encoder.net

    def build_decoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        """ Loads a decoder module for the autoencoder.

        The hidden units are reversed from the encoder module.
        """
        decoder = DNN(
            dinput=latent_dim,
            doutput=input_dim,
            hidden_units=hidden_units[::-1],
            **kwargs
        )
        return decoder.net

    def encode(self, inputs, **kwargs):
        """ Computes the encodings given input features."""
        return self.encoder_net(inputs, **kwargs)

    def decode(self, code, **kwargs):
        """ Recovers the input from the encodings."""
        return self.decoder_net(code, **kwargs)

    def outputs(self, inputs, **kwargs):
        """ Computes the ouput given input features."""
        code = self.encode(inputs, **kwargs)
        recon = self.decode(code, **kwargs)
        outputs = self.Outputs(recon, code)
        return outputs

    def loss(self, inputs, outputs):
        """ Computes the loss on some inputs and its corresponding compositions."""
        mse = self.mse(outputs.recon, inputs).mean(dim=-1)
        # Computes the norm of the embedding to regularize.
        reg = torch.norm(outputs.code, dim=-1) ** 2
        total = mse + self.beta * reg
        comps = self.LossComps(mse, reg)
        return total, comps

    def forward(self, inputs, **kwargs):
        """ Given inputs, returns the loss, the composition of loss and the outputs."""
        outs = self.outputs(inputs, **kwargs)
        loss, comps = self.loss(inputs, outs)

        return loss, comps, outs

    def compute_encoding_shift(self, dataset, labels):
        """ Computes the shift between encodings of control cells and encodings of perturbed cells.
        """
        self.code_means = dict()
        # Computes encodings.
        inputs = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
        codes = self.encode(inputs)
        # Labels denote which perturbation on cells.
        for key in labels.unique():
            mask = labels == key
            self.code_means[key] = codes[mask.values].mean(0)

        return
    
    def predict(self, inputs, source, target, decode=True):
        """ Predicts the features of cells under some perturbation using AutoEncoder.
        
        Args:
            model: AutoEncoder, model used for prediction.
            inputs: inputs of the model.
            source: initial condition of the cells.
            target: target condition of the cells.
            decode: bool, whether to use the decoding module. Default is True.
        """
        # Switches to evaluation mode.
        self.eval()
        # Computes the shift from encodings of original cells to encodings of perturbed cells.
        shift = self.code_means[target] - self.code_means[source]
        # Computes encodings of the inputs.
        codes = self.encode(inputs)
        # Only returns shifted encodings.
        if not decode:
            return codes + shift
        # Returns reconstructions.
        outputs = self.decode(codes + shift)
        return outputs


class ConditionalAutoEncoder(AutoEncoder):
    """ Conditional autoencoders inherited from the AutoEncoder class.

    Attributes:
        conditions: list of conditions considered.
        n_cats: number of conditions considered.
    """

    def __init__(self, *args, conditions, **kwargs):
        """ Initializes. """
        self.conditions = conditions
        self.n_cats = len(conditions)
        super(ConditionalAutoEncoder, self).__init__(*args, **kwargs)
        return

    def build_encoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        """ Builds the encoder module using the method from AutoEncoder. 
        The input dimension equals to the feature dimension plus the dimension
        of one-hot vectors which represent the conditions. """
        net = super().build_encoder(
            input_dim=input_dim + self.n_cats,
            latent_dim=latent_dim,
            hidden_units=hidden_units,
            **kwargs
        )

        return net

    def build_decoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        """ Builds the decoder module using the method from AutoEncoder. 
        The encoding dimension equals to the latent dimension plus the dimension
        of one-hot vectors which represent the conditions."""
        net = super().build_decoder(
            input_dim=input_dim,
            latent_dim=latent_dim + self.n_cats,
            hidden_units=hidden_units,
            **kwargs
        )

        return net

    def condition(self, data, labels):
        """ Extends the features with conditions using one-hot vectors. """
        conds = nn.functional.one_hot(labels, self.n_cats)
        return torch.cat([data, conds], dim=1)

    def encode(self, inputs, **kwargs):
        """ Encodes the input features with conditions. """
        data, labels = inputs
        cond = self.condition(data, labels)
        return self.encoder_net(cond)

    def decode(self, codes, **kwargs):
        """ Decodes the latent embeddings with conditions."""
        data, labels = codes
        cond = self.condition(data, labels)
        return self.decoder_net(cond)

    def outputs(self, inputs, decode_as=None, **kwargs):
        """ Computes the outputs using inputs with conditions. """
        data, label = inputs
        assert len(data) == len(label)

        # Uses original labels for decoding unless special requests. 
        decode_label = label if decode_as is None else decode_as

        if isinstance(decode_label, str):
            raise NotImplementedError

        if isinstance(decode_label, int):
            decode_label = decode_label * torch.ones(len(data), dtype=int)

        # Computes embeddings and outputs.
        code = self.encode((data, label), **kwargs)
        recon = self.decode((code, decode_label), **kwargs)
        outputs = self.Outputs(recon, code)
        return outputs

    def forward(self, inputs, beta=None, **kwargs):
        """ Forward pass function to compute the outputs, the corresponding loss 
        and its compositions. """
        values, _ = inputs
        outs = self.outputs(inputs, **kwargs)
        loss, comps = self.loss(values, outs)
        return loss, comps, outs
    
    def predict(self,inputs,target):
        """ Predicts the reconstruction for some inputs with conditions using ConditionalAutoEncoder.
        
        Args:
            model: ConditionalAutoEncoder, model to reconstruct the inputs.
            inputs: inputs for the model.
            target: the condition used for the reconstruction.
            
        Returns:
            The reconstruction outputs.
        """
        # Switches to evaluation mode.
        self.eval()
        # Represents the condition with its index.
        target_code = self.conditions.index(target)
        outputs = self.outputs(inputs, decode_as=target_code).recon
        return outputs
