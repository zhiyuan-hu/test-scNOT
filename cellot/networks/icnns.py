""" Implementation of input convex neural networks.
"""


import torch
from torch import autograd
import numpy as np
from torch import nn
from numpy.testing import assert_allclose


# Shortcuts for activation functions.
ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
}


class NonNegativeLinear(nn.Linear):
    """ Implementation of a single neural network layer with nonnegative weights.

    Attributes:
        beta(float): the scale for the softplus kernel. 
            Default is 1.
    """

    def __init__(
        self,
        *args,
        beta=1.0,
        **kwargs
    ):
        """ Initializes NonNegativeLinear with input beta value."""
        # First initializes with superclass.
        super(NonNegativeLinear, self).__init__(*args, **kwargs)
        self.beta = beta
        return

    def forward(self,x):
        """ Computes the output of this layer given the input."""
        # Uses the softplus kernel to ensure nonnegative weights.
        return nn.functional.linear(x, self.kernel(), self.bias)

    def kernel(self):
        """ Builds the softplus kernel of weights using the attribute beta."""
        return nn.functional.softplus(self.weight, beta=self.beta)


class ICNN(nn.Module):
    """ Implementation of a input convex nueral network.

    Attributes:
        input_dim(int): the dimension of the input.
        hidden_units(list): the number of units on each hidden layer.
        activation(str): the employed activation function. 
            Default is 'LeakyReLU'.
        softplus_W_kernels(bool): whether to use softplus to ensure nonnegative weights. 
            Default is False.
        softplus_beta(float): the scale for the softplus kernel.
            Default is 1.
        fnorm_penalty(float): the penalty strength for negative weights. 
            Default is 0.
        kernel_init_fxn: a function used to intialize weights.
            Default is None.
    """

    def __init__(
        self,
        input_dim,
        hidden_units,
        activation="LeakyReLU",
        softplus_W_kernels=False,
        softplus_beta=1,
        # std is not used here!
        #std=0.1,
        fnorm_penalty=0,
        kernel_init_fxn=None,
    ):
        """ Initializes. """
        # Assigns arguments to attributes.
        super(ICNN, self).__init__()
        self.fnorm_penalty = fnorm_penalty
        self.softplus_W_kernels = softplus_W_kernels

        if isinstance(activation, str):
            activation = ACTIVATIONS[activation.lower().replace("_", "")]
        self.sigma = activation

        # Outputs a scalar.
        units = hidden_units + [1]

        # z_{l+1} = \sigma_l(W_l*z_l + A_l*x + b_l)
        # W_0 = 0

        # If using softplus to ensure nonnegativity:
        if self.softplus_W_kernels:
            # Uses NonNegativeLinear as unit layer.
            def WLinear(*args, **kwargs):
                return NonNegativeLinear(*args, **kwargs, beta=softplus_beta)
        else:
            # Uses normal Linear layer.
            WLinear = nn.Linear

        # Defines weights for every layer. 
        self.W = nn.ModuleList(
            [
                WLinear(idim, odim, bias=False)
                for idim, odim in zip(units[:-1], units[1:])
            ]
        )

        # Defines feed-forward weights for every layer.
        self.A = nn.ModuleList(
            [nn.Linear(input_dim, odim, bias=True) for odim in units]
        )

        # Initializes weights using kernel_init_fxn if required.
        if kernel_init_fxn is not None:

            for layer in self.A:
                kernel_init_fxn(layer.weight)
                nn.init.zeros_(layer.bias)

            for layer in self.W:
                kernel_init_fxn(layer.weight)

        return

    def forward(self, x):
        """ Computes the output of the ICNN given some input x."""
        z = self.sigma(0.2)(self.A[0](x))
        # Ensures positive hessian for better performance.
        z = z * z

        for W, A in zip(self.W[:-1], self.A[1:-1]):
            z = self.sigma(0.2)(W(z) + A(x))

        y = self.W[-1](z) + self.A[-1](x)

        return y

    def transport(self, x:torch.tensor):
        """ Computes the predicted perturbed features of a cell given its original features.

        According to theory, the prediction is the gradient of the scalar output w.r.t to the 
        network input, evaluated at the specific samples x here.  
        """
        assert x.requires_grad

        # Computes gradients as predictions. 
        (output,) = autograd.grad(
            self.forward(x),
            x,
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((x.size()[0], 1), device=x.device).float(),
        )
        return output

    def clamp_w(self):
        """ Forcibly sets negative weights to zero to ensure nonnegativity."""
        # If using softplus kernels, nonnegativity is automatically guaranteed.
        if self.softplus_W_kernels:
            return

        for w in self.W:
            w.weight.data = w.weight.data.clamp(min=0)
        return

    def penalize_w(self):
        """ Computes the penalty for negative weights."""
        return self.fnorm_penalty * sum(
            map(lambda x: torch.nn.functional.relu(-x.weight).norm(), self.W)
        )


def test_icnn_convexity(icnn:ICNN):
    """ Tests if an instance of ICNN is indeed input convex.

    Args:
        icnn: an instance of the class ICNN.

    Raises:
        AssertionError: if icnn is not input convex.
    """

    data_dim = icnn.A[0].in_features

    zeros = np.zeros(100)

    # Tests for 100 times.
    for _ in range(100):
        # For each iteration generates 100 samples.
        x = torch.rand((100, data_dim))
        y = torch.rand((100, data_dim))

        fx = icnn(x)
        fy = icnn(y)

        # Compares t*fx+(1-t)*fy with f(t*x+(1-t)*y)
        for t in np.linspace(0, 1, 10):
            fxy = icnn(t * x + (1 - t) * y)
            res = (t * fx + (1 - t) * fy) - fxy
            res = res.detach().numpy().squeeze()
            assert_allclose(np.minimum(res, 0), zeros, atol=1e-6)
