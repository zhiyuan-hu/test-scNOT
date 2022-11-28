""" Implementation of the CellOT structure.
"""


from pathlib import Path
import torch
from collections import namedtuple
from cellot.networks.icnns import ICNN
from absl import flags
from typing import Optional


FLAGS = flags.FLAGS


# FGPair is a subclass of tuple whose first element is named f
# and second element g.
FGPair = namedtuple("FGPair", "f g")


class CellOT():
    """ Implementation of the CellOT structure.
    
    Attributes:
        f(ICNN): one of the dual networks.
        g(ICNN): one of the dual networks.
        opts(FGPair): optimizers for f and g.
    """

    def __init__(self,config,restore=None,**kwargs) -> None:
        self.build_networks(config,**kwargs)
        self.build_opts(config)
        self.restore(restore)

    def build_networks(self,config, **kwargs):
        """ Builds dual potential ICNNs with specified configuration."""
        # Function to extract the information for initialization
        def unpack_kernel_init_fxn(name="uniform", **kwargs):
            if name == "normal":
            # Uses normal distribution to initialize.
                def init(*args):
                    return torch.nn.init.normal_(*args, **kwargs)
            elif name == "uniform":
            # Uses uniform distribution to initialize.
                def init(*args):
                    return torch.nn.init.uniform_(*args, **kwargs)
            else:
                raise ValueError
            return init

        # Uses default configuration for hidden layers if not passed by kwargs.
        kwargs.setdefault("hidden_units", [64] * 4)
        # Uploads model information to kwargs.
        # e.g. parameters specific to g are stored in config.model.g.
        kwargs.update(dict(config.get("model", {})))

        # Removes extra information.
        kwargs.pop("name")
        if "latent_dim" in kwargs:
            kwargs.pop("latent_dim")
        # Extracts model information respectively. 
        fupd = kwargs.pop("f", {})
        gupd = kwargs.pop("g", {})

        # Parameters for ICNN f.
        fkwargs = kwargs.copy()
        fkwargs.update(fupd)
        fkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
            **fkwargs.pop("kernel_init_fxn")
        )

        # Parameters for ICNN g.
        gkwargs = kwargs.copy()
        gkwargs.update(gupd)
        gkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
            **gkwargs.pop("kernel_init_fxn")
        )

        self.f = ICNN(**fkwargs)
        self.g = ICNN(**gkwargs)

        # Prints model parameters if required.
        if "verbose" in FLAGS and FLAGS.verbose:
            print(self.g)
            print(kwargs)

        return

    def build_opts(self,config):
        """ Builds optimizers for CELLOT networks."""
        # Loads optimizer's configuration.
        kwargs = dict(config.get("optim", {}))
        assert kwargs.pop("optimizer", "Adam") == "Adam"
        # Model configuration for f and g, empty if not existed in kwargs.
        fupd = kwargs.pop("f", {})
        gupd = kwargs.pop("g", {})
        # Configuration for f.
        fkwargs = kwargs.copy()
        fkwargs.update(fupd)
        fkwargs["betas"] = (fkwargs.pop("beta1", 0.9), fkwargs.pop("beta2", 0.999))
        # Configuration for g.
        gkwargs = kwargs.copy()
        gkwargs.update(gupd)
        gkwargs["betas"] = (gkwargs.pop("beta1", 0.9), gkwargs.pop("beta2", 0.999))
        # Loads optimizers for f and g.
        self.opts = FGPair(
            f=torch.optim.Adam(self.f.parameters(), **fkwargs),
            g=torch.optim.Adam(self.g.parameters(), **gkwargs),
        )
        return

    def restore(self,restore=None):
        """ Restores a previous state. """
        if restore is not None and Path(restore).exists():
            ckpt = torch.load(restore)
            self.f.load_state_dict(ckpt["f_state"])
            self.opts.f.load_state_dict(ckpt["opt_f_state"])

            self.g.load_state_dict(ckpt["g_state"])
            self.opts.g.load_state_dict(ckpt["opt_g_state"])
        return
    
    def compute_loss_f(
        self,
        source:torch.tensor,
        target:torch.tensor,
        transport:Optional[torch.tensor]=None
        ):
        """ Computes the loss of ICNN f given input features and predicted features."""
        # If no prediction is passed, uses g to predict.
        if transport is None:
            transport = self.g.transport(source)
        # Computes the loss, implemented from part of equation (9) in the CELLOT paper.
        return -self.f(transport) + self.f(target)
    
    def compute_loss_g(
        self,
        source:torch.tensor,
        transport:Optional[torch.tensor]=None
        ):
        """ Computes the loss of ICNN g given input features and predicted features."""
        # If no prediction is passed, uses g to predict.
        if transport is None:
            transport = self.g.transport(source)
        # Computes the loss, implemented from part of equation (9) in the CellOT paper.
        return self.f(transport) - torch.multiply(source, transport).sum(-1, keepdim=True)
    
    def compute_g_constraint(self,form=None,beta=0):
        """ Computes the additional penalty for negative weights of g.

        Args:
            form: indicates which type of measure is performed for negative weights. Default is None.
            beta: penalty strength. Default is 0.
        """
        # No penalty is computed
        if form is None or form == "None":
            return 0
        # If clamping is performed, no penalty is computed.
        if form == "clamp":
            self.g.clamp_w()
            return 0
        # Penalizes negative weights.
        elif form == "fnorm":
            if beta == 0:
                return 0
            return beta * sum(map(lambda w: w.weight.norm(p="fro"), self.g.W))
        raise ValueError
    
    def compute_w2_distance(self,source,target,transport=None):
        """ Computes the 2-norm Wasserstein distance between predictions and targets."""
        # If no prediction is passed, uses g to predict.
        if transport is None:
            transport = self.g.transport(source).squeeze()
        with torch.no_grad():
            Cpq = (source * source).sum(1, keepdim=True) + (target * target).sum(1, keepdim=True)
            Cpq = 0.5 * Cpq

            cost = (
                self.f(transport)
                - torch.multiply(source, transport).sum(-1, keepdim=True)
                - self.f(target)
                + Cpq
            )
            cost = cost.mean()
        return cost
    
    def predict(self,inputs:torch.tensor):
        """Predicts the cell features after perturbation with CellOT."""
        # Switches to evaluation mode.
        self.g.eval()
        outputs = self.g.transport(inputs.requires_grad_(True))
        return outputs


def numerical_gradient(param:float, fxn, *args, eps=1e-4):
    """ Numerically computes the gradient of a function at some point.

    Args:
        param: float, the parameter to define the input position of the function.
        fxn: the function of which the gradient is computed on.
        *args: the arguments of the function.
        eps: the small value perturbed on the param. Default is 1e-4.
    
    Returns:
        The gradient of fxn at *args.
    """

    # Computes the function value at param+eps.
    with torch.no_grad():
        param += eps
    plus = float(fxn(*args))

    # Computes the function value at param-eps.
    with torch.no_grad():
        param -= 2 * eps
    minus = float(fxn(*args))

    # Resets param to its original value.
    with torch.no_grad():
        param += eps

    # Computes and returns the gradient.
    return (plus - minus) / (2 * eps)
