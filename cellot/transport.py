import anndata
from torch.utils.data import DataLoader
from cellot.models.ae import ConditionalAutoEncoder
from cellot.networks.icnns import ICNN


def transport(config, model, dataset, return_as="anndata", dosage=None, **kwargs):
    """ An integrated function to predict cell features using different models.
    
    Args:
        config: ConfigDict, configuration info.
        model: model from which the prediction is computed.
        dataset: data used for prediction.
        return_as: the returned datatype. Default is "anndata".
        dosage: float between 0 and 1, the drug dosage. Default is None.
        **kwargs: additional arguments.
    
    Returns:
        The predictions of the cell features after particular perturbation.
    """

    # Gets model name. Uses "cellot" if nonexist.
    name = config.model.get("name", "cellot")

    # Loads input data.
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs = next(iter(loader))

    if name == "cellot":
        outputs = transport_cellot(model, inputs)
    elif name == "scgen":
        outputs = transport_scgen(
            model,
            inputs,
            source=config.data.source,
            target=config.data.target,
            **kwargs
        )
    elif name == "cae":
        outputs = transport_cae(model, inputs, target=config.data.target)
    # Unexpected model names.
    else:
        raise ValueError

    # Linear change w.r.t to dosage.
    if dosage is not None:
        outputs = (1 - dosage) * inputs + dosage * outputs

    if return_as == "anndata":
        outputs = anndata.AnnData(
            outputs.detach().numpy(),
            obs=dataset.adata.obs.copy(),
            var=dataset.adata.var.copy(),
        )

    return outputs


def transport_cellot(model, inputs):
    """ Predicts the cell features after perturbation with CellOT.
    
    Args:
        model: the CellOT model used for prediction.
        inputs: features of the control cells.
    
    Returns:
        The predictions of the cell features after perturbation.
    """

    f, g = model
    # Switches to evaluation mode.
    g.eval()
    outputs = g.transport(inputs.requires_grad_(True))
    return outputs


def transport_scgen(model, inputs, source, target, decode=True):
    """ Predicts the features of cells under some perturbation using AutoEncoder.
    
    Args:
        model: AutoEncoder, model used for prediction.
        inputs: inputs of the model.
        source: features of the control cells to induce the encoding shift.
        target: features of the perturbed cells to induce the encoding shift.
        decode: bool, whether to use the decoding module. Default is True.
    
    Returns:
        The predictions of the perturbed cell features if decode is True. Otherwise
        returns the encodings of the perturbed cell features.
    """

    # Switches to evaluation mode.
    model.eval()
    # Computes the shift from encodings of control cells to encodings of perturbed cells.
    shift = model.code_means[target] - model.code_means[source]
    # Computes encodings of the inputs.
    codes = model.encode(inputs)
    # Only returns shifted encodings.
    if not decode:
        return codes + shift
    # Returns reconstructions.
    outputs = model.decode(codes + shift)
    return outputs


def transport_cae(model:ConditionalAutoEncoder, inputs, target):
    """ Predicts the reconstruction for some inputs with conditions using ConditionalAutoEncoder.
    
    Args:
        model: ConditionalAutoEncoder, model to reconstruct the inputs.
        inputs: inputs for the model.
        target: the condition used for the reconstruction.
        
    Returns:
        The reconstruction outputs.
    """

    # Switches to evaluation mode.
    model.eval()
    # Represents the condition with its index.
    target_code = model.conditions.index(target)
    outputs = model.outputs(inputs, decode_as=target_code).recon
    return outputs
