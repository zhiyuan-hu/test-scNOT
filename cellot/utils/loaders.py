""" Functions to load data, models.
"""


import cellot.models
from cellot.data.cell import load_cell_data
from ml_collections import ConfigDict


def load_data(config:ConfigDict, **kwargs) -> tuple:
    """ Loads the data for experiments.
    
    Args:
        config: ConfigDict that contains the configuration info.
        **kwargs: additional arguments.
        
    Returns:
        A tuple consists of data used for the experiment.
    """

    # Gets datatype from config. Uses 'cell' if no info is provided by config.
    data_type = config.get("data.type", "cell")
    # Uses load_cell_data() as the loader.
    if data_type in ["cell", "cell-merged", "tupro-cohort"]:
        loadfxn = load_cell_data
    # Unexpected datatypes.
    else:
        raise ValueError

    return loadfxn(config, **kwargs)


def load_model(config:ConfigDict, restore=None, **kwargs):
    """ Loads the model for experiments.
    
    Args:
        config: ConfigDict that contains the configuration info.
        restore: filepath from which a previous state is restored. Default is None.
        **kwargs: additional arguments.
        
    Returns:
        A model and its corresponding optimizer.
    """

    # Gets model info. Uses "cellot" if not provided by config.
    name = config.get("model.name", "cellot")
    # Uses corresponding loader functions.
    if name == "cellot":
        loadfxn = cellot.models.load_cellot_model

    elif name == "scgen":
        loadfxn = cellot.models.load_autoencoder_model

    elif name == "cae":
        loadfxn = cellot.models.load_autoencoder_model
    # Unexpected models.
    else:
        raise ValueError

    return loadfxn(config, restore=restore, **kwargs)


def load(config:ConfigDict, restore=None, include_model_kwargs=False, **kwargs):
    """ An integrated loader function for both data and models.
    
    Args:
        config: ConfigDict that contains the configuration info.
        restore: filepath from which a previous state of the model is restored. Default is None.
        include_model_kwargs: bool, indicating whether to return model parameters. Default is False.
        **kwargs: additional arguments.
    
    Returns:
        The model, its corresponding optimizer, the data loader. Also returns model parameters as a 
        dict if include_model_kwargs is True.
    """

    # Loads data.
    loader, model_kwargs = load_data(config, include_model_kwargs=True, **kwargs)
    # Loads models.
    model, opt = load_model(config, restore=restore, **model_kwargs)

    if include_model_kwargs:
        return model, opt, loader, model_kwargs

    return model, opt, loader
