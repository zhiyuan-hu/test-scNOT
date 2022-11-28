#!/usr/bin/python3
# author: Stefan Stark

import anndata
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
#from cellot.models import load_autoencoder_model
from cellot.utils import load_config
from cellot.data.utils import cast_dataset_to_loader
from cellot.utils.helpers import nest_dict
from absl import logging
import re
from math import ceil


class AnnDataDataset(Dataset):
    """ AnnData dataset inherited from torch.utils.data.Dataset to be loaded to 
        ML models.
    
    Attributes:
        adata: an anndata.AnnData object consists of the majority of the data.
        obs: the observation labels, used for picking a subset from whole data.
            Default is None.
        categories: a list to categorize cells. Default is None.
        include_index: whether to include the index of the cell when passing an
            item. Default is False.
    """

    def __init__(self, adata:anndata.AnnData, obs=None, categories=None, include_index=False,
                 dim_red=None):
        """ Initializes. """
        self.adata = adata
        # Converts the design matrix to np.float32.
        self.adata.X = self.adata.X.astype(np.float32)
        self.obs = obs
        self.categories = categories
        self.include_index = include_index

    def __len__(self):
        """ Gets the size of the dataset. """
        return len(self.adata)

    def __getitem__(self, idx):
        """ Gets one item from the dataset. """
        # Extracts the value.
        value = torch.tensor(self.adata.X[idx].todense())

        # Includes the category info.
        if self.obs is not None:
            meta = self.categories.index(self.adata.obs[self.obs].iloc[idx])
            value = value, int(meta)

        # Includes the label for this observation.
        if self.include_index:
            return self.adata.obs_names[idx], value

        return value


def read_list(arg):
    """ Reads a list from storage or a sequence of variables.
    
    Args:
        arg: str indicating the filepath or a sequence of variables.
    
    Returns:
        A list containing the arg.
    """
    
    if isinstance(arg, str):
        # Reads from storage.
        arg = Path(arg)
        assert arg.exists()
        lst = arg.read_text().split()
    else:
        # Packs arguments.
        lst = arg

    return list(lst)


def read_single_anndata(config, path=None):
    """ Reads a single anndata file from storage.
    
    Args:
        config: ConfigDict containing the configuration info.
        path: filepath from where the data is read. Default is None.
    
    Returns:
        An anndata.AnnData object representing the dataset.
    """

    # Uses config data path if no path specified. 
    if path is None:
        path = config.data.path

    data = anndata.read(path)

    # Subsets by features specified.
    if 'features' in config.data:
        features = read_list(config.data.features)
        data = data[:, features].copy()

    # Selects subset of individuals.
    if 'individuals' in config.data:
        data = data[data.obs[config.data.individuals[0]].isin(
          config.data.individuals[1])]

    # label conditions as source/target distributions
    # config.data.{source,target} can be a list now
    # e.g., config.data.source contains which cells are 
    # used as the source distribution. 
    transport_mapper = dict()
    for value in ['source', 'target']:
        key = config.data[value]
        if isinstance(key, list):
            for item in key:
                transport_mapper[item] = value
        else:
            transport_mapper[key] = value

    data.obs['transport'] = (
            data.obs[config.data.condition]
            .apply(transport_mapper.get)
    )

    # Uses all other unlabeled observations as the target distribution.
    if config.data['target'] == 'all':
        data.obs['transport'].fillna('target', inplace=True)

    mask = data.obs['transport'].notna()
    if 'subset' in config.data:
        for key, value in config.data.subset.items():
            if not isinstance(value, list):
                value = [value]
            mask = mask & data.obs[key].isin(value)

    # Writes train/test/valid into split column
    if 'datasplit' in config:
        data.obs['split'] = split_cell_data(data, **config.datasplit)

    return data[mask].copy()


def read_merged_anndata(config, pathlist=None):
    """ Reads multiple anndata files and merges them into one dataset.
    
    Args:
        config: ConfigDict containing the configuration info.
        pathlist: list of filepaths from where the data is read. 
            Default is None.
    
    Returns:
        An anndata.AnnData object representing the dataset.
    """

    def read_list(arg):
        if isinstance(arg, list):
            return arg

        path = Path(arg)
        assert path.exists()
        return path.read_text().split()

    def iterate(pathlist):
        for path in pathlist:
            data = read_single_anndata(config, path)
            if np.isnan(data.X).any():
                logging.warning(f'Omitting due to NA {path}')
                continue

            yield data.uns['sample'], data

    assert (config.data.type == 'cell-merged')

    if pathlist is None:
        pathlist = read_list(config.data.paths)

    merged = anndata.concat(
        dict(iterate(pathlist)),
        label='sample',
        index_unique='-'
    )

    return merged


def load_cell_data(
        config,
        data=None,
        split_on=None,
        return_as='loader',
        include_model_kwargs=False,
        **kwargs):
    """ Integrated function to load cell data.
    
    Args:
        config: ml_collections.config_dict.ConfigDict, containing configuration
            information. 
        data: datasets to be loaded.
            Default is None.
        split_on: attributes on which the data is splitted.
            Default is None.
        return_as: the order to return the results.
            Default is 'loader'.
        include_model_kwargs: bool, whether to return model parameters.
            Default is False.
        **kwargs: additional arguments.
    
    Returns:
        A tuple containing the data objects requested by return_as.
    """

    if isinstance(return_as, str):
        return_as = [return_as]

    assert set(return_as).issubset({'anndata', 'dataset', 'loader'})
    config.data.condition = config.data.get('condition', 'drug')
    condition = config.data.condition

    # Loads data.
    if data is None:
        if config.data.type == 'cell':
            data = read_single_anndata(config, **kwargs)

        elif config.data.type == 'cell-merged':
            data = read_merged_anndata(config, **kwargs)

    # Subsets data by selecting particular properties.
    if config.data.get('select') is not None:
        keep = pd.Series(False, index=data.obs_names)
        for key, value in config.data.select.items():
            if not isinstance(value, list):
                value = [value]
            keep.loc[data.obs[key].isin(value)] = True
            assert keep.sum() > 0

        data = data[keep].copy()

    if 'dimension_reduction' in config.data:
        genes = data.var_names.to_list()
        name = config.data.dimension_reduction.name
        if name == 'pca':
            dims = config.data.dimension_reduction.get(
                    'dims', data.obsm['X_pca'].shape[1])

            data = anndata.AnnData(
                    data.obsm['X_pca'][:, :dims],
                    obs=data.obs.copy(),
                    uns=data.uns.copy())
            data.uns['genes'] = genes

    if 'ae_emb' in config.data:
        # load path to autoencoder
        assert config.get('model.name', 'cellot') == 'cellot'
        path_ae = Path(config.data.ae_emb.path)
        model_kwargs = {'input_dim': data.n_vars}
        config_ae = load_config(path_ae/'config.yaml')
        ae_model, _ = load_autoencoder_model(
            config_ae, restore=path_ae/'cache/model.pt',
            **model_kwargs)

        inputs = torch.Tensor(
                data.X
                if not sparse.issparse(data.X)
                else data.X.todense())

        genes = data.var_names.to_list()
        data = anndata.AnnData(
                ae_model.eval().encode(inputs).detach().numpy(),
                obs=data.obs.copy(),
                uns=data.uns.copy()
        )
        data.uns['genes'] = genes

    # cast to dense and check for nans
    if sparse.issparse(data.X):
        data.X = data.X.todense()
    assert not np.isnan(data.X).any()

    dataset_args = dict()
    model_kwargs = {}

    model_kwargs['input_dim'] = data.n_vars

    if config.get('model.name') == 'cae':
        condition_labels = sorted(data.obs[condition].cat.categories)
        model_kwargs['conditions'] = condition_labels
        dataset_args['obs'] = condition
        dataset_args['categories'] = condition_labels

    if split_on is None:
        if config.model.name == 'cellot':
            # datasets & dataloaders accessed as loader.train.source
            split_on = ['split', 'transport']

        elif config.model.name == 'scgen' or config.model.name == 'cae':
            split_on = ['split']

        else:
            raise ValueError

    if isinstance(split_on, str):
        split_on = [split_on]

    for key in split_on:
        assert key in data.obs.columns

    # Splits data by split_on.
    if len(split_on) > 0:
        splits = {
            (key if isinstance(key, str) else '.'.join(key)): data[index]
            for key, index
            in data.obs[split_on].groupby(split_on).groups.items()
        }

        dataset = nest_dict({
            key: AnnDataDataset(val.copy(), **dataset_args)
            for key, val
            in splits.items()
        }, as_dot_dict=True)

    else:
        dataset = AnnDataDataset(data.copy(), **dataset_args)

    if 'loader' in return_as:
        kwargs = dict(config.dataloader)
        kwargs.setdefault('drop_last', True)
        loader = cast_dataset_to_loader(dataset, **kwargs)

    returns = list()
    for key in return_as:
        if key == 'anndata':
            returns.append(data)

        elif key == 'dataset':
            returns.append(dataset)

        elif key == 'loader':
            returns.append(loader)

    # Returns model parameters.
    if include_model_kwargs:
        returns.append(model_kwargs)

    if len(returns) == 1:
        return returns[0]

    return tuple(returns)


def split_cell_data_train_test(
        data,
        groupby=None,
        random_state:int=0,
        holdout:dict=None,
        **kwargs
        ):
    """  Splits the cell data into training set and testing set.

    If passes an nonempty groupby, proportions between training and testing 
    samples in the same group will stay the same.

    Args:
        data: anndata.Anndata object to be splitted.
        groupby: which observation metadata to group by. 
            Default is None, i.e., no grouping is performed.
        random_state: int, random seed for reproducible results.
            Default is 0.
        holdout: dict, containing which samples to be held out.
            Default is None.
        **kwargs: additional arguments.

    Returns:
        A pandas.Series containing the partition of each observation.
    """

    # Creates a series to represent train-test split.
    split = pd.Series(None, index=data.obs.index, dtype=object)
    # Default no grouping performed.
    groups = {None: data.obs.index}
    # Groups the observations by groupby. groups is a dict: group -> [index].
    if groupby is not None:
        groups = data.obs.groupby(groupby).groups

    # Assigns train-test split.
    for key, index in groups.items():
        trainobs, testobs = train_test_split(
                index,
                random_state=random_state,
                **kwargs)
        split.loc[trainobs] = 'train'
        split.loc[testobs] = 'test'

    # Assigns holdout set.
    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[data.obs[key].isin(value)] = 'ood'

    return split


def split_cell_data_train_test_eval(
        data,
        test_size=0.15,
        eval_size=0.15,
        groupby=None,
        random_state=0,
        holdout=None,
        **kwargs):
    """  Splits the cell data into training set, testing set and validation set.

    If passes an nonempty groupby, proportions between training and testing 
    samples in the same group will stay the same.

    Args:
        data: anndata.Anndata object to be splitted.
        test_size: float in [0,1], proportion of the testing set.
            Default is 0.15.
        eval_size: float in [0,1], proportion of the evaluation set.
            Default is 0.15.
        groupby: which observation metadata to group by. Default is None, 
            i.e., no grouping is performed.
        random_state: int, random seed for reproducible results.
            Default is 0.
        holdout: dict, containing which samples to be held out.
            Default is None.
        **kwargs: additional arguments.

    Returns:
        A pandas.Series containing the partition of each observation.
    """

    split = pd.Series(None, index=data.obs.index, dtype=object)
    # Assigns holdout set.
    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[data.obs[key].isin(value)] = 'ood'

    # Groups observations.
    groups = {None: data.obs.loc[split != 'ood'].index}
    if groupby is not None:
        groups = data.obs.loc[split != 'ood'].groupby(groupby).groups

    # Splits.
    for key, index in groups.items():
        training, evalobs = train_test_split(
                index, random_state=random_state, test_size=eval_size)

        trainobs, testobs = train_test_split(
                training, random_state=random_state, test_size=test_size)

        split.loc[trainobs] = 'train'
        split.loc[testobs] = 'test'
        split.loc[evalobs] = 'eval'

    return split


def split_cell_data_toggle_ood(
        data, holdout, key, mode,
        random_state=0,
        **kwargs
        ):
    ''' Holds out ood sample, coordinated with iid split.

    ood stands for out-of-distribution, defined with (key,holdout) pair.
    For ood mode: holds out all cells from a sample.
    For iid mode: includes half of cells in split.

    Args:
        data: anndata.Anndata object to be splitted.
        holdout: indeces of observations to be held out.
        key: the identifier attribute of observations to be held out.
        mode: str, work mode.
            Chooses in {'ood','iid'}.
        random_state: int, random seed for reproducible results.
            Default is 0.
        **kwargs: additional arguments.
    
    Returns:
        A pandas.Series containing the partition of each observation.
    '''

    # Splits all observations into train and test.
    split = split_cell_data_train_test(
            data, random_state=random_state, **kwargs)

    if not isinstance(holdout, list):
        value = [holdout]

    # Identifies ood observations.
    ood = data.obs_names[data.obs[key].isin(value)]
    # Splits ood observations.
    trainobs, testobs = train_test_split(
            ood,
            random_state=random_state,
            test_size=0.5)

    if mode == 'ood':
        # Holds out all ood samples.
        split.loc[trainobs] = 'ignore'
        split.loc[testobs] = 'ood'

    elif mode == 'iid':
        # Includes half of ood samples in train.
        split.loc[trainobs] = 'train'
        split.loc[testobs] = 'ood'

    else:
        raise ValueError

    return split


def split_cell_data(data, name='train_test', **kwargs):
    """ An integrated function for splitting the data in different manners.
    
    Args:
        data: anndata.Anndata object to be splitted.
        name: split manners. Default is 'train_test'.
        **kwargs: additional arguments.
    
    Returns:
        A pandas.Series containing the partition of each observation.
    """
    
    if name == 'train_test':
        split = split_cell_data_train_test(data, **kwargs)
    elif name == 'toggle_ood':
        split = split_cell_data_toggle_ood(data, **kwargs)
    elif name == 'train_test_eval':
        split = split_cell_data_train_test_eval(data, **kwargs)
    else:
        raise ValueError

    return split.astype('category')
