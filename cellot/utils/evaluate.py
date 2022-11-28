from tqdm.auto import tqdm
import pandas as pd
from cellot.losses.mmd import mmd_distance
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy
import anndata
import torch
import numpy as np
from cellot.utils.loaders import load_data, load_model, load
from cellot.utils import load_config
from cellot.models.ae import compute_scgen_shift
from cellot.transport import transport


def compute_knn_enrichment(pushfwd, treated, return_joint=False, ncells=None):
    if ncells is None:
        ncells = min(len(pushfwd), len(treated))
    assert ncells <= len(pushfwd)
    assert ncells <= len(treated)

    pushfwd = pushfwd.sample(n=ncells)
    treated = treated.sample(n=ncells)

    joint = pd.concat((pushfwd, treated), axis=0)

    labels = pd.concat((
        pd.Series('pushfwd', index=pushfwd.index),
        pd.Series('treated', index=treated.index)
    )).astype('category')

    n_neighbors = min([251, ncells])
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(joint)
    dists, knn = model.kneighbors(pushfwd)
    # assert np.all(knn[:, 0] == np.arange(len(knn)))
    knn = pd.DataFrame(knn[:, 1:], index=pushfwd.index)

    enrichment = knn.applymap(lambda x: labels.iloc[x] == 'pushfwd')
    knn = knn.applymap(lambda x: labels.index[x])

    if return_joint:
        return knn, enrichment, joint

    return knn, enrichment


def compute_drug_signature_differences(control, treated, pushfwd):
    base = control.mean(0)

    true = treated.mean(0) - base
    pred = pushfwd.mean(0) - base

    diff = true - pred
    return diff


def compute_mmd_df(
        target, transport,
        gammas=None,
        subsample=False,
        ncells=None, nreps=5):

    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    gammas = list(gammas)

    def compute(ncells=None):
        for g in tqdm(gammas * nreps, desc='mmd', leave=False):
            mmd = mmd_distance(
                target if ncells is None else target.sample(ncells),
                transport if ncells is None else transport.sample(ncells),
                g)

            yield g, mmd

    if subsample:
        if ncells is None:
            ncells = min(len(target), len(transport))
        else:
            ncells = min(len(target), len(transport), ncells)
    elif ncells is not None:
        assert ncells <= min(len(target), len(transport))

    mmd = pd.DataFrame(compute(ncells), columns=['gamma', 'mmd'])
    return mmd


def patch_scgen_shift(config, model):
    assert config.model.name == "scgen"

    loader = load_data(config)
    labels = loader.train.dataset.adata.obs[config.data.condition]
    compute_scgen_shift(model, loader.train.dataset, labels=labels)
    return


def load_projectors(aedir, embedding, where):
    config = load_config(aedir / "config.yaml")
    ae, *_, dataset = load(
        config,
        restore=aedir / "cache" / "model.pt",
        split_on=["transport"],
        return_as="dataset",
    )
    ae.eval()
    patch_scgen_shift(config, ae)

    try:
        pcs = dataset.source.adata.varm["PCs"]
        features = dataset.source.adata.var_names

    except KeyError:
        if 'dimension_reduction' in config.data:
            tmp = deepcopy(config)
            del tmp.data.dimension_reduction
            dataset = load_data(tmp, split_on=[], return_as="anndata")
            pcs = dataset.varm["PCs"]
            features = dataset.var_names

    if embedding == "pca":

        def encode(df):
            return df @ pcs

        def decode(df):
            df = df @ pcs.T
            df.columns = features
            return df

    elif embedding == "ae":

        def encode(df):
            codes = ae.encode(torch.Tensor(df.values)).detach().numpy()
            return pd.DataFrame(codes, index=df.index)

        def decode(df):
            recon = ae.decode(torch.Tensor(df.values)).detach().numpy()
            return pd.DataFrame(recon, index=df.index, columns=features)

    else:
        assert embedding is None

        def encode(df):
            return df

        def decode(df):
            return df

    return encode, decode


def read_embedding_context(config):

    embedding = None
    if "ae_emb" in config.data:
        embedding = "ae"
    elif "dimension_reduction" in config.data:
        assert config.data.dimension_reduction.name == "pca"
        embedding = "pca"

    return embedding


def load_all_inputs(config, setting, embedding, where):
    dataset, model_kwargs = load_data(
        config,
        split_on=["split", "transport"],
        return_as="dataset",
        include_model_kwargs=True)

    if setting == 'iid':
        to_pushfwd = dataset.test.source

    elif setting == 'ood':
        to_pushfwd = dataset.ood.source

    else:
        raise ValueError(f'unknown setting, {setting} must be [iid, ood]')

    # reload control and treated as per embedding & where
    if embedding == 'pca' and where == 'data_space':
        # remove pca projections
        assert config.data.dimension_reduction.name == 'pca'
        config = deepcopy(config)
        del config.data.dimension_reduction
        dataset = load_data(
            config,
            split_on=["split", "transport"],
            return_as="dataset")
        pass

        if config.model.name == 'identity':
            to_pushfwd = dataset.test.source

    elif config.model.name == 'cellot' and embedding == 'ae' and where == 'data_space':
        # remove ae projections
        assert 'ae_emb' in config.data
        config = deepcopy(config)
        del config.data.ae_emb
        dataset = load_data(
            config,
            split_on=["split", "transport"],
            return_as="dataset")

    if setting == 'iid':
        control = pd.concat(
            (dataset.test.source.adata.to_df(), dataset.train.source.adata.to_df())
        )

        treated = pd.concat(
            (dataset.test.target.adata.to_df(), dataset.train.target.adata.to_df())
        )

    elif setting == 'ood':
        control = dataset.ood.source.adata.to_df()

        if 'target' in dataset.ood.keys():
            treated = dataset.ood.target.adata.to_df()
        else:
            treated = pd.concat((
                dataset.test.target.adata.to_df(),
                dataset.train.target.adata.to_df()
            ))

    else:
        raise ValueError(f'unknown setting, {setting} must be [iid, ood]')

    obs = load_data(config, split_on=[], return_as='anndata').obs
    return control, treated, to_pushfwd, obs, model_kwargs


def load_conditions(expdir, where, setting):
    embedding = read_embedding_context(
        load_config(expdir.parent / "model-cellot" / "config.yaml")
    )
    encode, decode = load_projectors(expdir.parent / "model-scgen", embedding, where)

    config = load_config(expdir / "config.yaml")
    if "ae_emb" in config.data:
        config.data.ae_emb.path = str(expdir.parent / "model-scgen")

    control, treated, to_pushfwd, obs, model_kwargs = load_all_inputs(config, setting, embedding, where)
    if config.model.name != 'cellot' and embedding == 'ae' and where == 'latent_space':
        control = encode(control)
        treated = encode(treated)
        pass

    if config.model.name == "identity":
        imputed = to_pushfwd.adata.to_df()
        if embedding == 'ae' and where == 'latent_space':
            imputed = encode(imputed)

    elif config.model.name == "random":
        treated, imputed = np.array_split(treated.sample(frac=1), 2)

    else:
        assert config.model.name in {"scgen", "cae", "cellot"}
        model, *_ = load_model(
            config, restore=expdir / "cache" / "model.pt", **model_kwargs
        )

        if config.model.name == "scgen" and not hasattr(model, "code_means"):
            patch_scgen_shift(config, model)

        if config.model.name == 'scgen' and embedding == "ae" and where == "latent_space":
            imputed = transport(
                config, model, to_pushfwd, decode=False, return_as=None
            )

            imputed = pd.DataFrame(
                imputed.detach().numpy(), index=to_pushfwd.adata.obs_names
            )
        else:
            imputed = transport(config, model, to_pushfwd).to_df()

        # think about how to project
        if config.model.name == 'cellot' and where == 'data_space':
            imputed = decode(imputed)

        elif config.model.name != 'cellot' and embedding == 'pca' and where == 'data_space':
            imputed = decode(imputed)

        elif config.model.name == 'cae' and embedding == 'ae' and where == 'latent_space':
            imputed = encode(imputed)

    imputed = anndata.AnnData(imputed, obs=obs.loc[imputed.index])

    return control, treated, imputed
