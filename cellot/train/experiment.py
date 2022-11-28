import os
from absl import flags
import random
from ml_collections import ConfigDict
from pathlib import Path
import yaml
from cellot.utils.helpers import parse_cli_opts
from cellot.utils import load_config

import string

ALPHABET = string.ascii_lowercase + string.digits

# Defines hyperparameters for global use.
flags.DEFINE_string("outroot", "./results", "Root directory to write model output")
flags.DEFINE_string("model_name", "", "Name of model class")
flags.DEFINE_string("data_name", "", "Name of dataset")
flags.DEFINE_string("preproc_name", "", "Name of dataset")
flags.DEFINE_string("experiment_name", "", "Name for experiment")
flags.DEFINE_string("submission_id", "", "UUID generated by bash script submitting job")
flags.DEFINE_string(
    "drug", "", "Compute OT map on drug, change outdir to outdir/drugs/drug"
)
flags.DEFINE_string("celldata", "", "Short cut to specify config.data.path & outdir")
flags.DEFINE_string("outdir", "", "Path to outdir")

FLAGS = flags.FLAGS


def name_expdir():
    """ Automatically generates the export directory for an experiment.

    Returns:
        An instance of Path containing the target export directory.
    """

    # Generates a format string for experiment name.
    experiment_name = FLAGS.experiment_name
    if len(FLAGS.drug) > 0:
        if len(experiment_name) > 0:
            experiment_name = f"{experiment_name}/drug-{FLAGS.drug}"
        else:
            experiment_name = f"drug-{FLAGS.drug}"

    # Uses predefined export directory if exists.
    if len(FLAGS.outdir) > 0:
        expdir = FLAGS.outdir

    else:
        expdir = os.path.join(
            FLAGS.outroot,
            FLAGS.data_name,
            FLAGS.preproc_name,
            experiment_name,
            f"model-{FLAGS.model_name}",
        )

    return Path(expdir)


def generate_random_string(n=8):
    """ Generates random strings from ASCII lowercase letters and digits. 
    
    Args:
        n: int, length of random strings to be generated.
            Default is 8.
    
    Returns:
        A random string with length n.
    """

    return "".join(random.choice(ALPHABET) for _ in range(n))


def write_config(path, config):
    """ Writes the configuration to a YAML file.

    Args:
        path: Path, the path where the YAML file is stored.
        config: ConfigDict or dict, the configuration to be stored.
    """

    if isinstance(config, ConfigDict):
        full = path.resolve().with_name("." + path.name)
        config.to_yaml(stream=open(full, "w"))
        config = config.to_dict()

    yaml.dump(config, open(path, "w"))
    return


def parse_config_cli(path, args):
    if isinstance(path, list):
        config = ConfigDict()
        for path in FLAGS.config:
            config.update(yaml.load(open(path), Loader=yaml.UnsafeLoader))
    else:
        config = load_config(path)

    opts = parse_cli_opts(args)
    config.update(opts)

    if len(FLAGS.celldata) > 0:
        config.data.path = str(FLAGS.celldata)
        config.data.type = "cell"
        config.data.source = "control"

    drug = FLAGS.drug
    if len(drug) > 0:
        config.data.target = drug

    return config


def prepare(argv):
    """ Prepares the experiment information including export directory and
        configuration.
        
    Args:
        argv: configuration to be loaded.
    
    Returns:
        A ConfigDict object containing the configuration information and 
        a Path object for exporting results.
    """
    
    _, *unparsed = flags.FLAGS(argv, known_only=True)

    if len(FLAGS.celldata) > 0:
        celldata = Path(FLAGS.celldata)

        if len(FLAGS.data_name) == 0:
            FLAGS.data_name = str(celldata.parent.relative_to("datasets"))

        if len(FLAGS.preproc_name) == 0:
            FLAGS.preproc_name = celldata.stem

    if FLAGS.submission_id == "":
        FLAGS.submission_id = generate_random_string()

    if FLAGS.config is not None or len(FLAGS.config) > 0:
        config = parse_config_cli(FLAGS.config, unparsed)
        if len(FLAGS.model_name) == 0:
            FLAGS.model_name = config.model.name

    outdir = name_expdir()

    if FLAGS.config is None or FLAGS.config == "":
        FLAGS.config = str(outdir / "config.yaml")
        config = parse_config_cli(FLAGS.config, unparsed)

    return config, outdir