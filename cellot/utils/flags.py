""" Helper functions regarding flags.
"""


from pathlib import Path
import json
from absl import logging, flags
import re

FLAGS = flags.FLAGS


def write_flagfile(path, cfg=None, include_json=False, mode='w', **kwargs):
    ''' Stores absl.flags to a flagfile.
    
    Args:
        path: str, filepath where the flags are to be stored.
        cfg: absl.FlagValues, configuration to be stored. Default is None.
        include_json: bool, whether to store another json file. Default is False.
        mode: str, the work mode. Default is 'w'.
        **kwargs: additional arguments.
    '''

    # Syncs configurations.
    if cfg is None:
        cfg = FLAGS

    flag_list_pp = collect_serializable_flags(cfg, **kwargs)
    # Under dry mode, doesn't write.
    if hasattr(cfg, 'dry') and cfg.dry:
        logging.info('Dry mode -- will not write %s', str(path))
        return

    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Writes flagfile to path
    with open(path, mode) as fout:
        sorted_items = sorted(flag_list_pp.items(), key=rank_flag_module)
        for key, srl_list in sorted_items:
            fout.write(f'# {key}\n')
            fout.write('\n'.join(srl_list) + '\n')
            fout.write('\n')

    logging.info('Wrote flags to: %s', str(path))

    if include_json:
        lut = flags_to_json(cfg, **kwargs)
        with open(path.with_name(f'.{path.name}.json'), 'w') as fp:
            json.dump(lut, fp)
    return


def collect_serializable_flags(flags:flags.FlagValues, keyflags=False):
    """ Collects all serializable flags. 
    
    Args:
        flags: flags.FlagValues containing the configuration information.
        keyflags: bool, whether to only return key flags. Default is False.
    
    A serializable object is able to be transformed into a stream of bytes, so that
    recovering the identical object from the stream is possible.  

    Returns:
        A dict containing all serializable flags.
    """

    flag_list_pp = dict()
    # Gets the flags' info by dict.
    flagdict = get_flagdict(flags, keyflags)

    for key, flag_list in flagdict.items():
        flag_list_pp_list = list()
        for fl in flag_list:

            # Only writes absl flags if present.
            # fl.present is True if this flag 
            # was parsed from command line flags.
            if key.startswith('absl'):
                if not fl.present:
                    continue

            # srl is a flag as it appeared on cmdline
            # e.g. "--arg=value"
            srl = fl.serialize()
            if len(srl) > 1:  # (non-bool) default flags are empty
                flag_list_pp_list.append(srl)

        if len(flag_list_pp_list) > 0:
            flag_list_pp[key] = flag_list_pp_list
    return flag_list_pp


def get_flagdict(cfg:flags.FlagValues, keyflags=False) -> dict:
    """ Returns the dict of module_name -> list of defined flags. 
    
    Args:
        cfg: flags.FlagValues containing the configuration information.
        keyflags: bool, whether to only return key flags (flags one defines in the code). 
            Default is False.
        
    Returns:
        A dict, with keys being module names (str), and values being lists
        of Flag objects.
    """
    
    if keyflags:
        return cfg.key_flags_by_module_dict()

    return cfg.flags_by_module_dict()


def rank_flag_module(item):
    ''' Sorts tensorflow and absl flags to the last.

    Args:
        item: a key-value pair.

    Returns:
        A tuple of the key and its rank information. 
    '''

    key, _ = item
    if key.startswith('absl'):
        rank = 100
    elif key.startswith('tensorflow'):
        rank = 10
    else:
        rank = 0
    return (rank, key)


def flags_to_json(cfg, blacklist=None, keyflags=False):
    """ Puts flags into a dict to be stored by json format.

    Args:
        cfg: flags.FlagValues containing the configuration information.
        blacklist: list specifying which flags NOT to be stored. Default is None.
        keyflags: bool, whether to only return key flags. Default is False.
    
    Returns:
        A dict with its keys being flags' names and its values being flags' values.
    """

    # Default blacklist is tensorflow and absl relating configs.
    if blacklist is None:
        blacklist = [r'^tensorflow.*', r'^absl.*']
    blacklist = [re.compile(x) for x in blacklist]

    flagdict = get_flagdict(cfg, keyflags)

    # Look-up table.
    lut = dict()
    for key, flag_list in flagdict.items():
        # Skips tensorflow and absl module.
        if key.startswith('tensorflow.') or key.startswith('absl.'):
            continue

        for arg in flag_list:
            # Returns parsed flag value as string.
            if isinstance(arg.value, list):
                val = arg._get_parsed_value_as_string(arg.value)
                val = val.lstrip("'").rstrip("'")

            else:
                val = arg.value

            lut[arg.name] = val

    return lut
