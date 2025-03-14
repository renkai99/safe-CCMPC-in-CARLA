import argparse
import os

def dir_path(s):
    """Directory path type for argparse"""
    if os.path.isdir(s):
        return s
    else:
        raise argparse.ArgumentTypeError(
                f"readable_dir:{s} is not a valid path")

def file_path(s):
    """Directory path type for argparse"""
    if os.path.isfile(s):
        return s
    else:
        raise argparse.ArgumentTypeError(
                f"{s} is not an existing file path")

def str_kv(kv):
    """Used to identify and convert key=value arguments into a tuple (key, value). Value stays as a str.
    This is to be passed as the type when calling argparse.ArgumentParser.add_argument()

    Parameters
    ----------
    kv: str
        a key=value argument
    
    Returns
    -------
    tuple
        (key, value) from key=value
    """
    try:
        k, v = kv.split("=")
    except:
        raise argparse.ArgumentTypeError(f"value {kv} is not separated by one '='")
    return (k, v)

def int_kv(kv):
    """Used to identify and convert key=value arguments into a tuple (key, int(value)). Value is converted into an int.
    This is to be passed as the type when calling argparse.ArgumentParser.add_argument()
    
    Parameters
    ----------
    kv: str
        a key=value argument
    
    Returns
    -------
    tuple
        (key, int(value)) from key=value
    """
    try:
        k, v = kv.split("=")
    except:
        raise argparse.ArgumentTypeError(f"value {kv} is not separated by one '='")
    try:
        v = int(v)
    except:
        raise argparse.ArgumentTypeError(f"right side of {kv} should be int")
    return (k, v)

KV_TYPES = (str_kv, int_kv)

def make_dict(ll):
    return {k: v for (k, v) in ll}

def make_merge_dict(ll):
    d = {}
    for k, v in ll:
        if k in d:
            d[k].add(v)
        else:
            d[k] = set([v])
    return d

class ParseKVToDictAction(argparse.Action):
    """Parse argument with nargs '+' of type str_kv or int_kv into a dict."""
    def __init__(self, option_strings, dest, nargs=None, type=None, **kwargs):
        if nargs != '+':
            raise argparse.ArgumentTypeError(f"ParseKVToDictAction can only be used for arguments with nargs='+' but instead we have nargs={nargs}")
        if type not in KV_TYPES:
            raise argparse.ArgumentTypeError(f"ParseKVToDictAction can only be used for arguments with type=dict_str_vals or type=dict_num_vals but instead we have type={type}")
        super(ParseKVToDictAction, self).__init__(option_strings, dest,
                nargs=nargs, type=type, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, make_dict(values))

class ParseKVToMergeDictAction(argparse.Action):
    """Parse argument with nargs '+' of type str_kv or int_kv into a dict."""
    def __init__(self, option_strings, dest, nargs=None, type=None, **kwargs):
        if nargs != '+':
            raise argparse.ArgumentTypeError(f"ParseKVToDictAction can only be used for arguments with nargs='+' but instead we have nargs={nargs}")
        if type not in KV_TYPES:
            raise argparse.ArgumentTypeError(f"ParseKVToDictAction can only be used for arguments with type=dict_str_vals or type=dict_num_vals but instead we have type={type}")
        super(ParseKVToMergeDictAction, self).__init__(option_strings, dest,
                nargs=nargs, type=type, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, make_merge_dict(values))
