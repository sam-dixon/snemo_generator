import os
import copy
import pickle
import numpy as np
from snemo_gen import DATADIR


def sample_snemo_kde(n, model, random_state=None):
    """Generate samples from a KDE of the SNEMO parameters.

    Args:
        n: Number of parameter samples
        model: sncosmo.Model object to use

    Returns:
        list of dictionaries of SNEMO parameters
    """
    kde_fname = '{}_KDE_published.pkl'.format(model.source.name)
    kde_path = os.path.join(DATADIR, kde_fname)
    param_names = ['MB', *model.source.param_names[1:]]
    with open(kde_path, 'rb') as f:
        KDE, rot, eigenvals, grid = pickle.load(f)
    unscaled_samples = KDE.sample(n, random_state=random_state)
    samples = (rot @ np.diag(np.sqrt(eigenvals)) @ unscaled_samples.T).T
    param_dict = [dict(zip(param_names, samp)) for samp in samples]
    return param_dict


def MB_to_c0(param_dict, model, z, t0):
    """Calculate the value of c0 given the other model parameters, MB and the redshift

    Args:
        param_dict: dictionary of the model parameters sampled from the KDE
        model: sncosmo.Model object to use
        z: Redshift of the object

    Returns:
        param_dict_with_c0: dictionary of all the model parameters, including the correct c0, that
                            are needed to use the sncosmo simulation tools (i.e. sncosmo.realize_lcs)
    """
    model = copy.copy(model)
    model.set(z=z, t0=t0)
    for param_name in model.source.param_names[1:]:
        model.set(**{param_name: param_dict[param_name]})
    model.set_source_peakabsmag(param_dict['MB'] - 19.1, 'bessellb', 'ab')
    param_dict_with_c0 = dict(zip(model.param_names, model.parameters))
    return param_dict_with_c0