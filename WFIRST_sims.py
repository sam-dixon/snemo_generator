"""Generates a simulated supernova survey using SNEMO as the underlying SED model from a survey
information file
"""
import os
import copy
import pickle
import sncosmo
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_and_register(path, name=None):
    """Read and register non-built-in bandpasses into the sncosmo registry
    
    Args:
        path: path to filter definition
        name: name of bandpass. If None, uses the filename.
    """
    if name is None:
        name = os.path.basename(path).split('.')[0]
    wave, trans = np.loadtxt(path, unpack=True)
    band = sncosmo.Bandpass(wave=wave, trans=trans, name=name)
    sncosmo.registry.register(band)

# Load in WFIRST filters
for f in os.listdir('data/filts/'):
    read_and_register(os.path.join('data/filts/', f))

# Load prism information
prism_info_path = 'data/prism_info.pkl'
prism_info = pickle.load(open(prism_info_path, 'rb'))
prism_wave = prism_info[0.125][1]


def sample_snemo_kde(n, model):
    """Generate samples from a KDE of the SNEMO parameters.

    Args:
        n: Number of parameter samples
        model: sncosmo.Model object to use

    Returns:
        list of dictionaries of SNEMO parameters
    """
    kde_path = 'data/{}_KDE_published.pkl'.format(model.source.name)
    param_names = ['MB', *model.source.param_names[1:]]
    with open(kde_path, 'rb') as f:
        KDE, rot, eigenvals, grid = pickle.load(f)
    unscaled_samples = KDE.sample(n, random_state=0)
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
    model.set_source_peakabsmag(param_dict['MB']-19.1, 'bessellb', 'ab')
    param_dict_with_c0 = dict(zip(model.param_names, model.parameters))
    return param_dict_with_c0


def lc_obs_info(obs):
    """Convert the observation information in David's dictionary to one that will work with realize_lcs
    
    Args:
        obs: SN_observations like dict
        
    Returns:
        snemo_obs: dict with required keys for realize_lcs
    """
    out = {}
    # convert from python 2 >:(
    filts = [x.decode('utf-8') for x in obs['filts']]
    # add 'lsst'
    out['filts'] = ['lsst'+x if len(x)==1 else x for x in filts]
    out['times'] = obs['dates']
    out['snr'] = obs['fluxes']/obs['dfluxes']
    return out


def prism_obs_info(obs, z, t0, stacked=False):
    """Convert the observation information in David's dictionary to one that will work with realize_spectra
    
    Args:
        obs: SN_observations like dict
        stacked: whether or not to use the stacked spectrum
        
    Returns:
        snemo_obs: dict with required keys for realize_spectra
    """
    out = {}
    z = np.round_(z, 3)
    out['wave'] = prism_wave
    if stacked:
        out['times'] = np.array([t0])
        out['snr'] = np.sqrt(np.sum(prism_info[z][0]**2, axis=0)).reshape(1, len(prism_wave))
    else:
        out['times'] = t0 + (prism_info[z][2]-300)
        out['snr'] = prism_info[z][0]
    return out


def realize_spectra(observations, model, params, zs, t0s, silent=False):
    """Realize spectra from a set of observations.

    Args:
        observations: list of dicts of observation info, each containing keys `wave`, `times`, `snr`.
        model: Model to simulate with
        params: List of parameters to feed to the model.
        zs: List of redshifts
        t0s: List of times-of-max

    Returns:
        list of dicts of realized data for each item in params.
        
        Each dict has the following keys:
            z (float): redshift
            t0 (float): time of maximum brightness
            coefs_orig (list, length=16): original SNEMO coefficients
            coefs_orig_dict (dict): SNEMO coefficient dictionary
            wave (w x 1 np.array): observer frame wavelength of the observations
            time (t x 1 np.array): observer frame times that the object was observed
            true_flux (t x w np.array): true flux in erg/s/cm^2/AA at each time the object was observed
            flux (t x w np.array): noised flux of the object
            flux_err (t x w np.array): error on the observed flux
    """
    survey_data = []
    if len(observations)==1:
        observations = observations*len(params)
    for i, obs in tqdm(enumerate(observations), total=len(params), disable=silent):
        wave = obs['wave']
        times = obs['times']
        snr = obs['snr']
        assert snr.shape == (*times.shape, *wave.shape)
        z = zs[i]
        t0 = t0s[i]
        par = params[i]
        data = {'z': z,
                't0': t0,
                'coefs_orig': list(par.values()),
                'coefs_orig_dict': par}
        model.set(z=z, t0=t0)
        for param_name in model.source.param_names[1:]:
            model.set(**{param_name: params[i][param_name]})
        model.set_source_peakabsmag(params[i]['MB']-19.1, 'bessellb', 'ab')
        flux = model.flux(wave=wave, time=times)
        dflux = np.abs(flux/snr)
        noise = np.random.randn(*flux.shape)*dflux
        data['time'] = times
        data['true_flux'] = flux
        data['flux_err'] = dflux
        data['flux'] = flux + noise
        data['wave'] = wave
        survey_data.append(data)
    return survey_data


def realize_lcs(observations, model, params, zs, t0s, silent=True):
    """Realize light curves from a set of observations.
    
    Args:
        observations: list of dicts with observation info. Required keys: `times`, `filts`, `snr`
        model: Model to simulate with
        params: List of parameters to feed the model
        zs: List of redshifts
        t0s: List of times-of-max
        
    Returns:
        list of dicts of realized data. Each dict is an sncosmo compatible light curve data structure.
        
        Each dict has the following keys:
            z (float): redshift
            t0 (float): time of maximum brightness
            coefs_orig (list, length=16): original SNEMO coefficients
            coefs_orig_dict (dict): SNEMO coefficient dictionary
            time (t x 1 np.array): observer frame times that the object was observed
            true_flux (t x 1 np.array): true flux in erg/s/cm^2/AA at each time the object was observed
            flux (t x 1 np.array): noised flux of the object
            flux_err (t x 1 np.array): error on the observed flux
            band (t x 1 np.array): bandpasses used
            zp: zp of 25 used
            zpsys: AB magnitude system
    """
    survey_data = []
    if len(observations)==1:
        observations = observations*len(params)
    for i, obs in tqdm(enumerate(observations), total=len(params), disable=silent):
        bands = obs['filts']
        times = obs['times']
        snr = obs['snr']
        z = zs[i]
        t0 = t0s[i]
        par = params[i]
        data = {'z': z,
                't0': t0,
                'coefs_orig': list(par.values()),
                'coefs_orig_dict': par}
        model.set(z=z, t0=t0)
        for param_name in model.source.param_names[1:]:
            model.set(**{param_name:  params[i][param_name]})
        model.set_source_peakabsmag(params[i]['MB']-19.1, 'bessellb', 'ab')
        flux = model.bandflux(band=bands, time=times, zp=25, zpsys='ab')
        dflux = np.abs(flux/snr)
        noise = np.random.randn(*flux.shape)*dflux
        data['lc_time'] = times
        data['true_lc_flux'] = flux
        data['lc_flux_err'] = dflux
        data['lc_flux'] = flux + noise
        data['band'] = bands
        data['zp'] = [25.] * len(flux)
        data['zpsys'] = ['ab'] * len(flux)
        survey_data.append(data)
    return survey_data


def main(survey_path, model_name):
    survey = pickle.load(open(survey_path, 'rb'), encoding='latin-1')
    
    print('Loading model')
    if 'ext' in model_name:
        model_path = 'data/extended_models/{}.dat'.format(model_name)
        model = sncosmo.Model(source=sncosmo.models.SNEMOSource(model_path, name=model_name[4:]))
    else:
        model = sncosmo.Model(source=model_name)
    params = sample_snemo_kde(len(survey['SN_observations']), model)
    
    zs = survey['SN_table']['redshifts']
    daymaxes = survey['SN_table']['daymaxes']

    print('Generating observations...')
    survey_with_prism_spectra = []
    for i, obs in tqdm(enumerate(survey['SN_observations']), total=len(survey['SN_observations'])):
        lc_obs = lc_obs_info(obs)
        sn = realize_lcs([lc_obs],
                         model,
                         [params[i]],
                         [zs[i]],
                         [daymaxes[i]],
                         silent=True)[0]
        if np.round_(zs[i], 3) in prism_info.keys():
            ts_obs = prism_obs_info(obs, zs[i], daymaxes[i], stacked=False)
            stack_obs = prism_obs_info(obs, zs[i], daymaxes[i], stacked=True)
            ts = realize_spectra([ts_obs],
                                 model,
                                 [params[i]],
                                 [zs[i]],
                                 [daymaxes[i]],
                                 silent=True)[0]
            stack = realize_spectra([stack_obs],
                                    model,
                                    [params[i]],
                                    [zs[i]],
                                    [daymaxes[i]],
                                    silent=True)[0]
            sn['prism_wave'] = ts['wave']
            sn['prism_ts_time'] = ts['time']
            sn['prism_ts_true_flux'] = ts['true_flux']
            sn['prism_ts_flux'] = ts['flux']
            sn['prism_ts_flux_err'] = ts['flux_err']
            sn['stacked_prism_time'] = stack['time'][0]
            sn['stacked_prism_true_flux'] = stack['true_flux'][0]
            sn['stacked_prism_flux'] = stack['flux'][0]
            sn['stacked_prism_flux_err'] = stack['flux_err'][0]
        survey_with_prism_spectra.append(sn)
        
    print('Saving...')
    save_path = 'data/{}_{}'.format(model_name, survey_path.split('/')[-1].split('.')[0])
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for i, sn in tqdm(enumerate(survey_with_prism_spectra), total=len(survey_with_prism_spectra)):
        fname = '{:04d}.pkl'.format(i)
        path = os.path.join(save_path, fname)
        with open(path, 'wb') as f:
            pickle.dump(sn, f)


if __name__=='__main__':
    survey_path = 'data/1hr_per_pointing_imaging+prism.pkl'
    model_name = 'ext_snemo7'
    main(survey_path, model_name)


