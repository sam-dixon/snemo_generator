"""Train Kernel Density Estimates of the distribution of SNEMO coefficients
"""
import os
import copy
import pickle
import sncosmo
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


FITS = pd.read_csv('data/anon_snf_fits_published_snemo_and_salt.csv', index_col=0)


def get_data(model_name):
    """Gathers all the SNf fit data for a given model
    
    Args:
        model_name: Name of SED model (one of 'salt2', 'snemo2', 'snemo7', or 'snemo15')
    
    Returns:
        df: pandas.DataFrame containing model coefficients
    """
    model = sncosmo.Model(source=model_name)
    df = pd.DataFrame()
    for i, param_name in enumerate(model.source.param_names):
        df[param_name] = FITS['{}_{}'.format(model_name, param_name)]
    df['MB'] = FITS[model_name+'_MB191']
    df['z'] = FITS['z_helio']
    return df


def calc_kde(model_name):
    """Calculates an estimate of the probability distribution for the coefficients
    of the given model based on the observed distribution of coefficients in the SNf
    sample.
    
    Args:
        model_name: Name of SED model (one of 'salt2', 'snemo2', 'snemo7', or 'snemo15')
        
    Returns:
        kde: trained scikit-learn KDE object with bandwidth optimized through grid-search
             cross-validation
        v: Unitary matrix from SVD of coefficient covariance matrix
        l: Vector of singular values from SVD of coefficent cov. mat.
        grid: scikit-learn GridSearchCV object with history of cross-validation
    """
    df = get_data(model_name)
    coefs = df[['MB', 'As']+['c'+str(i) for i in range(1, int(model_name[5:]))]].values
    v, l, vinv = np.linalg.svd(np.cov(coefs.T))
    rescaled_coefs = (np.diag(1./np.sqrt(l)) @ vinv @ coefs.T).T
    
    params = {'bandwidth': np.logspace(-1, 1, 100)}
    grid = GridSearchCV(KernelDensity(), params, cv=3)
    grid.fit(rescaled_coefs)
    kde = grid.best_estimator_
    return kde, v, l, grid


def main():
    for model_name in ['SNEMO2', 'SNEMO7', 'SNEMO15']:
        print('Fitting KDE for ' + model_name)
        kde, v, l, grid = calc_kde(model_name)
        pickle.dump(calc_kde(model_name), open('data/{}_KDE_published.pkl'.format(model_name), 'wb'))

        
if __name__=='__main__':
    main()

    