"""Writes SNEMO component files that extend the wavelength coverage to the UV and IR using the Hsiao template
"""

import os
import sncosmo
import numpy as np
from snemo_gen import DATADIR
from scipy.interpolate import RectBivariateSpline as spl


OUTPATH = os.path.join(DATADIR, 'extended_models')


def main(model_name, output_dir=OUTPATH):
    # Load SNEMO model
    snemo = sncosmo.get_source(model_name)
    WAVES = snemo._wave
    PHASES = snemo._phase
    fluxes = np.array([flux(PHASES, WAVES) for flux in snemo._model_fluxes]).T
    
    # Load Hsiao template
    hsiao = sncosmo.Model('hsiao')
    
    # Define extended wavelengths
    EXTWAVE = np.concatenate([np.arange(1000, WAVES[0], 2),
                              WAVES,
                              np.arange(np.around(WAVES[-1], -1), 20010, 2)])
    
    # Extend mean spectral timeseries (SNEMO C0)
    uv_waves = EXTWAVE[EXTWAVE<WAVES[0]]
    ir_waves = EXTWAVE[EXTWAVE>WAVES[-1]]
    opt_waves = EXTWAVE[(EXTWAVE>=WAVES[0]) & (EXTWAVE<=WAVES[-1])]
    
    uv_join_wave = WAVES[0]
    ir_join_wave = WAVES[-1]
    uv_scale = fluxes[0, :, 0]/(hsiao.flux(time=PHASES, wave=uv_join_wave).T)
    ir_scale = fluxes[-1, :, 0]/(hsiao.flux(time=PHASES, wave=ir_join_wave).T)
    
    ext_flux0 = np.concatenate([uv_scale * hsiao.flux(time=PHASES, wave=uv_waves).T,
                                fluxes[:, :, 0],
                                ir_scale * hsiao.flux(time=PHASES, wave=ir_waves).T])

    # Apodize additional components
    uv_join_end = 3500 # location (in Angstroms) to stop smoothing in the UV
    ir_join_start = 8400 # location (in Angstroms) to start smoothing in the IR
    
    def apodize(x):
        if x < uv_join_end:
            return 1/(uv_join_end-uv_join_wave)*(x-uv_join_wave)
        elif x > ir_join_start:
            return 1 - 1/(ir_join_wave-ir_join_start)*(x-ir_join_start)
        else:
            return 1

    window = np.array(list(map(apodize, WAVES)))
    apodized_comps = (window * fluxes[:, :, 1:].T).T
    
    ext_flux1 = np.concatenate([np.zeros((len(uv_waves), *apodized_comps.shape[1:])),
                                    apodized_comps,
                                    np.zeros((len(ir_waves), *apodized_comps.shape[1:]))])
    
    # Reshape
    ext_flux0 = ext_flux0.reshape((*ext_flux0.shape, 1))
    ext_flux = np.concatenate([ext_flux0, ext_flux1], axis=-1)
    
    # Rebin
    minwave = 1000
    maxwave = 20000
    velocity = 1000
    nbins = int(np.log10(maxwave/minwave) / np.log10(1 + velocity/3.e5) + 1)
    velspace_EXTWAVE = np.geomspace(1000, 20000, nbins)
    
    flux_spls = [spl(PHASES, EXTWAVE, f) for f in ext_flux.T]
    velspace_flux = np.array([s(PHASES, velspace_EXTWAVE) for s in flux_spls]).T
    
    # Write to file
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    out = []
    for i, phase in enumerate(PHASES):
        for j, wave in enumerate(velspace_EXTWAVE):
            out.append([phase, wave, *velspace_flux[j, i]])
    np.savetxt(os.path.join(output_dir, 'ext_{}.dat'.format(snemo.name)), out)

    
if __name__=='__main__':
    for ncomp in [2, 7, 15]:
        main('snemo{}'.format(ncomp))