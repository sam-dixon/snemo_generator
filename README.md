# Roman Simulations with SNEMO

This repository contains all of the necessary code to take a survey simulation file from David Rubin (repo to be shared soon) to a simulated supernova survey using SNEMO as the underlying SED model. Comments, issues, and pull requests are welcome!

## Installation
Setup a conda environment and activate it with
```
conda env create -f environment.yml
conda activate snemo_gen
```

Then install the package with 
```
python setup.py install
```

## Data files
[An example input simulation file can be found here](https://berkeley.box.com/s/gyhihojco0eh6i2trsc19mvh9kbkopwm). You must download this file and move it to the `data` directory to be able to run `roman_sims.py`.

The KDE files and extended SNEMO models are included in `snemo_gen/data` and are automatically included when installing the `snemo_gen` package.

## Running the code 
There are three scripts to reproduce this work:

1. `generate_KDE.py`: Models the underlying distribution of the SNEMO coefficients
2. `v0_extension.py`: Extends the wavelength range of the model
3. `roman_sims.py`: Creates light-curves and spectra using the signal-to-noise ratios produced in David's code

Steps 1 and 2 only need to be run once. Step 3 can then be rerun for any new survey simulation file. The outputs of steps 1 and 2 (i.e. the KDE files and extended SNEMO templates) can be used for other analyses using the `snemo_gen` package.

## Using the KDE files
The KDE files model the distribution of coefficients for each SNEMO version. There are some tricky parts to using these files because the KDEs are fit in a rotated and rescaled space and we estimate the distribution of the absolute magnitudes rather than the $c_0$ scaling parameters directly. The KDE files that are generated with `generate_KDE.py` are a list of 4 elements:

1. `kde`: the trained scikit-learn `KernelDensity` estimator
2. `v`: unitary matrix used in the whitening transformation applied to the data before fitting the KDE
3. `l`: singular values of the SVD transformation
4. `grid`: the scikit-learn `GridSearchCV` object produced in the hyperparameter determination

Sampling directly from the KDE gives a vector of coefficients in the rotated and rescaled coefficient space. Applying the inverse transform to the samples gives the coefficient vector:

```
unscaled_samples = kde.sample(n_samples)
samples = (np.diag(1./np.sqrt(l)) @ vinv @ unscaled_samples.T).T
```

This vector gives the values of the following coefficients (note the order):

* $M_B + 19.1$: the peak absolute Bessell $B$-band magnitude, plus an arbitrary offset of 19.1
* $A_s$: the color parameter
* $c_i$, $i>0$: the SNEMO coefficients (not including $c_0$, since this is captured by $M_B + 19.1$)

In order to produce an `sncosmo.Model` object with coefficients corresponding to a sampled vector, we would use:

```
sample = samples[i] # Select one sample from the samples produced above

# This would use the components released with Saunders et al. 2018
# See https://snfactory.lbl.gov/snemo/
# Below section details how to use the extended wavelength version
model = sncosmo.Model(source='snemo15')

# Set the redshift and time of max
model.set(z=z, t0=t0)

# Set all coefficients *except* the scaling
param_dict = dict(zip(model.source.param_names[1:], sample[1:]))
model.set(**param_dict)

# Set the scaling parameter
model.set_source_peakabsmag(sample[0]-19.1, 'bessellb', 'ab')
```

This functionality is wrapped in `snemo_gen.kde.sample_snemo_kde` and `snemo_gen.kde.MB_to_c0`.

## Using the extended SNEMO templates
The extended SNEMO templates produced by `v0_extension.py` can be used in `sncosmo` just like any other templates. We can create a source object and use that source within a `Model` with

```
snemo_source = sncosmo.models.SNEMOSource('extended_models/ext_snemo15.dat')
snemo_ext = sncosmo.Model(source=snemo_source)
```
You can replace `ext_snemo15.dat` with `ext_snemo7.dat` or `ext_snemo2.dat` depending on which version of SNEMO you'd like to use.

## Using the Roman simulation files
`roman_sim.py` produces one pickle file for each object in the survey in a subdirectory of `data` with the same name as the input file. Each of these pickle files contains a dictionary with the following keys:

* `z`: (`float`) redshift of object
* `t0`: (`float`) time of maximum
* `coef_orig`: (`np.array` of length 16) describing model coefficients
* `lc_time`: (`np.array` of length `n`) observer-frame times that the light curve was observed
* `true_lc_flux`: (`np.array` of length `n`) true flux in the light curve at each observation
* `lc_flux`: (`np.array` of length `n`) observed flux in the light curve at each observation
* `lc_flux_err`: (`np.array` of length `n`) error in observed flux
* `band`: (`np.array` of length `n`) bandpasses of the observations
* `zp`: (`np.array` of length `n`) always set to 25.
* `zpsys`: (`np.array` of length `n`) AB magnitudes are used
* `prism_wave`: (`np.array` of length `w`) observer-frame wavelength of the prism
* `prism_ts_time`: (`np.array` of length `t`) observer-frame times that a prism spectrum was taken
* `prism_ts_true_flux`: (`t x w np.array`) true flux from the prism at each time
* `prism_ts_flux`: (`t x w np.array`) observed flux from the prism at each time
* `prism_ts_flux_err`: (`t x w np.array`) error on observed flux
* `stacked_prism_time`: (`float`) time of observation of the single stacked near-max prism
* `stacked_prism_true_flux`: (`array` of length w) true flux
* `stacked_prism_flux`: (`array` of length w) observed, noisy flux
* `stacked_prism_flux_err`: (`array` of length w) error on observed flux

**Note**: `coef_orig` is the vector sampled from the KDE, not the `sncosmo` model parameter vector. The only difference is in the scaling parameter; the `sncosmo` model parameter vector is equivalent to the $c_0$ parameter from Saunders et al. 2018, where the sample from the KDE is the absolute B-band magnitude of the object zeroed by the mean (i.e. MB-19.1).
