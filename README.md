# MF-component-separation (`mfcs`)

Multi-frequency component separation at the power-spectrum level for
cross-correlations between large-scale structure (LSS) tracers and
multi-frequency sky maps (e.g. Planck HFI/LFI).

The `mfcs` package implements two methods:

1. **Maximum-likelihood bandpower reconstruction (MLBR)**
   (`mfcs.MaxLikeBandpowerReconstructor`). It takes the cross-power spectra
   between a set of LSS tracers and a set of frequency maps and recovers the
   cross-spectrum between each LSS tracer and each sky component (e.g. tSZ and
   CIB), with their full covariance. It was introduced in
   [La Posta et al. (arXiv:2603.04269)](https://arxiv.org/abs/2603.04269) and
   used in [Wayland et al. (arXiv:2609.31774)](https://arxiv.org/abs/2609.31774)
   to measure tSZ and CIB cross-correlations with cosmic shear.
2. **Large-scale cross-correlation tomography**
   (`mfcs.LargeScaleTomographer`). It reconstructs the bias-weighted mean of a
   sky component, $\langle bU\rangle(z)$ (e.g. the electron pressure
   $\langle bP_e\rangle$ from tSZ, or the star-formation rate density
   $\langle b\rho_{\rm SFR}\rangle$ from CIB), as a function of redshift. It
   jointly fits galaxy auto-correlations and galaxy-map cross-correlations on
   large scales, using a model that does not depend on the small-scale galaxy
   bias. The method is presented in
   [La Posta et al. (arXiv:2603.04269)](https://arxiv.org/abs/2603.04269) and
   builds on the methodology of
   [Maleubre et al. (arXiv:2508.05319)](https://arxiv.org/abs/2508.05319).

The package also provides SED utilities (blackbody, CMB and tSZ SEDs,
bandpass convolution) and an implementation of the Béthermin et al. SIDES
CIB SED model.

---

## Installation

Clone the repository and install it with `pip`:

```bash
git clone https://github.com/amy-wayland/MF-component-separation.git
cd MF-component-separation
pip install .
```

Use `pip install -e .` for an editable (development) install. To also install
the packages needed to run the example notebooks, use
`pip install ".[examples]"`.

`pyccl` contains compiled code. If installing it with `pip` fails, install it
with conda first (`conda install -c conda-forge pyccl`) and then run
`pip install .`.

Then, in Python:

```python
import mfcs
```

The data files in `data/` (Planck bandpasses and the SIDES SED library) are
not installed with the package. The example notebooks load them with paths
relative to the repository root, so run the notebooks from there.

### Dependencies

| Package  | Used for                                                      |
|----------|---------------------------------------------------------------|
| `numpy`, `scipy` | Everything                                             |
| [`sacc`](https://github.com/LSSTDESC/sacc)  | Input data vectors, covariances, tracers and bandpower windows |
| [`pyccl`](https://github.com/LSSTDESC/CCL) | Theory templates in `LargeScaleTomographer` |
| [`emcee`](https://emcee.readthedocs.io)     | Bias posterior sampling in `LargeScaleTomographer.get_linear_biases` |
| `healpy`, `matplotlib` | Used only in the example notebooks (beams, pixel windows, plots); installed with the `examples` extra |

`pip install .` installs all the packages above except `healpy` and
`matplotlib`. The code requires Python ≥ 3.10 and numpy ≥ 2.0 (it uses
`np.trapezoid`). It has been tested with numpy 2.5, scipy 1.18, sacc 2.4,
pyccl 3.3 and emcee 3.1.

---

## Package contents

```
mfcs/
├── mlbr.py        # MaxLikeBandpowerReconstructor
├── lstomo.py      # LargeScaleTomographer
├── sed_utils.py   # BnuBB, dBnudlogT, CMB_sed, tSZ_sed, BPass
├── SIDES_sed.py   # SIDESBethermin (CIB SED model)
└── utils.py       # internal helpers
data/
├── planck_bandpass/  # Planck LFI/HFI bandpasses (from pyilc)
└── SIDES_SED/        # SIDES SED library (from the SIDES public release)
```

See [`data/README.md`](data/README.md) for the provenance of the data files.

---

## Input data

Both main classes take a [`sacc.Sacc`](https://sacc.readthedocs.io) object
containing:

* **Tracers** for the LSS samples and the frequency maps, referred to by
  name. Galaxy tracers used by `LargeScaleTomographer` must be `NZ`
  tracers (with `z` and `nz` attributes). Map tracers can optionally carry
  a `beam` attribute (an array over the tracer's `ell`), which is used to
  forward-model the beam (see below).
* **Power spectra** for all LSS × map cross-correlations used (and, for
  `LargeScaleTomographer`, the galaxy auto-correlations). The data type is
  chosen from the spin of the field: `cl_00` for spin 0, `cl_0e` for spin 2
  (E-modes only), and `cl_0e`/`cl_0b` for `'e'`/`'b'`. Either tracer
  ordering is accepted.
* **Covariance**: a joint covariance matrix for all the power spectra.
* **Bandpower windows**: needed only by MLBR with
  `beam_correction_kind='bpw'`.

The sacc files used in the example notebooks are not included in this
repository. The notebooks load them from the parent directory (`../*.fits`).

### Units and conventions (`sed_utils`)

* Frequencies are in GHz, temperatures in K and intensities in MJy/sr.
* `BnuBB`, `dBnudlogT`, `CMB_sed` and `tSZ_sed` return intensities in MJy/sr.
  `tSZ_sed(nu)` is the intensity per unit Compton-$y$.
* `BPass.convolve(sed_func)` integrates an SED over the bandpass and returns
  it in K_CMB, i.e. the effective SED of that component in a CMB-temperature
  map. `fast=True` uses a coarser frequency sampling of the bandpass core.

---

## Method 1: maximum-likelihood bandpower reconstruction

### Model

For each LSS tracer $g$, the measured cross-spectrum with map $\nu$ is
modelled as a linear combination of the component cross-spectra:

$$
C_\ell^{g\nu} = B^\nu_\ell \sum_c S^{g}_{c}(\nu)\, C_\ell^{gc},
$$

where $S^{g}_c(\nu)$ is the SED of component $c$ in map $\nu$ and
$B^\nu_\ell$ is the beam. The SED may depend on $g$, for example for the
CIB, whose effective SED depends on the redshift kernel of the LSS tracer.
Stacking all the data into a vector $\mathbf{d}$ with covariance
$\mathsf{C}$, the model is $\mathbf{d} = \mathsf{S}\,\mathbf{c} + \mathbf{n}$.
The maximum-likelihood solution for the component bandpowers $\mathbf{c}$
and its covariance are

$$
\hat{\mathbf{c}} = \left(\mathsf{S}^T\mathsf{C}^{-1}\mathsf{S}\right)^{-1}
\mathsf{S}^T\mathsf{C}^{-1}\mathbf{d},\qquad
\mathrm{Cov}(\hat{\mathbf{c}}) = \left(\mathsf{S}^T\mathsf{C}^{-1}\mathsf{S}\right)^{-1}.
$$

### Usage

```python
c = mfcs.MaxLikeBandpowerReconstructor(
    s,                          # sacc.Sacc object
    lss_names=wl_names,         # LSS tracers
    map_names=planck_names,     # frequency maps
    comp_names=['tSZ', 'CIB'],  # components to separate
    seds=seds,                  # see below
    maps_in_x='all',            # or {lss_name: [map_names]}
    lmax=lmaxs,                 # scalar, or {map_name: lmax}
    lmin=30,
    lss_spin=2,                 # 0, 2, 'e' or 'b'
    beam_correction_kind='bpw') # 'mid', 'bpw' or 'none'

cls, cov = c.get_component_cls(get_cov=True)

ind = c.get_component_indices('KiDS1000__0', 'tSZ')
ells = c.common_ells['KiDS1000__0']
cl_tsz, err_tsz = cls[ind], np.sqrt(np.diag(cov)[ind])
```

**Arguments**

* `seds`: a dictionary of SEDs (already convolved with the bandpass and in
  the units of each map). Each component takes one of two forms:
  * `{comp: {map_name: value}}`: the same SED for all LSS tracers (e.g. tSZ);
  * `{comp: {lss_name: {map_name: value}}}`: a different SED for each LSS
    tracer (e.g. CIB, averaged over the tracer's redshift kernel).

  You can change the SEDs without reloading the data by calling
  `c.update_seds(new_seds)`. This is useful for marginalising over SED
  parameters.
* `maps_in_x`: which maps to cross-correlate with each LSS tracer.
* `lmax`: a per-map scale cut, so that noisy or contaminated channels can be
  cut at lower $\ell$ (e.g. 857 GHz).
* `beam_correction_kind`: how map beams are included in $\mathsf{S}$:
  * `'mid'`: the beam is evaluated at the effective $\ell$ of each bandpower;
  * `'bpw'`: the beam is averaged over the bandpower window functions in the
    sacc file;
  * `'none'`: no beam correction (e.g. if the spectra are already
    beam-deconvolved).

  Beams are interpolated from `tracer.ell`/`tracer.beam`, and taken to be 1
  below and 0 above the tabulated range.

**Output layout**: the component bandpowers are ordered by LSS tracer, then
by component, then by multipole. The multipoles for tracer `g` are
`c.common_ells[g]`, the union of the multipoles that survive the scale cuts
in any of that tracer's cross-correlations. Use
`get_component_indices(lss_name, comp_name)` to extract each block.

---

## Method 2: large-scale cross-correlation tomography

### Model

For each galaxy sample $g$ (treated as a narrow redshift bin), the galaxy
auto-spectrum and the cross-spectra with each map $\nu$ are modelled on large
scales as

$$
C_\ell^{gg} = \sum_t A^{gg}_t\, \mathcal{T}^{gg}_{t,\ell},\qquad
C_\ell^{g\nu} = B^\nu_\ell \sum_c S_c(\nu) \sum_t A^{gc}_t\, \mathcal{T}^{gc}_{t,\ell}.
$$

The templates $\mathcal{T}_t$ are computed with CCL for unit-bias galaxy
tracers and the user-supplied component tracers, using the following 3D
power spectra:

* `k0`: the non-linear matter power spectrum $P_{\rm NL}(k,z)$;
* `k2`, …, `k{2n_max}`: $(k/k_{\rm p})^{2n} P_{\rm NL}(k,z)$, with
  $k_{\rm p}=0.1\,{\rm Mpc}^{-1}$, to absorb scale-dependent bias;
* `N`: a constant (stochastic/shot-noise-like) term, which can be disabled
  with `include_stochasticity=False`.

All amplitudes $A$ enter linearly, so they are solved for in one step by
generalised least squares. The linear biases are then

$$
b_g = \sqrt{A^{gg}_{k0}},\qquad \langle b\,U_c\rangle_g = A^{gc}_{k0}/b_g,
$$

so the result does not depend on the galaxy bias or on the details of
small-scale clustering. Scale cuts are set by default by
$\ell_{\rm max} = k_{\rm max}\,\chi(\bar z_g)$.

### Usage

```python
tomo = mfcs.LargeScaleTomographer(
    s,
    gal_names=gal_names,          # galaxy NZ tracers
    map_names=planck_names,       # sky maps
    comp_names=['tSZ', 'CIB'],
    tracer_classes={'tSZ': ccl.tSZTracer,     # same tracer for all maps
                    'CIB': cib_tracers},      # {map_name: callable(cosmo)}
    seds={'tSZ': tsz_sed_dict,                # {map_name: value}
          'CIB': color_corr},                 # or None (=1)
    maps_in_x='all',
    n_max=1, kmax=0.3, lmin=30,
    map_spin=0)                   # scalar or {map_name: spin}

amps, cov_amps = tomo.amps, tomo.cov_amps   # amplitudes and covariance
chi2, ndof, pte = tomo.get_chi2()           # goodness of fit

bg, bU, samples, names = tomo.get_linear_biases(nsteps=10000, burnin=2000)
# bg[gal_name]             = (mean, std, best_fit)
# bU[(gal_name, comp_name)] = (mean, std, best_fit)
```

**Arguments**

* `tracer_classes`: for each component, either a callable `f(cosmo)` that
  returns a `pyccl.Tracer` (used for every map and scaled by that map's SED),
  or a dictionary `{map_name: f}` with a different tracer for each map (e.g.
  when the SED is redshift-dependent and so has to go into the radial
  kernel, as for the CIB, or for different lensing source samples).
* `seds`: `{comp: {map_name: value}}` or `{comp: None}`. Missing or `None`
  SEDs are set to 1. An SED is required for any component whose tracer is not
  given per map.
* `kmax` / `lmax` / `lmin`: scale cuts. If `kmax` is set, it overrides `lmax`.
* `cosmo`: a `pyccl.Cosmology` (default: `CosmologyVanillaLCDM`). Call
  `tomo.update_cosmo(cosmo)` to recompute the templates and amplitudes for a
  different cosmology.
* `beam_correction_kind`: how map beams are applied to the galaxy-map
  templates. The options are `'mid'` (default), `'bpw'` and `'none'`, and they
  mean the same as in `MaxLikeBandpowerReconstructor`. Use `'none'` if the
  spectra in the sacc file are already beam-deconvolved.
* `amp_indices`: maps amplitude names of the form `'{gal}_{gal}_{term}'` and
  `'{gal}_{comp}_{term}'` to their index in `tomo.amps`.

`get_linear_biases` samples the posterior of $(b_g, \langle bU_c\rangle_g)$
with `emcee`, using a Gaussian likelihood for the fitted `k0` amplitudes. This
handles the non-linear mapping from amplitudes to biases.

---

## Example notebooks

| Notebook | Method | What it does |
|----------|--------|--------------|
| [`tSZCompSepExample.ipynb`](tSZCompSepExample.ipynb) | MLBR | Separates the tSZ and CIB cross-correlations with KiDS-1000 cosmic shear (5 tomographic bins) from Planck NPIPE 100–857 GHz maps. Builds the tSZ SED and a per-bin CIB SED (SIDES model averaged over each lensing kernel) by convolving with the Planck bandpasses, and uses bandpower-window beam correction. |
| [`tSZCIBTomography.ipynb`](tSZCIBTomography.ipynb) | Tomography | Jointly reconstructs $\langle bP_e\rangle(z)$ and $\langle b\rho_{\rm SFR}\rangle(z)$ from DESI BGS and LRG galaxies cross-correlated with Planck NPIPE maps. The CIB is modelled with SIDES SEDs placed in per-frequency CCL tracer kernels. |
| [`LensingTomographyExample.ipynb`](LensingTomographyExample.ipynb) | Tomography | A validation case with a single "matter" component. Reconstructs $\sigma_8(z)$ from DESI BGS/LRG and Quaia galaxies cross-correlated with Planck PR4 CMB lensing, with and without DES Y3 cosmic shear (spin-2, using `maps_in_x` to keep only source bins behind each lens bin), and compares with the ΛCDM prediction. |

All notebooks add the Planck beams and HEALPix pixel windows to the map
tracers in the sacc file before running, since the input files do not
contain them.

---

## Citation

If you use this code, please cite:

* **MLBR and large-scale tomography**: A. La Posta, D. Alonso,
  C. García-García, S. Maleubre, *Joint tomographic measurement of thermal
  Sunyaev Zeldovich and the cosmic infrared background*,
  [arXiv:2603.04269](https://arxiv.org/abs/2603.04269).
* **Tomography methodology**: S. Maleubre, M. Zennaro, D. Alonso,
  I. McCarthy, M. Schaller, J. Schaye, *The impact of galaxy bias on
  cross-correlation tomography*,
  [arXiv:2508.05319](https://arxiv.org/abs/2508.05319).
* **Application of MLBR to cosmic shear**: A. Wayland, A. La Posta,
  D. Alonso, B. Jego, M. Béthermin, *Joint measurements of thermal
  Sunyaev-Zel'dovich and cosmic infrared background cross-correlations with
  cosmic shear*, [arXiv:2609.31774](https://arxiv.org/abs/2609.31774).

If you use the bundled data, please also credit
[`pyilc`](https://github.com/jcolinhill/pyilc) (Planck bandpasses) and the
[SIDES](https://gitlab.lam.fr/mbethermin/sides-public-release) public release
(CIB SED library; Béthermin et al. 2017).
