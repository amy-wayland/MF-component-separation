import numpy as np
import pyccl as ccl
import healpy as hp
import matplotlib.pyplot as plt
import sacc
import os
from pyilc import fg
from scipy.io import readsav
from scipy.interpolate import interp1d, RegularGridInterpolator

# ----- Physical SED Model -----

class BetherminSED:
    """
    Class to build CIB SED from Bethermin 2017 model.
    It reads the IDL saveset file, build interpolators
    and is callable to compute the observed flux density
    at a given redshift and frequency.
    """
    c = 2.99792458e8  # m/s
    w_jy = 1e26  # Watt to Jy
    L_sun = 3.828e26
    Mpc_to_m = 3.086e22

    def __init__(self, SED_U_file):
        self._set_attributes_from_sed_file(SED_U_file)

    def _set_attributes_from_sed_file(self, SED_U_file):
        SED_u = readsav(SED_U_file)
        self.rf_lambda = SED_u["lambda_grids"] * 1e-6
        self.rf_nu = self.c / self.rf_lambda
        self.u = SED_u["ugrid"]

        # 2D array of nuLnu for a grid of <U> and wavelengths
        self.nuLnu_ms = SED_u["sed_u_ms"].copy()
        self.nuLnu_sb = SED_u["sed_u_sb"].copy()

        idx = np.where(
            (self.rf_lambda >= 8e-6) & (self.rf_lambda <= 1000e-6)
        )
        self.nuLnu_ms /= np.trapz(
            self.nuLnu_ms[:, idx],
            np.log(self.rf_lambda[idx]),
            axis=-1
        )
        self.nuLnu_sb /= np.trapz(
            self.nuLnu_sb[:, idx],
            np.log(self.rf_lambda[idx]),
            axis=-1
        )

        # Build interpolators
        self.nuLnu_ms = RegularGridInterpolator(
            (np.log10(self.u), self.rf_lambda),
            np.log10(self.nuLnu_ms),
            bounds_error=False,
            fill_value=None,
        )
        self.nuLnu_sb = RegularGridInterpolator(
            (np.log10(self.u), self.rf_lambda),
            np.log10(self.nuLnu_sb),
            bounds_error=False,
            fill_value=None,
        )

    def mix_seds(self,
                 z,
                 nu,
                 sb_frac_at_z0=0.015,
                 sb_frac_at_zsat=0.030,
                 sb_zsat=1.0,
                 logU_ms_int=np.log10(5.0),
                 logU_ms_slope=0.25,
                 Ums_zsat=4.0,
                 logU_sb_int=np.log10(31.0),
                 logU_sb_slope=0.0):
        """
        Mix the main sequence and starburst SEDs based on the given parameters.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to compute the flux density.
        nu : float or array-like
            Observed frequency(ies) in Hz at which to compute the flux density.
        sb_frac_at_z0 : float, optional
            Fraction of starburst galaxies at z=0. Default is 0.015.
        sb_frac_at_zsat : float, optional
            Fraction of starburst galaxies at saturation redshift. Default is 0.030.
        sb_zsat : float, optional
            Saturation redshift for starburst fraction. Default is 1.0.
        logU_ms_int : float, optional
            Logarithm of the mean radiation field intensity for main sequence galaxies at z=0. Default is log10(5.0).
        logU_ms_slope : float, optional
            Slope of the redshift evolution of the mean radiation field intensity for main sequence galaxies. Default is 0.25.
        Ums_zsat : float, optional
            Saturation redshift for the mean radiation field intensity of main sequence galaxies. Default is 4.0.
        logU_sb_int : float, optional
            Logarithm of the mean radiation field intensity for starburst galaxies at z=0. Default is log10(31.0).
        logU_sb_slope : float, optional
            Slope of the redshift evolution of the mean radiation field intensity for starburst galaxies. Default is 0.0.

        Returns
        -------
        nuLnu : float or array-like
            The mixed luminosity at the given redshift(s) and frequency(ies).
        """
        nu_rf = nu * (1 + z)
        lambda_rf = self.c / nu_rf

        # Fraction of SB is 1.5% at z=0 and 3.0% at z=1 in Bethemin15/17
        # Here we use a linear relation with a saturation redshift
        slope = (sb_frac_at_zsat - sb_frac_at_z0) / sb_zsat
        sb_fraction = sb_frac_at_z0 + slope * np.minimum(z, sb_zsat)

        # SB and MS fractions
        sb_fraction *= 5.3/0.87
        ms_fraction = 1.0 - sb_fraction

        # log10(<U>)(z) = log10(<U>)(z=0) + alpha * z
        # where alpha is 0.25 and intercept is log10(5) from Bethermin17
        log10U_ms = logU_ms_int + logU_ms_slope * z
        U_ms = 10 ** log10U_ms
        U_ms = np.where(z >= Ums_zsat, 10 ** (logU_ms_int + logU_ms_slope * Ums_zsat), U_ms)

        # For SB galaxies, there is no redshift evolution in Bethermin17
        log10U_sb = logU_sb_int + logU_sb_slope * z
        U_sb = 10 ** log10U_sb
        U_sb = np.where(U_sb < U_ms, U_ms, U_sb)

        nuLnu_ms = 10**self.nuLnu_ms((np.log10(U_ms), lambda_rf))
        nuLnu_sb = 10**self.nuLnu_sb((np.log10(U_sb), lambda_rf))

        return ms_fraction * nuLnu_ms + sb_fraction * nuLnu_sb

    def __call__(self, 
                 z,
                 nu,
                 cosmo,
                 sb_frac_at_z0=0.015,
                 sb_frac_at_zsat=0.030,
                 sb_zsat=1.0,
                 logU_ms_int=np.log10(5.0),
                 logU_ms_slope=0.25,
                 Ums_zsat=4.0,
                 logU_sb_int=np.log10(31.0),
                 logU_sb_slope=0.0):

        z = np.atleast_1d(z)
        nu = np.atleast_1d(nu)
        Z, NU = np.meshgrid(z, nu)

        nuLnu = self.mix_seds(
            Z,
            NU,
            logU_ms_int=logU_ms_int,
            logU_ms_slope=logU_ms_slope,
            Ums_zsat=Ums_zsat,
            logU_sb_int=logU_sb_int,
            logU_sb_slope=logU_sb_slope,
            sb_frac_at_z0=sb_frac_at_z0,
            sb_frac_at_zsat=sb_frac_at_zsat,
            sb_zsat=sb_zsat,
        )
        dist = cosmo.luminosity_distance(1/(1+Z.flatten())) * self.Mpc_to_m
        dist = dist.reshape(Z.shape)
        nuSnu = nuLnu / (4 * np.pi * dist**2)
        Snu = nuSnu / NU
        Snu *= self.w_jy * self.L_sun

        return Snu

def bandpass_sed(nu, sed, passband):
    # Interpolate passband
    nu_bp, bp = passband

    nu_bp_ghz = nu_bp / 1e9
    nu_ghz = nu / 1e9
    bp_int = interp1d(
        nu_bp_ghz,
        bp,
        bounds_error=False,
        fill_value="extrapolate"
    )
    bp_int = bp_int(nu_ghz)
    
    Snu = np.trapz(sed * bp_int[:, None], nu_ghz, axis=0) / np.trapz(bp_int, nu_ghz)
    return Snu

# ----- Configuration -----

cosmo = ccl.Cosmology(
    Omega_c=0.2610,
    Omega_b=0.04920,
    h=0.6762,
    sigma8=0.8149,
    n_s=0.9709,
)

sacc_file = "/mnt/extraspace/waylanda/outputs_DESY3_NPIPEnu/cls_cov.fits"
sed_u_file = "SED_U_finegrids.save"
sed_path = "/mnt/users/waylanda/tSZxWL_project/component_separation/pyilc/data"

shears = [
    "DESY3wl__0",
    "DESY3wl__1",
    "DESY3wl__2",
    "DESY3wl__3",
]

CMB = [
    "Planck__NPIPEnu100",
    "Planck__NPIPEnu143",
    "Planck__NPIPEnu217",
    "Planck__NPIPEnu353",
    "Planck__NPIPEnu545",
    "Planck__NPIPEnu857",
]

freqs = [int(tr.split("nu")[-1]) for tr in CMB]

unit_K_to_MJy_per_sr = {
    100: 244.1,
    143: 371.74,
    217: 483.69,
    353: 287.45,
    545: 58.04,
    857: 2.27,
}

color_corr = {
    100: 1.076,
    143: 1.017,
    217: 1.119,
    353: 1.097,
    545: 1.068,
    857: 0.995,
}

# ----- Load spectra -----

s = sacc.Sacc.load_fits(sacc_file)
s.keep_selection(ell__gt=30, ell__lt=2*2048)

to_keep = []
for i, shear1 in enumerate(shears):
    for shear2 in shears[i:]:
        to_keep.append(s.indices("cl_ee", tracers=(shear1, shear2)))
    for cmb in CMB:
        ell__lt = 10000 if not "857" in cmb else 300
        idx = s.indices("cl_0e", tracers=(shear1, cmb), ell__lt=ell__lt)
        if len(idx) == 0:
            print("Index was empty, flipping order")
            idx = s.indices("cl_0e", tracers=(cmb, shear1), ell__lt=ell__lt)
        to_keep.append(idx)
to_keep = np.concatenate(to_keep)
s.keep_indices(to_keep)

lb = s.get_ell_cl("cl_ee", shears[0], shears[0])[0]
n_bins = len(lb)
n_wlbins = len(shears)
n_freqs = len(CMB)

# ----- Beam correction -----

fwhm = {
    "Planck__NPIPEnu030": 32.29,
    "Planck__NPIPEnu044": 27.94,
    "Planck__NPIPEnu070": 13.08,
    "Planck__NPIPEnu100": 9.66,
    "Planck__NPIPEnu143": 7.22,
    "Planck__NPIPEnu217": 4.90,
    "Planck__NPIPEnu353": 4.92,
    "Planck__NPIPEnu545": 4.67,
    "Planck__NPIPEnu857": 4.22,
}

nside = 2048
lmax = 3*nside-1
pixwin = hp.pixwin(nside, lmax=lmax)

bls = {
    band: hp.gauss_beam(fwhm=np.deg2rad(fwhm / 60.0), lmax=lmax)
    for band, fwhm in fwhm.items()
}

blvec = []
for tr1, tr2 in s.get_tracer_combinations():
    if tr1.startswith("DESY3wl") and tr2.startswith("DESY3wl"):
        cltype = "cl_ee"
    else:
        cltype = "cl_0e"

    l, _ = s.get_ell_cl(cltype, tr1, tr2)
    bl = np.ones(len(l))
    #print(len(l))
    if "Planck" in tr1:
        idx = s.indices("cl_0e", tracers=(tr1, tr2))
        bpw = s.get_bandpower_windows(idx).weight
        bl *= np.einsum("ij,i->j", bpw, bls[tr1] * pixwin)
    if "Planck" in tr2:
        idx = s.indices("cl_0e", tracers=(tr1, tr2))
        bpw = s.get_bandpower_windows(idx).weight
        bl *= np.einsum("ij,i->j", bpw, bls[tr2] * pixwin)
    blvec.append(bl)

bbs = {
    (tr1, tr2): blvec[i]
    for i, (tr1, tr2) in enumerate(s.get_tracer_combinations())
}

blvec = np.concatenate(blvec)

# ----- Load bandpasses and tSZ SED -----

bps = {}

for freq in freqs:
    fname = (
        f"{sed_path}/HFI_BANDPASS_F{freq:03d}_reformat.txt"
        if freq >= 100
        else f"{sed_path}/LFI_BANDPASS_F{freq:03d}_reformat.txt"
    )

    nu_bp, bp_data = np.loadtxt(fname, unpack=True)
    bps[freq] = [nu_bp * 1e9, bp_data]

tSZ = fg.get_mix_bandpassed(
    [
        (
            f"{sed_path}/HFI_BANDPASS_F{freq:03d}_reformat.txt"
            if freq >= 100
            else f"{sed_path}/LFI_BANDPASS_F{freq:03d}_reformat.txt"
        )
        for freq in freqs
    ],
    comp="tSZ",
)

tSZs = {}

for wltracer in shears:
    for i, freq in enumerate(freqs):
        cmb = f"Planck__NPIPEnu{freq}"
        tSZs[(wltracer, cmb)] = tSZ[i]

# ----- WL kernels ------

wl_tracers_ccl = {
    wltracer: ccl.WeakLensingTracer(cosmo, dndz=(s.tracers[wltracer].z, s.tracers[wltracer].nz))
    for wltracer in shears
}

# ----- Build data vector -----

idx_cross = []
idx_ss = []
ordering = []
wl_pairs = []

for i, shear1 in enumerate(shears):
    for shear2 in shears[i:]:
        idx = s.indices("cl_ee", tracers=(shear1, shear2))

        if len(idx) == 0:
            idx = s.indices("cl_ee", tracers=(shear2, shear1))

        idx_ss.append(idx)
        wl_pairs.append((shear1, shear2))

for shear in shears:
    for cmb in CMB:
        idx = s.indices("cl_0e", tracers=(shear, cmb))

        if len(idx) == 0:
            idx = s.indices("cl_0e", tracers=(cmb, shear))
            ordering.append((cmb, shear))
        else:
            ordering.append((shear, cmb))

        idx_cross.append(idx)

idx_ss = np.concatenate(idx_ss)
idx_cross = np.concatenate(idx_cross)

covmat = s.covariance.covmat.copy()
covmat /= blvec[:, None] * blvec[None, :]

ps_vec = s.mean / blvec

covmat_cross = covmat[np.ix_(idx_cross, idx_cross)] * 1e12
ps_vec_cross = ps_vec[idx_cross] * 1e6

Cinv = np.linalg.inv(covmat_cross)

s_cross = s.copy()
s_cross.keep_indices(idx_cross)

# ----- Build mixing matrix -----

def build_S(CIB_dict):
    """
    Build frequency mixing matrix.

    Columns are:
        [tSZ bandpowers for each WL bin,
         CIB bandpowers for each WL bin]
    """

    S = np.zeros((s_cross.mean.shape[0], 2 * n_bins * n_wlbins))

    start_idx = 0

    for tr1, tr2 in ordering:
        if tr1 == tr2:
            continue

        if "Planck" in tr1:
            wltracer, cmb_tracer = tr2, tr1
        else:
            wltracer, cmb_tracer = tr1, tr2

        ell, _ = s_cross.get_ell_cl("cl_0e", tr1, tr2)

        if len(ell) == 0:
            ell, _ = s_cross.get_ell_cl("cl_0e", tr2, tr1)

        local_nbins = len(ell)
        idx_row = np.arange(start_idx, start_idx + local_nbins)
        start_idx += local_nbins

        subS = np.zeros((local_nbins, 2 * n_bins))

        tSZ_sed = tSZs[(wltracer, cmb_tracer)]
        cib_sed = CIB_dict[(wltracer, cmb_tracer)]

        subS[:, :local_nbins] = np.diag(np.ones(local_nbins) * tSZ_sed)
        subS[:, n_bins:n_bins + local_nbins] = np.diag(
            np.ones(local_nbins) * cib_sed
        )

        col_start = shears.index(wltracer) * (2 * n_bins)
        col_end = (shears.index(wltracer) + 1) * (2 * n_bins)

        S[idx_row, col_start:col_end] = subS

    return S

# ----- Physical CIB SEDs -----

sed_class = BetherminSED(sed_u_file)

nu_ghz_grid = np.geomspace(2.0, 7000.0, 500)
nu_grid = nu_ghz_grid * 1e9

def theta_to_sed_params(theta):
    """
    Convert the four sampled U(z) parameters into the arguments
    expected by BetherminSED.
    """
    slope_ms, int_ms, slope_sb, int_sb = theta

    return {
        "logU_ms_int": int_ms,
        "logU_ms_slope": slope_ms,
        "logU_sb_int": int_sb,
        "logU_sb_slope": slope_sb,
        "sb_frac_at_z0": 0.015,
        "sb_frac_at_zsat": 0.030,
        "sb_zsat": 1.0,
        "Ums_zsat": 4.0,
    }

U_main_sequence = np.array([
    [0.375,  5.50,  1.50,   3.10],
    [0.625,  7.23,  2.47,   3.82],
    [0.875,  7.80,  2.69,   5.44],
    [1.125, 15.05,  6.68,   5.74],
    [1.375, 16.52,  6.47,   5.45],
    [1.625, 16.96,  6.15,  10.90],
    [1.875, 16.96,  6.15,  15.24],
    [2.250, 22.58,  6.27,  14.42],
    [2.750, 33.75, 14.29,  12.85],
    [3.250, 48.99, 11.32,  23.99],
    [3.750, 72.98, 36.98, 167.95],
])

U_starburst = np.array([
    [0.750, 29.80, 11.77,  9.60],
    [1.250, 26.92,  6.92,  2.88],
    [1.750, 37.68, 28.40, 11.32],
    [2.500, 22.22, 12.94, 50.77],
])

def prepare_logU_data(data):
    z = data[:, 0]
    U = data[:, 1]
    err_low = data[:, 2]
    err_high = data[:, 3]

    logU = np.log10(U)
    logU_low = np.log10(U - err_low)
    logU_high = np.log10(U + err_high)

    sigma_low = logU - logU_low
    sigma_high = logU_high - logU

    return z, logU, sigma_low, sigma_high


z_ms, logU_ms_data, sigma_ms_low, sigma_ms_high = (
    prepare_logU_data(U_main_sequence)
)

z_sb, logU_sb_data, sigma_sb_low, sigma_sb_high = (
    prepare_logU_data(U_starburst)
)

# ----- External likelihood -----

def logU_linear(z, slope, intercept):
    return intercept + slope * z

def predict_logU(theta):
    slope_ms, int_ms, slope_sb, int_sb = theta

    pred_ms = logU_linear(z_ms, slope_ms, int_ms)
    pred_sb_unconstrained = logU_linear(z_sb, slope_sb, int_sb)
    pred_ms_at_sb_z = logU_linear(z_sb, slope_ms, int_ms)
    pred_sb = np.maximum(pred_sb_unconstrained, pred_ms_at_sb_z)

    return pred_ms, pred_sb

def split_normal_loglike(
    observed,
    predicted,
    sigma_low,
    sigma_high,
):
    """
    Piecewise-Gaussian likelihood for asymmetric error bars.
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    residual = observed - predicted

    # If prediction > observation, then the residual is less than zero
    # and we use the upper error
    sigma = np.where(
        residual < 0.0,
        sigma_high,
        sigma_low,
    )

    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        return -np.inf

    return -0.5 * np.sum((residual / sigma) ** 2)


def loglike_external_U(theta):
    pred_ms, pred_sb = predict_logU(theta)

    lnlike_ms = split_normal_loglike(
        logU_ms_data,
        pred_ms,
        sigma_ms_low,
        sigma_ms_high,
    )

    lnlike_sb = split_normal_loglike(
        logU_sb_data,
        pred_sb,
        sigma_sb_low,
        sigma_sb_high,
    )

    return lnlike_ms + lnlike_sb

def compute_physical_CIB_seds_bin(theta, target_bin):
    sed_params = theta_to_sed_params(theta)

    wltracer = shears[target_bin]

    z = s.tracers[wltracer].z
    chis = ccl.comoving_radial_distance(
        cosmo,
        1.0 / (1.0 + z),
    )
    W = wl_tracers_ccl[wltracer].get_kernel(chis)

    snu_grid = sed_class(
        z,
        nu_grid,
        cosmo,
        **sed_params,
    )

    CIBs_bin = {}

    denominator = np.trapz(W * chis**2, z)

    if denominator <= 0.0:
        raise ValueError(
            f"Invalid WL-kernel normalisation for {wltracer}"
        )

    for freq in freqs:
        cmb_tracer = f"Planck__NPIPEnu{freq}"

        snu_band = bandpass_sed(
            nu_grid,
            snu_grid,
            bps[freq],
        )

        effective_snu = np.trapz(
            snu_band * W * chis**2,
            z,
        ) / denominator

        CIBs_bin[(wltracer, cmb_tracer)] = (
            effective_snu
            / unit_K_to_MJy_per_sr[freq]
            * color_corr[freq]
        )

    return CIBs_bin

def compute_physical_CIB_seds_all_bins(theta):
    CIBs_all = {}

    for bin_index in range(n_wlbins):
        CIBs_all.update(
            compute_physical_CIB_seds_bin(
                theta,
                bin_index,
            )
        )

    return CIBs_all

# ----- Component separation likelihood

def component_separation_solution(theta):
    """
    Conditional component-separated bandpowers for fixed SED parameters.

    Returns
    -------
    mean_omega : ndarray
        Conditional component-bandpower mean.
    cov_omega : ndarray
        Conditional covariance.
    T : ndarray
        Frequency mixing matrix.
    """
    try:
        CIB_dict = compute_physical_CIB_seds_all_bins(theta)
        T = build_S(CIB_dict)
    except (ValueError, FloatingPointError):
        return None, None, None

    fisher = T.T @ Cinv @ T
    rhs = T.T @ Cinv @ ps_vec_cross

    if (
        not np.all(np.isfinite(fisher))
        or not np.all(np.isfinite(rhs))
    ):
        return None, None, None

    sign, _ = np.linalg.slogdet(fisher)

    if sign <= 0:
        return None, None, None

    try:
        mean_omega = np.linalg.solve(fisher, rhs)
        cov_omega = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        return None, None, None

    return mean_omega, cov_omega, T

def loglike_component_separation(theta):
    mean_omega, cov_omega, T = component_separation_solution(theta)

    if mean_omega is None:
        return -np.inf

    residual = ps_vec_cross - T @ mean_omega

    chi2_min = residual @ Cinv @ residual

    if not np.isfinite(chi2_min):
        return -np.inf

    return -0.5 * chi2_min

# ----- MCMC configuration -----

def external_U_cobaya(
    slope_ms,
    int_ms,
    slope_sb,
    int_sb,
):
    theta = np.array([
        slope_ms,
        int_ms,
        slope_sb,
        int_sb,
    ])

    return loglike_external_U(theta)

def component_separation_cobaya(
    slope_ms,
    int_ms,
    slope_sb,
    int_sb,
):
    theta = np.array([
        slope_ms,
        int_ms,
        slope_sb,
        int_sb,
    ])

    return loglike_component_separation(theta)

sed_params_config = {
    "slope_ms": {
        "prior": {"dist": "norm", "loc": 0.31, "scale": 0.33},
        "ref": 0.25,
        "proposal": 0.03,
        "latex": r"\alpha_{\rm MS}",
    },
    "int_ms": {
        "prior": {"dist": "norm", "loc": 0.69, "scale": 0.75},
        "ref": np.log10(5.0),
        "proposal": 0.05,
        "latex": r"\log_{10}U_{{\rm MS},0}",
    },
    "slope_sb": {
        "prior": {"dist": "norm", "loc": -0.25, "scale": 1.8},
        "ref": 0.0,
        "proposal": 0.15,
        "latex": r"\alpha_{\rm SB}",
    },
    "int_sb": {
        "prior": {"dist": "norm", "loc": 1.63, "scale": 2.1},
        "ref": np.log10(31.0),
        "proposal": 0.15,
        "latex": r"\log_{10}U_{{\rm SB},0}",
    },
}

sed_params_config_cl_only = {
    "slope_ms": {
        "prior": {"dist": "norm", "loc": 0.31, "scale": 3 * 0.33},
        "ref": 0.25,
        "proposal": 0.05,
        "latex": r"\alpha_{\rm MS}",
    },

    "int_ms": {
        "prior": {"dist": "norm", "loc": 0.69, "scale": 3 * 0.75},
        "ref": np.log10(5.0),
        "proposal": 0.05,
        "latex": r"\log_{10}U_{{\rm MS},0}",
    },

    "slope_sb": {
        "prior": {"dist": "norm", "loc": -0.25, "scale": 3 * 1.8},
        "ref": 0.0,
        "proposal": 0.15,
        "latex": r"\alpha_{\rm SB}",
    },

    "int_sb": {
        "prior": {"dist": "norm", "loc": 1.63, "scale": 3 * 2.1},
        "ref": np.log10(31.0),
        "proposal": 0.15,
        "latex": r"\log_{10}U_{{\rm SB},0}",
    },
}

sampler_config = {
    "mcmc": {
        "Rminus1_stop": 0.01,
        "Rminus1_cl_stop": 0.05,
        "max_tries": 10000,
    }
}

info_external = {
    "likelihood": {
        "external_U": external_U_cobaya,
    },
    "params": sed_params_config,
    "sampler": sampler_config,
    "output": "chains/physical_sed_external",
    "force": True,
}

info_joint = {
    "likelihood": {
        "external_U": external_U_cobaya,
        "component_separation": component_separation_cobaya,
    },
    "params": sed_params_config,
    "sampler": sampler_config,
    "output": "chains/physical_sed_joint",
    "force": True,
}

info_cl_only = {
    "likelihood": {
        "component_separation": component_separation_cobaya,
    },
    "params": sed_params_config_cl_only,
    "sampler": sampler_config,
    "output": "chains/physical_sed_cl_only",
    "force": True,
}

from cobaya import run

updated_external, sampler_external = run(info_external)
updated_joint, sampler_joint = run(info_joint)
updated_cl_only, sampler_cl_only = run(info_cl_only)
