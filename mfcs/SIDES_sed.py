import numpy as np
from scipy.io import readsav
from scipy.interpolate import RegularGridInterpolator


class SIDESBethermin(object):
    """
    Class to build CIB SED from Bethermin 2017 model.
    It reads the IDL saveset file, build interpolators
    and is callable to compute the observed flux density
    at a given redshift and frequency.
    """
    c = 2.99792458e8  # m/s
    w_MJy = 1e20  # Watt to MJy
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
        self.nuLnu_ms /= np.trapezoid(
            self.nuLnu_ms[:, idx],
            np.log(self.rf_lambda[idx]),
            axis=-1
        )
        self.nuLnu_sb /= np.trapezoid(
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
                 nu_ghz,
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
        nu_ghz : float or array-like
            Observed frequency(ies) in GHz at which to compute the
            flux density.
        sb_frac_at_z0 : float, optional
            Fraction of starburst galaxies at z=0. Default is 0.015.
        sb_frac_at_zsat : float, optional
            Fraction of starburst galaxies at saturation redshift.
            Default is 0.030.
        sb_zsat : float, optional
            Saturation redshift for starburst fraction. Default is 1.0.
        logU_ms_int : float, optional
            Logarithm of the mean radiation field intensity for main sequence
            galaxies at z=0. Default is log10(5.0).
        logU_ms_slope : float, optional
            Slope of the redshift evolution of the mean radiation field
            intensity for main sequence galaxies. Default is 0.25.
        Ums_zsat : float, optional
            Saturation redshift for the mean radiation field intensity of
            main sequence galaxies. Default is 4.0.
        logU_sb_int : float, optional
            Logarithm of the mean radiation field intensity for starburst
            galaxies at z=0. Default is log10(31.0).
        logU_sb_slope : float, optional
            Slope of the redshift evolution of the mean radiation field
            intensity for starburst galaxies. Default is 0.0.

        Returns
        -------
        nuLnu : float or array-like
            The mixed luminosity at the given redshift(s) and
            frequency(ies).
        """
        nu_rf = nu_ghz * 1E9 * (1 + z)
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
        U_ms = np.where(z >= Ums_zsat,
                        10**(logU_ms_int+logU_ms_slope*Ums_zsat), U_ms)

        # For SB galaxies, there is no redshift evolution in Bethermin17
        log10U_sb = logU_sb_int + logU_sb_slope * z
        U_sb = 10 ** log10U_sb
        U_sb = np.where(U_sb < U_ms, U_ms, U_sb)

        nuLnu_ms = 10**self.nuLnu_ms((np.log10(U_ms), lambda_rf))
        nuLnu_sb = 10**self.nuLnu_sb((np.log10(U_sb), lambda_rf))

        return ms_fraction * nuLnu_ms + sb_fraction * nuLnu_sb

    def sed_avgz(self, nu_ghz, z, wz, cosmo, **kwargs):
        """
        Compute the average SED over a redshift distribution.

        Parameters
        ----------
        nu_ghz : float or array-like
            Observed frequency(ies) in GHz at which to compute
            the flux density.
        z : array-like
            Redshift grid.
        wz : array-like
            Weights corresponding to the redshift grid.
        cosmo : astropy.cosmology instance
            Cosmology to use for distance calculations.
        **kwargs : dict
            Additional keyword arguments to pass to `sed_grid`.
        """
        # Compute the average SED over the redshift distribution
        # using the provided weights
        norm = 1 / np.trapezoid(wz, z)
        sed = self.sed_grid(z, nu_ghz, cosmo, **kwargs)
        return norm * np.trapezoid(wz * sed, z, axis=-1)

    def sed_grid(self,
                 z,
                 nu_ghz,
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
        nu_ghz = np.atleast_1d(nu_ghz)
        Z, NU = np.meshgrid(z, nu_ghz)

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
        dist = cosmo.luminosity_distance(1/(1+z)) * self.Mpc_to_m
        nuSnu = nuLnu / (4 * np.pi * dist[None, :]**2)
        Snu = nuSnu / (1E9*nu_ghz[:, None])
        Snu *= self.w_MJy * self.L_sun

        return Snu
