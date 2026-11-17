import numpy as np


# Note: the following units are assumed throughout
#  - Temperatures are in Kelvin
#  - Frequencies are in GHz
#  - Intensities are in MJy/sr


J_m2_to_MJy = 1E20  # Conversion from J/m^2 to MJy
# T_CMB = 2.7255
# T_CMB_uK = 2.7255E6
# hPlanck = 6.62607015E-25  # Units of J / GHz
# kBoltz = 1.380649E-23  # Units of J / K
clight = 0.299792458  # Units of m * GHz

# These are the constants used in pyilc. Using the above values,
# results (e.g. for the convolved tSZ SED) differ by up to 1%.
T_CMB = 2.726
T_CMB_uK = 2.726E6
hPlanck=6.626068e-25
kBoltz=1.3806503e-23
#clight=299792458.0


def BnuBB(nu, T=T_CMB, get_x=False):
    x = hPlanck * nu / (kBoltz * T)  # Unitless
    prefac_MJy = J_m2_to_MJy * 2 * hPlanck * nu**3 / clight**2
    ex = np.exp(x)
    if get_x:
        return prefac_MJy/(ex-1), x, ex
    return prefac_MJy/(ex-1)


def dBnudlogT(nu, T=T_CMB, get_x=False):
    BB, x, ex = BnuBB(nu, T=T, get_x=True)
    if get_x:
        return BB*x*ex/(ex-1), x, ex
    return BB*x*ex/(ex-1)


def tSZ_sed(nu, T=T_CMB):
    dBBdlT, x, ex = dBnudlogT(nu, T=T, get_x=True)

    return dBBdlT*(x*(ex+1)/(ex-1)-4)


def CMB_sed(nu, T=T_CMB):
    return dBnudlogT(nu, T=T)


class BPass(object):
    """
    A class to represent a bandpass.
    """
    def __init__(self, *, nu=None, trans=None, fname=None,
                 nsamples_fast=100):
        """
        Initialize the bandpass.

        Parameters
        ----------
        nu : array-like, optional
            The frequency values (in GHz).
        trans : array-like, optional
            The transmission values.
        fname : str, optional
            The name of the file containing the frequency
            and transmission data.
        nsamples_fast : int, optional
            The number of samples within the core of the bandpass
            to use for fast calculations.
        """
        if (nu is None or trans is None) and fname is None:
            raise ValueError("Must provide either (nu, trans) or fname.")
        if fname is not None:
            nu, trans = np.loadtxt(fname, unpack=True)
        self.nu = nu
        self.tau = trans

        # Select core of bandpass for rough calculations
        taumax = np.amax(self.tau)
        core = self.tau > 1E-3*taumax
        # Select nsamples_fast samples equi-spaced within the core
        self.nu_fast = np.linspace(np.amin(self.nu[core]),
                                   np.amax(self.nu[core]),
                                   nsamples_fast)
        self.tau_fast = np.interp(self.nu_fast, self.nu, self.tau)
        self.norm = 1/np.trapezoid(self.tau, self.nu)
        self.norm_fast = 1/np.trapezoid(self.tau_fast, self.nu_fast)
        self.nu_mean = self.integrate_over_tau(lambda nu: nu)
        self.unit_MJy_sr_to_K_CMB = self.integrate_over_tau(dBnudlogT)/T_CMB
        self.unit_MJy_sr_to_K_CMB_fast = self.integrate_over_tau(dBnudlogT, fast=True)/T_CMB

    def integrate_over_tau(self, func, fast=False, axis=-1):
        """
        Integrate the given function over the bandpass.

        Parameters
        ----------
        func : callable
            A function that takes frequency (in GHz) as input
            and returns a value (in any units).
        fast : bool, optional
            If True, use a faster but less accurate integration method.
        axis : int, optional
            The axis along which to integrate. Default is -1 (last axis).

        Returns
        -------
        float
            The bandpass-integrated value in the same units as the input function.
        """
        if fast:
            return np.trapezoid(func(self.nu_fast)*self.tau_fast,
                                self.nu_fast, axis=axis)*self.norm_fast
        return np.trapezoid(func(self.nu)*self.tau, self.nu, axis=axis)*self.norm

    def convolve(self, sed_func, fast=False, axis=-1):
        """
        Convolve the given SED with the bandpass.

        Parameters
        ----------
        sed_func : callable
            A function that takes frequency (in GHz) as input
            and returns the SED (in MJy/sr).
        fast : bool, optional
            If True, use a faster but less accurate convolution method.
        axis : int, optional
            The axis along which to convolve. Default is -1 (last axis).

        Returns
        -------
        float
            The bandpass-convolved SED value in units of K_CMB.
        """
        if fast:
            return self.integrate_over_tau(sed_func, fast=True, axis=axis)/self.unit_MJy_sr_to_K_CMB_fast
        return self.integrate_over_tau(sed_func, fast=fast, axis=axis)/self.unit_MJy_sr_to_K_CMB
