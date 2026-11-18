import numpy as np
from scipy.interpolate import interp1d


def _ones(ell):
    return 1 if np.isscalar(ell) else np.ones_like(ell)


def _cltyp(map_spin):
    if map_spin == 0:
        return 'cl_00'
    elif map_spin == 2:
        return 'cl_0e'
    elif map_spin in ['e', 'b']:
        return f'cl_0{map_spin}'
    else:
        raise ValueError(f"Invalid map_spin value: {map_spin}. "
                         "Must be 0, 2, 'e', or 'b'.")


def _beam_interpolator(ell, beam):
    """ Interpolate a tabulated beam. Outside the tabulated range,
    the beam is taken to be 1 at low ell and 0 at high ell."""
    return interp1d(ell, beam, bounds_error=False, fill_value=(1.0, 0.0))


def _check_beam_correction_kind(kind):
    """ Validate and normalise the beam correction kind."""
    if kind in ['None', None]:
        kind = 'none'
    if kind not in ['mid', 'bpw', 'none']:
        raise ValueError(
            f"Unknown beam correction kind {kind}. "
            f"Must be one of 'mid', 'bpw', or 'none'.")
    return kind


def _eval_beam(s, beamf, ells, inds, kind):
    """ Evaluate the beam for a set of bandpowers.

    Args:
        s: SACC file containing the bandpowers.
        beamf: Callable returning the beam as a function of ell.
        ells: Effective ells of the bandpowers.
        inds: Indices of the bandpowers in the SACC file.
        kind: 'mid' (beam evaluated at the effective ells), 'bpw'
            (beam averaged with the bandpower window functions),
            or 'none' (no beam).
    """
    if kind == 'mid':
        return beamf(ells)
    elif kind == 'bpw':
        bpw = s.get_bandpower_windows(inds)
        if bpw is None:
            raise ValueError(
                "Beam correction kind 'bpw' requires bandpower "
                "windows, but none were found in the SACC file.")
        bl = beamf(bpw.values)
        return np.einsum('ij,i->j', bpw.weight, bl)
    elif kind == 'none':
        return np.ones(len(ells))
    raise ValueError(f"Unknown beam correction kind {kind}.")
