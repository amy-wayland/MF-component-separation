import numpy as np


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
