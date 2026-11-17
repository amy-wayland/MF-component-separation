import numpy as np
import scipy.linalg as la
import sacc
from utils import _cltyp, _ones
from scipy.interpolate import interp1d


class MLBR(object):
    def __init__(self, sacc_file, lss_names, map_names, comp_names,
                 seds, maps_in_x, lmax, lmin, lss_spin,
                 beam_correction_kind='mid'):
        self.s = sacc_file
        self.lss_names = lss_names
        self.map_names = map_names
        self.comp_names = comp_names
        self.n_comps = len(comp_names)
        self.maps_in_x = maps_in_x
        self.lss_spin = lss_spin
        self.lmin = lmin
        self.lmax = lmax
        self.beam_correction_kind = beam_correction_kind
        self._check_sanity()
        self._collect_tracer_data()
        self._get_data_vector()
        self._build_comp_indices()
        self.update_seds(seds)

    def get_component_indices(self, lss_name, comp_name):
        if (lss_name, comp_name) not in self.comp_inds:
            raise ValueError(f"Component indices for LSS tracer {lss_name} "
                             f"and component {comp_name} not found.")
        return self.comp_inds[(lss_name, comp_name)]

    def _build_comp_indices(self):
        self.comp_inds = {}
        i_yet = 0
        for lss_name in self.lss_names:
            ells_common = self.common_ells[lss_name]
            n_ells = len(ells_common)
            for comp_name in self.comp_names:
                inds = i_yet + np.arange(n_ells)
                self.comp_inds[(lss_name, comp_name)] = inds
                i_yet += n_ells

    def update_seds(self, seds):
        seds_full = {}
        # Check all components have SEDs
        for comp_name in self.comp_names:
            seds_full[comp_name] = {}
            if comp_name not in seds:
                raise ValueError(f"SED for component {comp_name} not found in seds.")
            sed = seds[comp_name].copy()
            # Check if SEDs are defined per lss tracer or globally
            lss_in_each = [lss_name in sed for lss_name in self.lss_names]
            lss_in_any = np.any(lss_in_each)
            lss_in_all = np.all(lss_in_each)
            if lss_in_any and not lss_in_all:
                raise ValueError(f"SED for component {comp_name} is "
                                 f"defined for some but not all LSS tracers.")
            if not lss_in_all:
                # Check that the SED is defined for all maps
                for map_name in self.map_names:
                    if map_name not in sed:
                        raise ValueError(f"SED for component {comp_name} "
                                         f"does not have an entry for map {map_name}.")
                # Assign the global SED to each lss tracer
                for lss_name in self.lss_names:
                    seds_full[comp_name][lss_name] = sed
            else:
                seds_full[comp_name] = sed
                # Check that the SED is defined for all maps in each lss tracer
                for lss_name in self.lss_names:
                    for map_name in self.maps_in_x[lss_name]:
                        if map_name not in seds_full[comp_name][lss_name]:
                            raise ValueError(f"SED for component {comp_name} does not have "
                                             f"an entry for map {map_name} in LSS "
                                             f"tracer {lss_name}.")
        self.seds = seds_full
        self._build_s_matrix()

    def _check_sanity(self):
        # Check that all lss tracers exist in the sacc file
        for lss_name in self.lss_names:
            if lss_name not in self.s.tracers:
                raise ValueError(f"LSS tracer {lss_name} not found in sacc file.")
        # Check that all map tracers exist in the sacc file
        for map_name in self.map_names:
            if map_name not in self.s.tracers:
                raise ValueError(f"Map tracer {map_name} not found in sacc file.")
        # If maps_in_x is not 'all', check that the specified cross-maps are valid
        if self.maps_in_x != 'all':
            for lss_name in self.lss_names:
                if lss_name not in self.maps_in_x:
                    raise ValueError(f"LSS tracer {lss_name} not found in maps_in_x.")
                for map_name in self.maps_in_x[lss_name]:
                    if map_name not in self.map_names:
                        raise ValueError(f"Map tracer {map_name} not found in maps_in_x for LSS tracer {lss_name}.")
        else:
            # If maps_in_x is 'all', create a dictionary with all maps for each lss tracer
            self.maps_in_x = {lss_name: self.map_names for lss_name in self.lss_names}

        # Determine the appropriate cl_type for each map based on its spin
        if np.isscalar(self.lss_spin):
            self.clxtype = {n: _cltyp(self.lss_spin) for n in self.lss_names}
        elif isinstance(self.lss_spin, dict):
            self.clxtype = {n: _cltyp(self.map_spin[n]) for n in self.lss_names}
        else:
            raise ValueError("lss_spin must be either a scalar or a dictionary mapping lss tracer names to spins.")

        # lmax should be a scalar (applied to all cross-correlations) of a dictionary
        # with a different lmax for each map.
        if np.isscalar(self.lmax):
            self.lmax = {n: self.lmax for n in self.map_names}
        for m in self.map_names:
            if m not in self.lmax:
                raise ValueError(f"Map tracer {m} not found in lmax.")

        # Check beam correction type
        if self.beam_correction_kind in ['None', None]:
            self.beam_correction_kind = 'none'
        if self.beam_correction_kind not in ['mid', 'bpw', 'none']:
            raise ValueError(f"Unkown beam correction kind {self.beam_correction_kind}. "
                             f"Must be one of 'mid', 'bpw', or 'none'.")

    def _collect_tracer_data(self):
        self.tracer_data = {}
        for map_name in self.map_names:
            t = self.s.tracers[map_name]
            if hasattr(t, 'beam'):
                # Extrapolate to 1 at low ell and 0 at high ell
                beam = interp1d(t.ell, t.beam,
                                bounds_error=False,
                                fill_value=(1.0, 0.0))
            else:
                beam = _ones
            self.tracer_data[map_name] = {'beam': beam}

    def _eval_beam(self, beamf, ells, inds):
        if self.beam_correction_kind == 'mid':
            return beamf(ells)
        elif self.beam_correction_kind == 'bpw':
            bpw = self.s.get_bandpower_windows(inds)
            bl = beamf(bpw.values)
            return np.einsum('ij,i->j', bpw.weight, bl)
        elif self.beam_correction_kind == 'none':
            return np.ones(len(ells))

    def _get_data_vector(self):
        self.cl_info = {}
        self.common_ells = {}
        data_vector = []
        inds = []
        for lss_name in self.lss_names:
            ells_this, cls_this, cov_this, inds_this = [], [], [], []
            for map_name in self.maps_in_x[lss_name]:
                cl_type = self.clxtype[lss_name]
                ell, cl, cov, ind = self.s.get_ell_cl(cl_type, lss_name, map_name, return_cov=True, return_ind=True)
                if len(ell) == 0:
                    ell, cl, cov, ind = self.s.get_ell_cl(cl_type, map_name, lss_name, return_cov=True, return_ind=True)
                good = (ell >= self.lmin) & (ell <= self.lmax[map_name])
                ell = ell[good]
                cl = cl[good]
                cov = cov[good][:, good]
                ind = ind[good]
                ells_this.append(ell)
                cls_this.append(cl)
                cov_this.append(cov)
                inds_this.append(ind)
                data_vector.append(cl)
                inds.append(ind)
                beam = self._eval_beam(self.tracer_data[map_name]['beam'],
                                       ell, ind)
                self.cl_info[(lss_name, map_name)] = {'ell': ell, 'cl': cl, 'cov': cov,
                                                      'ind': ind, 'n_ells': len(ell),
                                                      'beam': beam}
            ells_common = np.unique(np.concatenate(ells_this))
            ells_common.sort()
            self.common_ells[lss_name] = ells_common
            if len(ells_common) > np.amax([len(ell) for ell in ells_this]):
                raise ValueError(f"Ells for LSS tracer {lss_name} do not match across maps.")
            for i, map_name in enumerate(self.maps_in_x[lss_name]):
                ell = ells_this[i]
                # Find indices of this ell array in the common ell array
                inds_in_common = np.searchsorted(ells_common, ell)
                self.cl_info[(lss_name, map_name)]['inds_in_common'] = inds_in_common
        self.data_vector = np.concatenate(data_vector)
        inds = np.concatenate(inds)
        self.cov = self.s.covariance.dense[inds][:, inds]
        self.inv_cov = np.linalg.inv(self.cov)

    def _build_s_matrix(self):
        blocks = []
        for lss_name in self.lss_names:
            ells_common = self.common_ells[lss_name]
            n_ells = len(ells_common)
            block_g = []
            for map_name in self.maps_in_x[lss_name]:
                nls = self.cl_info[(lss_name, map_name)]['n_ells']
                inds = self.cl_info[(lss_name, map_name)]['inds_in_common']
                beam = self.cl_info[(lss_name, map_name)]['beam']
                seds = np.array([self.seds[comp_name][lss_name][map_name]
                                 for comp_name in self.comp_names])
                # Create a block matrix where each component's SED is
                # multiplied by the beam and placed in the correct columns
                # The result is an array of shape (nls, n_comps * n_ells)
                on = np.ones(n_ells)
                block_gm = np.concatenate([np.diag(seds[icomp] * on)
                                           for icomp in range(self.n_comps)],
                                          axis=-1)
                block_gm = block_gm[inds] * beam[:, None]
                block_g.append(block_gm)
            block_g = np.concatenate(block_g, axis=0)
            blocks.append(block_g)
        self.Smat = la.block_diag(*blocks)

    def get_component_cls(self, get_cov=False):
        # Compute the component C_ell's using the formula:
        # C_comp = (S^T C^{-1} S)^{-1} S^T C^{-1} d
        St_Cinv = (self.inv_cov @ self.Smat).T
        St_Cinv_S = St_Cinv @ self.Smat
        St_Cinv_d = St_Cinv @ self.data_vector

        if get_cov:
            comp_cov = np.linalg.inv(St_Cinv_S)
            comp_cls = np.dot(comp_cov, St_Cinv_d)
            return comp_cls, comp_cov
        else:
            comp_cls = np.linalg.solve(St_Cinv_S, St_Cinv_d)
            return comp_cls
