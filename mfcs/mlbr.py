import numpy as np
import scipy.linalg as la
from .utils import (_cltyp, _ones, _beam_interpolator,
                    _check_beam_correction_kind, _eval_beam)


class MaxLikeBandpowerReconstructor(object):
    """
    A class for performing maximum-likelihood bandpower reconstruction
    (MLBR). Given the cross-power spectra between a set of LSS tracers
    and a set of (e.g. multi-frequency) maps, it reconstructs the
    cross-power spectra between each LSS tracer and each of the sky
    components contributing to the maps.

    For each LSS tracer g, the cross-power spectrum with map m is modelled
    as a linear combination of the component cross-spectra:

        C_ell^{gm} = B^m_ell * sum_c S^g_c(m) * C_ell^{gc},

    where S^g_c(m) is the SED of component c in map m (which may depend on
    the LSS tracer), and B^m_ell is the beam of map m. Writing this as
    d = S c + n, the maximum-likelihood solution for the component
    bandpowers c, and its covariance, are

        c = (S^T C^{-1} S)^{-1} S^T C^{-1} d,
        Cov(c) = (S^T C^{-1} S)^{-1},

    where C is the covariance of the data vector d.

    Args:
        sacc_file: A SACC file containing the data and covariance.
        lss_names: List of LSS tracer names in the SACC file.
        map_names: List of map names in the SACC file.
        comp_names: List of component names to be separated. The assumption
            is that all maps listed in `map_names` are linear combinations of
            all components listed in `comp_names`.
        seds: Dictionary of SEDs, already integrated over the bandpass of
            each map and in the units of the map. For each component in
            `comp_names`, the corresponding entry can be either:
            1. A dictionary mapping map names to SED values. In this case,
               the same SED is used for all LSS tracers (e.g. tSZ).
            2. A nested dictionary mapping LSS tracer names to dictionaries
               that map map names to SED values. In this case, a different
               SED can be used for each LSS tracer (e.g. CIB, whose
               effective SED depends on the redshift kernel of the LSS
               tracer). In this case, an SED must be provided for all LSS
               tracers.
            SEDs must be provided for all maps cross-correlated with each
            LSS tracer.
        maps_in_x: Either 'all' or a dictionary mapping LSS tracer names
            to lists of map names for cross-correlations. If 'all', all
            maps in `map_names` are used for cross-correlations with each
            LSS tracer.
        lmax: Maximum ell value for the analysis. If a single number, this
            scale cut is applied to all maps. Otherwise, this should be a
            dictionary mapping map names to their respective scale cuts,
            which allows e.g. noisy or contaminated maps to be cut at
            lower ell.
        lmin: Minimum ell value for the analysis.
        lss_spin: Spin of the LSS tracers. This may be a single value or a
            dictionary mapping LSS tracer names to spins. In all cases,
            allowed spin values are 0, 2, 'e', and 'b'. If 0, the tracer
            is assumed to be a scalar field. If 2, the tracer is assumed to
            be a spin-2 field, and only its E-mode component is used. If
            'e' or 'b', cross-correlations with the corresponding spin
            component will be used.
        beam_correction_kind: How to include the map beams in the model.
            Beams are read from the `beam` attribute of each map tracer
            (assumed to be 1 for tracers without one). If 'mid', the beam
            is evaluated at the effective ell of each bandpower. If 'bpw',
            the beam is averaged over the bandpower window functions stored
            in the SACC file. If 'none', no beam is applied, e.g. because
            the power spectra in the SACC file are already beam-deconvolved
            (default is 'mid').

    The reconstructed component bandpowers for each LSS tracer are
    defined on `common_ells[lss_name]`: the union of the ell values of
    all its cross-correlations surviving the scale cuts. These must be
    the same bandpowers across maps, except for different scale cuts.
    The output vector of component bandpowers is ordered by LSS tracer,
    then by component, then by ell. Use `get_component_indices` to
    extract the bandpowers for a given LSS tracer and component.
    """
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
        """
        Return the indices of the reconstructed bandpowers for a given
        LSS tracer and component in the vector of component bandpowers
        (and its covariance) returned by `get_component_cls`. The
        corresponding ell values are `common_ells[lss_name]`.

        Args:
            lss_name: Name of the LSS tracer.
            comp_name: Name of the component.

        Returns:
            inds: Array of indices.
        """
        if (lss_name, comp_name) not in self.comp_inds:
            raise ValueError(f"Component indices for LSS tracer {lss_name} "
                             f"and component {comp_name} not found.")
        return self.comp_inds[(lss_name, comp_name)]

    def _build_comp_indices(self):
        """
        Build a dictionary mapping (LSS tracer, component) pairs to the
        indices of their bandpowers in the component bandpower vector.
        """
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
        """
        Update the SEDs for the components and rebuild the mixing matrix.
        This does not require re-reading the data, and can therefore be
        used e.g. to marginalise over SED parameters.

        Args:
            seds: Dictionary of SEDs, in the same format as the `seds`
                argument of the class constructor.
        """
        seds_full = {}
        # Check all components have SEDs
        for comp_name in self.comp_names:
            seds_full[comp_name] = {}
            if comp_name not in seds:
                raise ValueError(f"SED for component {comp_name} "
                                 f"not found in seds.")
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
                                         f"does not have an entry for "
                                         f"map {map_name}.")
                # Assign the global SED to each lss tracer
                for lss_name in self.lss_names:
                    seds_full[comp_name][lss_name] = sed
            else:
                seds_full[comp_name] = sed
                # Check that the SED is defined for all maps in each lss tracer
                for lss_name in self.lss_names:
                    for map_name in self.maps_in_x[lss_name]:
                        if map_name not in seds_full[comp_name][lss_name]:
                            raise ValueError(
                                f"SED for component {comp_name} does not have "
                                f"an entry for map {map_name} in LSS "
                                f"tracer {lss_name}.")
        self.seds = seds_full
        self._build_s_matrix()

    def _check_sanity(self):
        """ Check that the provided inputs are consistent and valid."""
        # Check that all lss tracers exist in the sacc file
        for lss_name in self.lss_names:
            if lss_name not in self.s.tracers:
                raise ValueError(
                    f"LSS tracer {lss_name} not found in sacc file.")
        # Check that all map tracers exist in the sacc file
        for map_name in self.map_names:
            if map_name not in self.s.tracers:
                raise ValueError(
                    f"Map tracer {map_name} not found in sacc file.")
        # If maps_in_x is not 'all', check that the
        # specified cross-maps are valid
        if self.maps_in_x != 'all':
            for lss_name in self.lss_names:
                if lss_name not in self.maps_in_x:
                    raise ValueError(
                        f"LSS tracer {lss_name} not found in maps_in_x.")
                for map_name in self.maps_in_x[lss_name]:
                    if map_name not in self.map_names:
                        raise ValueError(
                            f"Map tracer {map_name} not found in "
                            f"maps_in_x for LSS tracer {lss_name}.")
        else:
            # If maps_in_x is 'all', create a dictionary
            # with all maps for each lss tracer
            self.maps_in_x = {lss_name: self.map_names
                              for lss_name in self.lss_names}

        # Determine the appropriate cl_type for each map based on its spin
        if np.isscalar(self.lss_spin):
            self.clxtype = {n: _cltyp(self.lss_spin)
                            for n in self.lss_names}
        elif isinstance(self.lss_spin, dict):
            self.clxtype = {n: _cltyp(self.lss_spin[n])
                            for n in self.lss_names}
        else:
            raise ValueError("lss_spin must be either a scalar or a "
                             "dictionary mapping lss tracer names to spins.")

        # lmax should be a scalar (applied to all cross-correlations)
        # of a dictionary with a different lmax for each map.
        if np.isscalar(self.lmax):
            self.lmax = {n: self.lmax for n in self.map_names}
        for m in self.map_names:
            if m not in self.lmax:
                raise ValueError(f"Map tracer {m} not found in lmax.")

        # Check beam correction type
        self.beam_correction_kind = _check_beam_correction_kind(
            self.beam_correction_kind)

    def _collect_tracer_data(self):
        """ Collect the beam of each map."""
        self.tracer_data = {}
        for map_name in self.map_names:
            t = self.s.tracers[map_name]
            if hasattr(t, 'beam'):
                beam = _beam_interpolator(t.ell, t.beam)
            else:
                beam = _ones
            self.tracer_data[map_name] = {'beam': beam}

    def _get_data_vector(self):
        """ Collect the data vector and covariance matrix from the SACC
        file for all relevant cross-correlations, apply scale cuts, and
        evaluate the beam for each of them."""
        self.cl_info = {}
        self.common_ells = {}
        data_vector = []
        inds = []
        for lss_name in self.lss_names:
            ells_this, cls_this, cov_this, inds_this = [], [], [], []
            for map_name in self.maps_in_x[lss_name]:
                cl_type = self.clxtype[lss_name]
                ell, cl, cov, ind = self.s.get_ell_cl(
                    cl_type, lss_name, map_name,
                    return_cov=True, return_ind=True)
                if len(ell) == 0:
                    ell, cl, cov, ind = self.s.get_ell_cl(
                        cl_type, map_name, lss_name,
                        return_cov=True, return_ind=True)
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
                beam = _eval_beam(self.s, self.tracer_data[map_name]['beam'],
                                  ell, ind, self.beam_correction_kind)
                self.cl_info[(lss_name, map_name)] = {
                    'ell': ell, 'cl': cl, 'cov': cov, 'ind': ind,
                    'n_ells': len(ell), 'beam': beam}
            ells_common = np.unique(np.concatenate(ells_this))
            ells_common.sort()
            self.common_ells[lss_name] = ells_common
            if len(ells_common) > np.amax([len(ell) for ell in ells_this]):
                raise ValueError(
                    f"Ells for LSS tracer {lss_name} do not "
                    f"match across maps.")
            for i, map_name in enumerate(self.maps_in_x[lss_name]):
                ell = ells_this[i]
                # Find indices of this ell array in the common ell array
                inds_in_common = np.searchsorted(ells_common, ell)
                self.cl_info[(lss_name,
                              map_name)]['inds_in_common'] = inds_in_common
        self.data_vector = np.concatenate(data_vector)
        inds = np.concatenate(inds)
        self.cov = self.s.covariance.dense[inds][:, inds]
        self.inv_cov = np.linalg.inv(self.cov)

    def _build_s_matrix(self):
        """ Build the mixing matrix S, which maps the component bandpowers
        into the data vector. S is block-diagonal, with one block per LSS
        tracer containing the beam-weighted SED of each component in each
        map."""
        blocks = []
        for lss_name in self.lss_names:
            ells_common = self.common_ells[lss_name]
            n_ells = len(ells_common)
            block_g = []
            for map_name in self.maps_in_x[lss_name]:
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
        """
        Compute the maximum-likelihood component bandpowers:

            c = (S^T C^{-1} S)^{-1} S^T C^{-1} d.

        Args:
            get_cov: If True, also return the covariance matrix of the
                component bandpowers, (S^T C^{-1} S)^{-1}
                (default is False).

        Returns:
            comp_cls: Vector of component bandpowers. Use
                `get_component_indices` to extract the bandpowers for a
                given LSS tracer and component.
            comp_cov: Covariance matrix of `comp_cls` (only returned if
                `get_cov` is True).
        """
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
