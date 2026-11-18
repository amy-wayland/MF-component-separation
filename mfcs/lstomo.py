import numpy as np
import pyccl as ccl
import scipy.linalg as la
import scipy.stats as stats
from .utils import (_cltyp, _ones, _beam_interpolator,
                    _check_beam_correction_kind, _eval_beam)


class LargeScaleTomographer(object):
    """
    A class for performing large-scale tomography.

    Args:
        sacc_file: A SACC file containing the data and covariance.
        gal_names: List of galaxy tracer names in the SACC file.
        map_names: List of map names in the SACC file.
        comp_names: List of component names to be separated. The assumption
            is that all maps listed in `map_names` are linear combinations of
            all components listed in `comp_names`.
        tracer_classes: Dictionary mapping component names to
            their corresponding tracer classes. The structure of this
            dictionary can be either:
            1. A dictionary mapping component names to tracer classes,
               where each tracer class is a callable that takes a cosmology
               object as its only argument, and returns a CCL tracer object.
               In this case, the same tracer class is used for all maps for
               a given component. The corresponding map-component power
               spectrum contribution will be scaled by the SED of that
               map-component pair if present in `seds`.
            2. A nested dictionary mapping component names to dictionaries
               that map map names to tracer classes. In this case, different
               tracer classes can be used for different maps for a given
               component. If present, the SEDs for these map-component
               pairs will be used to scale the corresponding power spectrum
               contributions.
        maps_in_x: Either 'all' or a dictionary mapping galaxy tracer
            names to lists of map names for cross-correlations. If 'all',
            all maps in `map_names` are used for cross-correlations with
            each galaxy tracer.
        seds: Optional dictionary mapping component names to their SEDs
            for each map. For any component-map pair not provided, the
            SED is assumed to be 1.
        n_max: Maximum order of k-dependence to include in the model
            (default is 1). Terms of the form k^(2n)*P(k) are included for
            n=0 to n_max.
        kmax: Maximum wavenumber for the analysis in units of Mpc^-1
            (default is 0.3).
        lmax: Maximum ell value for the analysis. If `kmax` is not `None`, this
            parameter is ignored. If a single number, this scale cut is
            applied to all galaxy tracers. Otherwise, this should be a
            dictionary mapping galaxy tracer names to their respective scale
            cuts. If `None`, all scales present in the sacc file are used
            (default is `None`).
        lmin: Minimum ell value for the analysis (default is 0).
        map_spin: Spin of the maps used. This may be a single value or a
            dictionary mapping map names to spins. In all cases, allowed
            spin values are 0, 2, 'e', and 'b'. If 0, the map is assumed
            to be a scalar map. If 2, the map is assumed to be a spin-2 map,
            and only its E-mode component is used. If 'e' or 'b',
            cross-correlations with the corresponding spin component will
            be used (default is 0).
        cosmo: Cosmology object to use in the calculation. If `None`, a
            vanilla LCDM cosmology is used (default is `None`).
        include_stochasticity: Whether to include a stochastic term in the
            model. (default is True).
        beam_correction_kind: How to include the map beams in the model
            for the galaxy-map cross-correlations. If 'mid', the beam is
            evaluated at the effective ell of each bandpower. If 'bpw', the
            beam is averaged over the bandpower window functions stored in
            the SACC file. If 'none', no beam is applied, e.g. because the
            power spectra in the SACC file are already beam-deconvolved
            (default is 'mid').
    """
    def __init__(self, sacc_file, gal_names, map_names, comp_names,
                 tracer_classes, maps_in_x='all', seds=None,
                 n_max=1, kmax=0.3, lmax=None, lmin=0, map_spin=0,
                 cosmo=None, include_stochasticity=True,
                 beam_correction_kind='mid'):
        self.s = sacc_file
        self.gal_names = gal_names
        self.map_names = map_names
        self.comp_names = comp_names
        self.n_comps = len(comp_names)
        self.tracer_classes = tracer_classes
        self.maps_in_x = maps_in_x
        self.n_max = n_max
        self.terms = ['N'] if include_stochasticity else []
        self.terms = self.terms + [f'k{2*i}' for i in range(self.n_max + 1)]
        self.n_terms = len(self.terms)
        self.kmax = kmax
        self.lmax = lmax
        self.lmin = lmin
        self.map_spin = map_spin
        self.beam_correction_kind = _check_beam_correction_kind(
            beam_correction_kind)
        if cosmo is None:
            cosmo = ccl.CosmologyVanillaLCDM()

        self._check_sanity(seds)
        self._collect_tracer_data()
        self._get_data_vector()
        self._build_amp_indices()
        self.update_SEDs(seds)
        self.update_cosmo(cosmo)

    def update_cosmo(self, cosmo):
        """
        Update the cosmological model used for tomography. In
        addition to updating the cosmology, this method also recomputes
        the theoretical power spectra and template matrices, and solves
        for the amplitudes.

        Args:
            cosmo: A ccl.Cosmology object.
        """
        self.cosmo = cosmo
        # Redshift and k arrays for power spectrum templates
        a_arr = ccl.get_pk_spline_a()
        lk_arr = ccl.get_pk_spline_lk()
        k_arr = np.exp(lk_arr)
        self.k_pivot = 0.1

        # Power spectrum templates
        self.pks = {}
        pknl = self.cosmo.get_nonlin_power()
        self.pks['k0'] = pknl
        self.pks['N'] = ccl.Pk2D(
            a_arr=a_arr, lk_arr=lk_arr,
            pk_arr=np.zeros((len(a_arr), len(lk_arr))),
            extrap_order_lok=0, extrap_order_hik=0, is_logp=True)
        for i in range(1, self.n_max + 1):
            self.pks[f'k{2*i}'] = ccl.Pk2D(
                a_arr=a_arr, lk_arr=lk_arr,
                pk_arr=np.log(np.array([
                    (k_arr/self.k_pivot)**(2*i) * pknl(k_arr, a)
                    for a in a_arr])),
                extrap_order_lok=1, extrap_order_hik=1, is_logp=True)

        # Create galaxy tracers for each galaxy sample
        self.gal_tracers = {
            gal_name: ccl.NumberCountsTracer(
                self.cosmo, has_rsd=False,
                dndz=(self.tracer_data[gal_name]['z'],
                      self.tracer_data[gal_name]['nz']),
                bias=(self.tracer_data[gal_name]['z'],
                      np.ones_like(self.tracer_data[gal_name]['z'])))
            for gal_name in self.gal_names}

        # Create map tracers for each map and its components
        self.map_tracers = {}
        for comp_name, tc in self.tracer_classes.items():
            if isinstance(tc, dict):
                for map_name in self.map_names:
                    self.map_tracers[(comp_name,
                                      map_name)] = tc[map_name](cosmo)
            else:
                tobj = tc(cosmo)
                for map_name in self.map_names:
                    self.map_tracers[(comp_name,
                                      map_name)] = tobj

        # Compute the theoretical power spectra for each
        # galaxy-map pair and component
        self.th_cls_gg = {}
        self.th_cls_gm = {}
        for gal_name in self.gal_names:
            # ell values for all correlations involving this galaxy tracer
            ell = self.cl_info[(gal_name, gal_name)]['ell']
            # Galaxy auto-correlations
            for t in self.terms:  # Auto-correlations for all terms
                cl = ccl.angular_cl(
                    self.cosmo,
                    self.gal_tracers[gal_name],
                    self.gal_tracers[gal_name],
                    ell,
                    p_of_k_a=self.pks[t])
                self.th_cls_gg[(gal_name, t)] = cl
            for comp_name in self.comp_names:
                tc = self.tracer_classes[comp_name]
                if isinstance(tc, dict):
                    for map_name in self.maps_in_x[gal_name]:
                        t_map = self.map_tracers[(comp_name, map_name)]
                        sed = self.seds[(comp_name, map_name)]
                        beam = self.cl_info[(gal_name, map_name)]['beam']
                        for t in self.terms:
                            cl = ccl.angular_cl(
                                self.cosmo,
                                self.gal_tracers[gal_name],
                                t_map,
                                ell,
                                p_of_k_a=self.pks[t])
                            self.th_cls_gm[(gal_name, map_name,
                                            comp_name, t)] = cl*sed*beam
                else:
                    # We can avoid computing all cls, since
                    # the same tracer is used for all maps.
                    # We just scale the result by the SED for each map.
                    for t in self.terms:
                        cl = ccl.angular_cl(
                            self.cosmo,
                            self.gal_tracers[gal_name],
                            self.map_tracers[(comp_name,
                                              self.maps_in_x[gal_name][0])],
                            ell,
                            p_of_k_a=self.pks[t])
                        for map_name in self.maps_in_x[gal_name]:
                            sed = self.seds[(comp_name, map_name)]
                            beam = self.cl_info[(gal_name, map_name)]['beam']
                            self.th_cls_gm[(gal_name, map_name,
                                            comp_name, t)] = cl*sed*beam

        # Build template matrices
        blocks = []
        for gal_name in self.gal_names:
            ell = self.cl_info[(gal_name, gal_name)]['ell']
            n_ell = len(ell)
            n_maps = len(self.maps_in_x[gal_name])
            # Galaxy auto-correlation templates
            templates_gg = np.array([
                self.th_cls_gg[(gal_name, t)] for t in self.terms])
            # Galaxy-map cross-correlation templates
            templates_gm = np.array([[[
                self.th_cls_gm[(gal_name, map_name, comp_name, t)]
                for map_name in self.maps_in_x[gal_name]]
                for comp_name in self.comp_names]
                for t in self.terms]).reshape([self.n_terms*self.n_comps,
                                               n_maps*n_ell])
            blocks.append(templates_gg.T)
            blocks.append(templates_gm.T)
        self.Tmat = la.block_diag(*blocks)

        self._solve_amplitudes()

    def _build_amp_indices(self):
        """
        Build a dictionary mapping amplitude names to their indices.
        """
        self.amp_indices = {}
        i = 0
        for gal_name in self.gal_names:
            for t in self.terms:
                self.amp_indices[f'{gal_name}_{gal_name}_{t}'] = i
                i += 1
            for t in self.terms:
                for comp_name in self.comp_names:
                    self.amp_indices[f'{gal_name}_{comp_name}_{t}'] = i
                    i += 1

    def _solve_amplitudes(self, regularisation_factor=None):
        """
        Solve for the amplitudes using the generalized least squares solution.
        This method computes the amplitudes that best fit the data given the
        theoretical templates and the covariance matrix.

        Args:
            regularisation_factor: Optional regularization factor to
                stabilize the solution. If provided, a diagonal
                regularization matrix is added to the Fisher information
                matrix before solving for the amplitudes.
        """
        # Solve for the amplitudes using the generalized least
        # squares solution
        T_iC = (self.inv_cov @ self.Tmat).T
        T_iC_T = T_iC @ self.Tmat
        T_iC_d = T_iC @ self.data

        # Solve for amplitudes
        if regularisation_factor is not None:
            preamp = np.linalg.solve(T_iC_T, T_iC_d)
            reg_matrix = np.diag(1.0/(regularisation_factor*preamp)**2)
            T_iC_T += reg_matrix
        self.amps = np.linalg.solve(T_iC_T, T_iC_d)
        self.icov_amps = T_iC_T

    @property
    def cov_amps(self):
        """
        Return the covariance matrix of the amplitudes.
        This is the inverse of the Fisher information matrix for
        the amplitudes.
        """
        return np.linalg.inv(self.icov_amps)

    def get_chi2(self):
        """
        Compute the chi-squared value for the fit.
        Returns:
            chi2: The chi-squared value.
            ndof: The number of degrees of freedom.
            pte: The probability to exceed (p-value) for the chi-squared value.
        """
        # Compute the chi-squared value for the fit
        model = self.Tmat @ self.amps
        residual = self.data - model
        chi2 = residual @ self.inv_cov @ residual
        ndof = len(self.data) - len(self.amps)
        pte = 1 - stats.chi2.cdf(chi2, ndof)
        return chi2, ndof, pte

    def update_SEDs(self, seds):
        """
        Update the SEDs for the components. All `None` SEDs are set to 1.
        """
        self.seds = {}
        if seds is None:
            for comp_name in self.comp_names:
                for map_name in self.map_names:
                    self.seds[(comp_name, map_name)] = 1
        else:
            for comp_name in self.comp_names:
                if comp_name not in seds:
                    raise ValueError(
                        f"SED for component {comp_name} not provided.")
                sed = seds[comp_name]
                if sed is not None:
                    for map_name in self.map_names:
                        if map_name not in sed:
                            raise ValueError(
                                f"SED for component {comp_name} "
                                f"in map {map_name} not provided.")
                        self.seds[(comp_name, map_name)] = sed[map_name]
                else:
                    for map_name in self.map_names:
                        self.seds[(comp_name, map_name)] = 1

    def _check_sanity(self, seds_input):
        """ Check that the provided inputs are consistent and valid."""
        # Check all galaxy tracers and maps are present in the SACC file
        for gal_name in self.gal_names:
            if gal_name not in self.s.tracers:
                raise ValueError(
                    f"Galaxy tracer {gal_name} not found in SACC file.")

        for map_name in self.map_names:
            if map_name not in self.s.tracers:
                raise ValueError(
                    f"Map {map_name} not found in SACC file.")

        # If maps_in_x is not 'all', check that the
        # specified cross-maps are valid
        if self.maps_in_x != 'all':
            for gal_name in self.gal_names:
                if gal_name not in self.maps_in_x:
                    raise ValueError(
                        f"Galaxy tracer {gal_name} not found in maps_in_x.")
                for map_name in self.maps_in_x[gal_name]:
                    if map_name not in self.map_names:
                        raise ValueError(
                            f"Map {map_name} in map list for "
                            f"galaxy tracer {gal_name} not found.")
        else:
            # If maps_in_x is 'all', create a dictionary
            # with all maps for each galaxy tracer
            self.maps_in_x = {gal_name: self.map_names
                              for gal_name in self.gal_names}

        # Check all tracer classes are provided for each map and component
        for comp_name in self.comp_names:
            if comp_name not in self.tracer_classes:
                raise ValueError(
                    f"Tracer class for component {comp_name} not provided.")
            tc = self.tracer_classes[comp_name]
            if isinstance(tc, dict):
                for map_name in self.map_names:
                    if map_name not in tc:
                        raise ValueError(
                            f"Tracer class for component {comp_name} "
                            f"in map {map_name} not provided.")
            else:
                # If not defined per map, then this
                # component must have an SED
                if seds_input is None or comp_name not in seds_input:
                    raise ValueError(
                        f"SED for component {comp_name} not provided.")

        # Determine the appropriate cl_type for each map based on its spin
        if np.isscalar(self.map_spin):
            self.clxtype = {m: _cltyp(self.map_spin)
                            for m in self.map_names}
        elif isinstance(self.map_spin, dict):
            self.clxtype = {m: _cltyp(self.map_spin[m])
                            for m in self.map_names}
        else:
            raise ValueError(
                "map_spin must be either a scalar or a "
                "dictionary mapping map names to spins.")

    def _iter_cls(self):
        for gal_name in self.gal_names:
            yield gal_name, gal_name, 'cl_00'
            for map_name in self.maps_in_x[gal_name]:
                yield gal_name, map_name, self.clxtype[map_name]

    def _collect_tracer_data(self):
        """ Collect redshift distributions and other relevant
        data for each tracer."""
        cosmo = ccl.CosmologyVanillaLCDM()
        self.tracer_data = {}
        for gal_name in self.gal_names:
            t = self.s.tracers[gal_name]
            z_mean = np.average(t.z, weights=t.nz)
            chi_mean = ccl.comoving_radial_distance(cosmo, 1/(1+z_mean))
            if self.kmax is not None:
                lmax = self.kmax * chi_mean
            elif np.isscalar(self.lmax):
                lmax = self.lmax
            elif isinstance(self.lmax, dict):
                lmax = self.lmax.get(gal_name, np.inf)
            else:
                lmax = np.inf
            self.tracer_data[gal_name] = {'z': t.z, 'nz': t.nz,
                                          'z_mean': z_mean,
                                          'chi_mean': chi_mean,
                                          'lmax': lmax}
        for map_name in self.map_names:
            t = self.s.tracers[map_name]
            if hasattr(t, 'beam'):
                beam = _beam_interpolator(t.ell, t.beam)
            else:
                beam = _ones
            self.tracer_data[map_name] = {'beam': beam}

    def _get_data_vector(self):
        """ Collect the data vector and covariance matrix from the SACC
        file for all relevant correlations."""
        self.cl_info = {}
        data_vector = []
        inds = []
        for t1, t2, cl_type in self._iter_cls():
            ell, cl, cov, ind = self.s.get_ell_cl(
                cl_type, t1, t2, return_cov=True, return_ind=True)
            if len(ell) == 0:
                ell, cl, cov, ind = self.s.get_ell_cl(
                    cl_type, t2, t1, return_cov=True, return_ind=True)
            if len(ell) == 0:
                raise ValueError(
                    f"No data found for {t1} x {t2} in SACC file.")
            lmax = self.tracer_data[t1]['lmax'] if t1 in self.gal_names else self.tracer_data[t2]['lmax']  # noqa: E501
            good = (ell >= self.lmin) & (ell <= lmax)
            ell = ell[good]
            cl = cl[good]
            cov = cov[good][:, good]
            ind = ind[good]
            self.cl_info[(t1, t2)] = {'n': len(ell), 'ell': ell,
                                      'cl': cl, 'cov': cov,
                                      'ind': ind}
            if t2 in self.map_names:
                # Beam for galaxy-map cross-correlations
                self.cl_info[(t1, t2)]['beam'] = _eval_beam(
                    self.s, self.tracer_data[t2]['beam'], ell, ind,
                    self.beam_correction_kind)
            data_vector.append(cl)
            inds.append(ind)
        self.data = np.concatenate(data_vector)
        inds = np.concatenate(inds)
        self.cov = self.s.covariance.dense[inds][:, inds]
        self.inv_cov = np.linalg.inv(self.cov)

        # Sanity check: check that the ells for all correlations
        # involving the same galaxy tracer are consistent
        for gal_name in self.gal_names:
            ell_ref = self.cl_info[(gal_name, gal_name)]['ell']
            for map_name in self.maps_in_x[gal_name]:
                ell_map = self.cl_info[(gal_name, map_name)]['ell']
                if not np.array_equal(ell_ref, ell_map):
                    raise ValueError(
                        f"Ell values for {gal_name} x {map_name} "
                        f"do not match those for {gal_name} x {gal_name}.")

    def get_linear_biases(self, nwalkers=None, nsteps=1000, burnin=200):
        """
        Compute the linear biases for each galaxy tracer and component, as well
        as their uncertainties. The linear bias for a galaxy tracer is defined
        as the square root of the linear amplitude of its auto-correlation,
        while the linear bias for a component is defined as the ratio of the
        linear amplitude of its cross-correlation with a galaxy tracer to the
        galaxy tracer's linear bias. The biases are computed by sampling the
        posterior distribution of the amplitudes using MCMC.

        Returns:
            biases: A dictionary mapping galaxy tracer names to their
                linear biases.
            comp_biases: A dictionary mapping component names to their
                linear biases.
            samples: The MCMC samples of the biases.
            param_names: A list of parameter names corresponding to
                the biases.
        """
        indices = []
        bf_params = []
        param_names = []
        for gal_name in self.gal_names:
            iAgg = self.amp_indices[f"{gal_name}_{gal_name}_k0"]
            indices.append(iAgg)
            # Avoid negative square roots due to numerical issues
            bg = np.sqrt(np.fabs(self.amps[iAgg]))
            bf_params.append(bg)
            param_names.append(f"b_{gal_name}")
            for comp_name in self.comp_names:
                iAgm = self.amp_indices[f"{gal_name}_{comp_name}_k0"]
                bgm = self.amps[iAgm]
                bf_params.append(bgm / bg)
                indices.append(iAgm)
                param_names.append(f"b_{gal_name}_{comp_name}")
        indices = np.array(indices)
        bf_params = np.array(bf_params)

        data = self.amps[indices]
        cov = self.cov_amps[indices][:, indices]
        icov = np.linalg.inv(cov)

        def logp(pars):
            th = []
            i = 0
            for gal_name in self.gal_names:
                bg = pars[i]
                th.append(bg**2)
                i += 1
                for comp_name in self.comp_names:
                    bgm = pars[i]
                    th.append(bgm * bg)
                    i += 1
            th = np.array(th)
            diff = data - th
            return -0.5 * np.dot(diff, np.dot(icov, diff))

        # Use emcee to sample the posterior distribution of the parameters
        import emcee

        ndim = len(bf_params)
        nwalkers = 2*ndim if nwalkers is None else nwalkers
        # Use the best-fit parameters as the starting point for the MCMC
        p0 = bf_params + 1e-4 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logp)
        sampler.run_mcmc(p0, nsteps)
        samples = sampler.get_chain(discard=burnin, flat=True)
        # Compute the mean and standard deviation of the
        # samples for each parameter
        means = np.mean(samples, axis=0)
        stds = np.std(samples, axis=0)
        biases = {}
        comp_biases = {}
        i = 0
        for gal_name in self.gal_names:
            biases[gal_name] = (means[i], stds[i], bf_params[i])
            i += 1
            for comp_name in self.comp_names:
                comp_biases[(gal_name,
                             comp_name)] = (means[i], stds[i],
                                            bf_params[i])
                i += 1
        return biases, comp_biases, samples, param_names
