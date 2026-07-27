import functions
import datetime
import numpy as np
from astropy import constants as const

class DBMFit:
    def __init__(self,*,gamma, wind, tinit,rinit,vinit,residual,is_valid):

        self.gamma = gamma
        self.wind = wind
        self.tinit = tinit
        self.rinit = rinit
        self.vinit = vinit
        self.residual = residual
        self.is_valid = is_valid

    def get_parameters(self):
        return self.gamma, self.wind, self.tinit, self.rinit, self.vinit, self.residual, self.is_valid


class DBMResult:
    def __init__(self, fits_dict=None):
        self.fits = fits_dict if fits_dict else {}

    def add_fit(self, fit_id, gamma, wind, tinit, rinit, vinit, residual):
        self.fits[fit_id] = DBMFit(
            gamma=gamma,
            wind=wind,
            tinit=tinit,
            rinit=rinit,
            vinit=vinit,
            residual=residual,
            is_valid=True if wind != 0 else False
        )

    def compute_best_fit(self):
        best_fit = {}
        lowest_residual = float('inf')
        fit_id_best = None

        for fit_id, fit_data in self.fits.items():
            if fit_data.is_valid and fit_data.residual < lowest_residual:
                lowest_residual = fit_data.residual
                best_fit = fit_data
                fit_id_best = fit_id

        self.fits = {}
        if fit_id_best is not None:
            self.fits[fit_id_best] = best_fit

    
def run_dbm_fitting_wrapper(
    *,
    time,
    distance_au,
    startfit,
    endfit,
    prediction_path,
    det_plot,
    silent,
    setup_config=None
):
    """
    Unified DBM fitting interface for the pipeline.
    """

    use_updated = False
    allow_multiple_dbm_fits = False
    updated_dbm_vinit_computation = 'first'
    kwargs_dbm = {}

    if setup_config is not None:
        use_updated = setup_config.get("use_dbm_updated", False)
        allow_multiple_dbm_fits = setup_config.get("allow_multiple_dbm_fits", False)
        updated_dbm_vinit_computation = setup_config.get("updated_dbm_vinit_computation", 'first')
        kwargs_dbm = setup_config.get("kwargs_dbm", {})

    if use_updated:
        gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata, speed = functions.DBMfitting_updated(
            time=time,
            distance_au=distance_au,
            prediction_path=prediction_path,
            det_plot=det_plot,
            startfit=startfit,
            silent=silent,
            compute_vinit=updated_dbm_vinit_computation,
            **kwargs_dbm
        )

    else:
        gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata, speed = functions.DBMfitting(
            time=time,
            distance_au=distance_au,
            prediction_path=prediction_path,
            det_plot=det_plot,
            startfit=startfit,
            endfit=endfit,
            silent=silent,
        )

    dbm_result = DBMResult()


    for i in range(len(gamma_valid)):
        fit_id = i
        dbm_result.add_fit(
            fit_id=fit_id,
            gamma=gamma_valid[i],
            wind=winds_valid[i],
            tinit=tinit,
            rinit=rinit,
            vinit=vinit,
            residual=res_valid[i]
        )
    
    if not allow_multiple_dbm_fits:
        dbm_result.compute_best_fit()
    
    return dbm_result, speed

def run_cme_hit_function_wrapper(
    *,
    delta,
    halfwidth,
    setup_config=None):

    """
    Unified CME hit function interface for the pipeline.
    """

    use_updated = False

    if setup_config is not None:
        use_updated = setup_config.get("use_cme_hit_function_updated", False)

    if use_updated:
        return functions.does_cme_hit(
            delta=delta,
            halfwidth=halfwidth
        )

    else:
        return functions.does_cme_hit_elevohi(
            delta=delta,
            halfwidth=halfwidth
        )
    
def run_arrival_computation_wrapper(
    *,
    R,
    vdrag,
    time_array,
    L1_r,
    tnum,
    f,
    halfwidth,
    delta_target,
    L1_lon,
    setup_config=None
):

    """
    Unified arrival time computation interface for the pipeline.
    """

    AU_in_km = const.au.to_value('km')
    use_updated = False
    use_elevo_ensembles = False

    if setup_config is not None:
        use_updated = setup_config.get("use_arrival_computation_updated", False)
        use_elevo_ensembles = setup_config.get("use_elevo_ensembles", False)

    if use_updated:
        d_target,_,_,_ = functions.elevo_analytic_new(R, f, halfwidth, delta_target)

        if use_elevo_ensembles:
            speed_target = vdrag
        else:
            d_target_diff = np.diff(d_target*AU_in_km, axis=0)
            speed_target = d_target_diff / np.diff(tnum, axis=0)

        pred, _ = functions.compute_arrival_wrapper(d_target, speed_target, time_array, L1_r)

    else:
        pred, d_target = functions.elevo_new(R, time_array, tnum, f, halfwidth, 1, delta_target, L1_r, L1_lon)

    return pred, d_target

def run_dbm_kinematics_wrapper(
    *,
    gamma,
    winds,
    tinit,
    rinit,
    vinit,
    setup_config=None
):

    """
    Unified DBM kinematics interface for the pipeline.
    """

    use_elevo_ensembles = False
    use_arrival_computation_updated = False

    if setup_config is not None:
        use_elevo_ensembles = setup_config.get("use_elevo_ensembles", False)
        use_arrival_computation_updated = setup_config.get("use_arrival_computation_updated", False)

        if use_elevo_ensembles and not use_arrival_computation_updated:
            raise ValueError("Elevo ensembles require the updated arrival computation.")
        
    if use_elevo_ensembles:
        rng = np.random.default_rng(42)
        n_ensemble = 1000
        gamma_ens = np.abs(rng.normal(gamma*1e7,0.025,n_ensemble))
        wind_ens = rng.normal(winds,50,n_ensemble)
        speed_ens = rng.normal(vinit,50,n_ensemble)

        time_step = datetime.timedelta(minutes=10)
        timegrid = 1440
        time_array = [tinit + i * time_step for i in range(timegrid)]
        tnum = [(t - tinit).total_seconds() for t in time_array]

        distance0 = rinit
        cme_r_ensemble, cme_v_ensemble = functions.compute_cme_ensemble(gamma_ens, wind_ens, speed_ens, np.array(tnum), distance0)

        cme_r_ensemble = cme_r_ensemble / const.au.to_value('km')  # convert back to AU

        cme_r_mean = cme_r_ensemble.mean(1)
        cme_r_std = cme_r_ensemble.std(1)
        cme_v_mean = cme_v_ensemble.mean(1)
        cme_v_std = cme_v_ensemble.std(1)

        cme_r=np.zeros([len(cme_r_mean), 3])
        cme_v=np.zeros([len(cme_v_mean), 3])

        cme_r[:,0]= cme_r_mean # mean of cme distance ensemble
        cme_r[:,1]=(cme_r_mean - 2*cme_r_std)  # lower limit of cme distance ensemble
        cme_r[:,2]=(cme_r_mean + 2*cme_r_std) # upper limit of cme distance ensemble
        cme_v[:,0]= cme_v_mean
        cme_v[:,1]=(cme_v_mean - 2*cme_v_std)
        cme_v[:,2]=(cme_v_mean + 2*cme_v_std)

        R = cme_r
        vdrag = cme_v
        tnum = np.array(tnum)[:, np.newaxis]  # reshape for consistency
        
    else:
        R, vdrag, time_array, tnum = functions.compute_dbm_kinematics_single(tinit, rinit, vinit, gamma, winds)
    
    return R, vdrag, time_array, tnum

class BaselineImplementation:

    def cme_hit(self, *, delta, halfwidth):
        return run_cme_hit_function_wrapper(
            delta=delta,
            halfwidth=halfwidth,
            setup_config=None
        )

    def dbm_fit(self, *, time, distance_au, startcut, endcut, prediction_path):
        return run_dbm_fitting_wrapper(
            time=time,
            distance_au=distance_au,
            startfit=startcut,
            endfit=endcut,
            prediction_path=prediction_path,
            det_plot=False,
            silent=1,
            setup_config=None
        )

    def dbm_kinematics(self, *, gamma, winds, tinit, rinit, vinit):
        return run_dbm_kinematics_wrapper(
            gamma=gamma,
            winds=winds,
            tinit=tinit,
            rinit=rinit,
            vinit=vinit,
            setup_config=None
        )
    
    def arrival(self, *, R, vdrag, time_array, tnum, f, halfwidth, delta_target, L1_r, L1_lon):
        return run_arrival_computation_wrapper(
            R=R,
            vdrag=vdrag,
            time_array=time_array,
            L1_r=L1_r,
            tnum=tnum,
            f=f,
            halfwidth=halfwidth,
            delta_target=delta_target,
            L1_lon=L1_lon,
            setup_config=None
        )
    
class UpdatedImplementation:

    def __init__(self, setup_config):
        self.setup_config = setup_config

    def cme_hit(self, *, delta, halfwidth):
        return run_cme_hit_function_wrapper(
            delta=delta,
            halfwidth=halfwidth,
            setup_config=self.setup_config
        )

    def dbm_fit(self, *, time, distance_au, startcut, endcut, prediction_path):
        return run_dbm_fitting_wrapper(
            time=time,
            distance_au=distance_au,
            startfit=startcut,
            endfit=endcut,
            prediction_path=prediction_path,
            det_plot=False,
            silent=1,
            setup_config=self.setup_config
        )

    def dbm_kinematics(self, *, gamma, winds, tinit, rinit, vinit):
        return run_dbm_kinematics_wrapper(
            gamma=gamma,
            winds=winds,
            tinit=tinit,
            rinit=rinit,
            vinit=vinit,
            setup_config=self.setup_config
        )

    def arrival(self, *, R, vdrag, time_array, tnum, f, halfwidth, delta_target, L1_r, L1_lon):
        return run_arrival_computation_wrapper(
            R=R,
            vdrag=vdrag,
            time_array=time_array,
            L1_r=L1_r,
            tnum=tnum,
            f=f,
            halfwidth=halfwidth,
            delta_target=delta_target,
            L1_lon=L1_lon,
            setup_config=self.setup_config
        )

