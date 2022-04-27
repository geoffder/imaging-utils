import numpy as np
import symfit as sf


class BiexpFitter:
    def __init__(self, est_tau1, est_tau2, amp=1.0, norm_amp=False):
        self.a0 = amp
        self.b0 = amp
        self.norm_amp = norm_amp
        a, b, g, t = sf.variables("a, b, g, t")
        tau1 = sf.Parameter("tau1", est_tau1)
        tau2 = sf.Parameter("tau2", est_tau2)
        y0 = sf.Parameter("y0", est_tau2)
        y0.value = 1.0
        y0.min = 0.5
        y0.max = 2.0
        self.ode_model = sf.ODEModel(
            {
                # sf.D(a, t): -a / tau1,
                # sf.D(b, t): -b / tau2,
                # HACK: trick model into fitting an initial value (always 1)
                # https://stackoverflow.com/questions/49149241/ode-fitting-with-symfit-for-python-how-to-get-estimations-for-intial-values
                sf.D(a, t): -a / tau1 * (sf.cos(y0) ** 2 + sf.sin(y0) ** 2),
                sf.D(b, t): -b / tau2 * (sf.cos(y0) ** 2 + sf.sin(y0) ** 2),
            },
            initial={
                t: 0,
                a: y0.value,
                b: y0.value,
                # a: amp,
                # b: amp
            },
        )
        self.model = sf.CallableNumericalModel(
            {g: self.g_func}, connectivity_mapping={g: {t, tau1, tau2, y0}}
        )
        self.constraints = [
            sf.GreaterThan(tau1, 0.0001),
            sf.GreaterThan(tau2, tau1),
        ]

    def g_func(self, t, tau1, tau2, y0):
        res = self.ode_model(t=t, tau1=tau1, tau2=tau2, y0=y0)
        g = res.b - res.a
        gmax = np.max(g)
        if self.norm_amp and not np.isclose(gmax, 0.0):
            return g * 1 / gmax
        else:
            # tp = (tau1 * tau2) / (tau2 - tau1) * np.log(tau2 / tau1)
            # factor = 1 / (-np.exp(-tp / tau1) + np.exp(-tp / tau2))
            # return g + factor
            return g

    def fit(self, x, y):
        self.results = sf.Fit(
            self.model, t=x, g=y, constraints=self.constraints
        ).execute()
        return self.results

    def calc_g(self, x):
        return self.model(t=x, **self.results.params)[0]
