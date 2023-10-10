import numpy as np
import ruptures as rpt
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch


class Segmentation(BaseEstimator):
    """
    Segmentation of our multivariate signals, uniform or adaptive (using `ruptures`).

    Inputs a list of multivariate signals.
    Outputs a list of x-breakpoints splitting the signals into several segments.
    Amounts to discretizing our signals along the x-axis.

    Parameters
    ----------
    uniform_or_adaptive : {'uniform', 'adaptive'}, default='uniform'
        Segmentation type. Possible values:

        - "uniform" : uniform segmentation given a number of segments.
        - "adaptive" : adaptive segmentation on the mean or slope features,
            given a number of segments (breakpoints) or a value of penalty
            factor.

    mean_or_slope : {None, 'mean', 'slope'}, default=None
        Feature on which the adaptive segmentation is done. Possible values:

        - None: if the segmentation is uniform.
        - 'mean': adaptive segmentation on the mean feature.
        - 'slope': adaptive segmentation on the slope feature.

    n_segments : int or None, default=8
        Number of segments for uniform segmentation or adaptive
            segmentation for number of segments (breakpoints).
        For adaptive segmentation, if n_segments=None, then adaptive
            segmentation based on the penalty is performed.

    pen_factor : float or None, default=None
        If is Number of segments for uniform segmentation or adaptive
            segmentation for number of segments (breakpoints).
        For adaptive segmentation, if pen_factor=None, then adaptive
            segmentation based on the number of breakpoints (segments) is
            performed.
    """

    def __init__(
        self,
        uniform_or_adaptive: str = "uniform",
        mean_or_slope: str = None,
        n_segments: int = 8,
        pen_factor: float = None,
    ) -> None:

        # Unit tests on the parameters:

        err_msg = (
            f"Choose 'uniform' or 'adaptive', not {uniform_or_adaptive}."
        )
        assert uniform_or_adaptive in ["uniform", "adaptive"], err_msg

        if uniform_or_adaptive == "uniform":
            err_msg = "For uniform segmentation, specify `n_segments`."
            assert n_segments is not None, err_msg

            err_msg = "For uniform segmentation, do not specify `pen_factor`."
            assert pen_factor is None, err_msg

            err_msg = (
                "For uniform segmentation, do not specify `mean_or_slope`."
            )
            assert mean_or_slope is None, err_msg

        if uniform_or_adaptive == "adaptive":
            err_msg = f"Choose 'mean' or 'slope', not {mean_or_slope}."
            assert mean_or_slope in ["mean", "slope"], err_msg

            err_msg = "Specify `n_segments` or `pen_factor`."
            assert (n_segments is not None) or (
                pen_factor is not None), err_msg

            err_msg = "Specify either `n_segments` or `pen_factor`."
            assert (n_segments is None) or (pen_factor is None), err_msg

        # Initializing the parameters
        self.uniform_or_adaptive = uniform_or_adaptive
        self.mean_or_slope = mean_or_slope
        self.n_segments = n_segments
        self.pen_factor = pen_factor

    def fit(self, *args, **kwargs):
        return self

    def transform(self, list_of_multivariate_signals):
        """Return list of change-points for each multivariate signal."""

        multivariate_signal = list_of_multivariate_signals[0]
        if multivariate_signal.shape[0] < multivariate_signal.shape[1]:
            raise ValueError("The shape of a multivariate signal seems wrong.")

        if self.uniform_or_adaptive == "uniform":
            list_of_bkps = [
                self.transform_uniform(multivariate_signal.shape[0]) for multivariate_signal in list_of_multivariate_signals
            ]
        elif self.uniform_or_adaptive == "adaptive":
            list_of_bkps = [
                self.transform_adaptive(multivariate_signal) for multivariate_signal in list_of_multivariate_signals
            ]

        b_transform_segmentation = Bunch(
            list_of_multivariate_signals=list_of_multivariate_signals,
            list_of_bkps=list_of_bkps,
        )
        return b_transform_segmentation

    def transform_uniform(self, n_samples):
        """Return list of equally spaced change-point indexes."""
        bkps = np.linspace(
            1, n_samples, num=self.n_segments + 1, dtype=int) - 1
        bkps = bkps[1:]
        bkps[-1] = n_samples
        return bkps.flatten().tolist()

    def transform_adaptive(self, multivariate_signal):
        """Return change-points indexes for mean or slope shifts."""

        if self.mean_or_slope == "slope":
            # BottomUp for slope
            algo = rpt.BottomUp(model="clinear", jump=1).fit(multivariate_signal)
        elif self.mean_or_slope == "mean":
            # Dynp for mean
            algo = rpt.KernelCPD(kernel="linear", jump=1).fit(multivariate_signal)

        if self.n_segments is not None:
            n_bkps = self.n_segments - 1
            bkps = algo.predict(n_bkps=n_bkps)
        elif self.pen_factor is not None:
            pen_value = self.get_penalty_value(multivariate_signal)
            bkps = algo.predict(pen=pen_value)

        return bkps

    def get_penalty_value(self, signal):
        """Return penalty value for a single signal."""
        n_samples = signal.shape[0]
        return self.pen_factor * np.log(n_samples)
