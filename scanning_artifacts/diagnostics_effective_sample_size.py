import numpy as np


def effective_x_information(
    corr_even,
    corr_odd,
    max_lag=None,
):
    """
    Estimate local effective independent-sample density along x.

    Returns
    -------
    redundancy : (X,)
        Number of nominal pixels per effective independent degree
        of freedom, approximately.

    independent_fraction : (X,)
        Effective independent degrees of freedom per nominal pixel.

    effective_spacing : (X,)
        Same as redundancy, interpreted as an approximate effective
        x-pixel spacing in nominal pixels.
    """

    corr = 0.5 * (np.asarray(corr_even) + np.asarray(corr_odd))

    L = (corr.shape[0] - 1) // 2

    if max_lag is None:
        max_lag = L

    max_lag = min(max_lag, L)

    # Positive and negative lags.
    # Average both directions before squaring.
    rho2_sum = np.zeros(corr.shape[1])
    n_valid = np.zeros(corr.shape[1])

    for lag in range(1, max_lag + 1):

        c_plus = corr[L + lag]
        c_minus = corr[L - lag]

        c = np.nanmean(
            np.stack([c_plus, c_minus]),
            axis=0,
        )

        valid = np.isfinite(c)

        rho2_sum[valid] += c[valid] ** 2
        n_valid[valid] += 1

    redundancy = 1.0 + 2.0 * rho2_sum

    independent_fraction = 1.0 / redundancy

    return {
        "redundancy": redundancy,
        "independent_fraction": independent_fraction,
        "effective_spacing": redundancy,
    }
