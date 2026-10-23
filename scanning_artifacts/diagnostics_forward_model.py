import numpy as np

from scipy.optimize import least_squares


def correlation_observations(
    corr_even,
    corr_odd,
    max_fit_lag=15,
    x_margin=15,
):
    """
    Turn measured C(x, dx) into flattened fit observations.

    Only positive lags are used because negative lags contain
    the same pixel pairs.
    """

    corr = 0.5 * (np.asarray(corr_even) + np.asarray(corr_odd))

    L = (corr.shape[0] - 1) // 2
    X = corr.shape[1]

    xs = []
    lags = []
    values = []

    for lag in range(1, min(L, max_fit_lag) + 1):

        x = np.arange(x_margin, X - lag - x_margin)

        y = corr[L + lag, x]

        good = np.isfinite(y)

        xs.append(x[good])
        lags.append(np.full(np.sum(good), lag))
        values.append(y[good])

    return (
        np.concatenate(xs),
        np.concatenate(lags),
        np.concatenate(values),
    )


def resonant_phase(x, x0, A):
    """
    Acquisition phase corresponding to spatial coordinate x.

    Valid for the monotonic half-sweep of the resonant scanner.
    """
    u = (np.asarray(x) - x0) / A

    if np.any(np.abs(u) >= 1):
        raise ValueError("Some x values lie outside resonant turning points.")

    return np.arcsin(u)


def phase_sinc_correlation(
    x,
    lag,
    x0,
    A,
    tau_theta,
    alpha,
):
    theta1 = resonant_phase(x, x0, A)

    theta2 = resonant_phase(x + lag, x0, A)

    dtheta = theta2 - theta1

    # np.sinc(z) = sin(pi*z)/(pi*z)
    return alpha * np.sinc(dtheta / tau_theta)


def fit_phase_sinc_fixed_geometry(
    corr_even,
    corr_odd,
    x0=254.0,
    A=271.67,
    max_fit_lag=15,
    x_margin=15,
    tau_init=None,
):
    x, lag, y = correlation_observations(
        corr_even,
        corr_odd,
        max_fit_lag=max_fit_lag,
        x_margin=x_margin,
    )

    if tau_init is None:
        # First zero looked roughly ~4 pixels near center.
        # dtheta/dx = 1/A at center.
        tau_init = 4 / A

    p0 = [
        tau_init,
        0.95,
    ]

    lower = [
        1e-4,
        0.0,
    ]

    upper = [
        0.2,
        1.0,
    ]

    def residuals(p):
        tau_theta, alpha = p

        prediction = phase_sinc_correlation(
            x,
            lag,
            x0=x0,
            A=A,
            tau_theta=tau_theta,
            alpha=alpha,
        )

        return prediction - y

    result = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.02,
    )

    tau_theta, alpha = result.x

    prediction = phase_sinc_correlation(
        x,
        lag,
        x0=x0,
        A=A,
        tau_theta=tau_theta,
        alpha=alpha,
    )

    residual = y - prediction

    rss = np.sum(residual**2)

    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - rss / ss_tot

    return {
        "x": x,
        "lag": lag,
        "y": y,
        "prediction": prediction,
        "residual": residual,
        "tau_theta": tau_theta,
        "alpha": alpha,
        "x0": x0,
        "A": A,
        "rss": rss,
        "r2": r2,
        "result": result,
    }


def predicted_correlation_map(
    X,
    max_lag,
    x0,
    A,
    tau_theta,
    alpha,
):
    C = np.full((2 * max_lag + 1, X), np.nan)

    C[max_lag] = 1.0

    for lag in range(1, max_lag + 1):

        x = np.arange(0, X - lag)

        c = phase_sinc_correlation(
            x,
            lag,
            x0=x0,
            A=A,
            tau_theta=tau_theta,
            alpha=alpha,
        )

        C[max_lag + lag, :-lag] = c

        # same physical pixel pairs, represented
        # from the opposite reference position
        C[max_lag - lag, lag:] = c

    return C


def first_zero_profile(
    X,
    x0,
    A,
    tau_theta,
):
    x = np.arange(X)

    inside = A**2 - (x - x0) ** 2

    result = np.full(X, np.nan)

    good = inside > 0

    result[good] = tau_theta * np.sqrt(inside[good])

    return result


def phase_collapse_data(
    corr_even,
    corr_odd,
    x0=254.0,
    A=271.67,
    max_lag=15,
    x_margin=20,
):
    """
    Convert measured spatial correlations C(x, dx)
    into resonant phase-coordinate correlations C(dtheta).

    Returns flattened arrays containing:
        dtheta
        correlation
        x
        lag
    """

    corr = 0.5 * (np.asarray(corr_even) + np.asarray(corr_odd))

    L = (corr.shape[0] - 1) // 2
    X = corr.shape[1]

    dtheta_all = []
    corr_all = []
    x_all = []
    lag_all = []

    for lag in range(1, min(L, max_lag) + 1):

        x = np.arange(x_margin, X - lag - x_margin)

        u1 = (x - x0) / A
        u2 = (x + lag - x0) / A

        good = (np.abs(u1) < 1) & (np.abs(u2) < 1)

        x = x[good]
        u1 = u1[good]
        u2 = u2[good]

        dtheta = np.arcsin(u2) - np.arcsin(u1)

        c = corr[L + lag, x]

        valid = np.isfinite(c)

        dtheta_all.append(dtheta[valid])
        corr_all.append(c[valid])
        x_all.append(x[valid])

        lag_all.append(np.full(np.sum(valid), lag))

    return {
        "dtheta": np.concatenate(dtheta_all),
        "corr": np.concatenate(corr_all),
        "x": np.concatenate(x_all),
        "lag": np.concatenate(lag_all),
    }


def estimate_phase_kernel(
    collapsed,
    n_bins=150,
    min_count=10,
):
    dtheta = collapsed["dtheta"]
    corr = collapsed["corr"]

    edges = np.linspace(
        0,
        np.nanmax(dtheta),
        n_bins + 1,
    )

    centers = 0.5 * (edges[:-1] + edges[1:])

    mean = np.full(n_bins, np.nan)
    std = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):

        select = (dtheta >= edges[i]) & (dtheta < edges[i + 1])

        count[i] = np.sum(select)

        if count[i] >= min_count:
            mean[i] = np.nanmean(corr[select])

            std[i] = np.nanstd(corr[select])

    return centers, mean, std, count


from scipy.optimize import least_squares
import numpy as np


def damped_sinc_kernel(
    dtheta,
    alpha,
    tau,
    sigma,
):
    return alpha * np.sinc(dtheta / tau) * np.exp(-0.5 * (dtheta / sigma) ** 2)


def fit_damped_sinc(collapsed):

    x = np.asarray(collapsed["dtheta"], dtype=float)

    y = np.asarray(collapsed["corr"], dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    p0 = [
        0.95,  # alpha
        0.014,  # first zero
        0.05,  # damping scale
    ]

    bounds = (
        [0.0, 0.001, 0.005],
        [1.0, 0.1, 1.0],
    )

    def residuals(p):

        return damped_sinc_kernel(x, *p) - y

    fit = least_squares(
        residuals,
        p0,
        bounds=bounds,
        loss="soft_l1",
        f_scale=0.02,
    )

    pred = damped_sinc_kernel(x, *fit.x)

    rss = np.sum((y - pred) ** 2)

    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - rss / ss_tot

    return {
        "alpha": fit.x[0],
        "tau": fit.x[1],
        "sigma": fit.x[2],
        "rss": rss,
        "r2": r2,
        "fit": fit,
    }


def resonant_geometry_from_margins(X, margin_left, margin_right):
    """
    Convert distances of the resonant turning points outside the
    recorded image into x0 and A.

    left turning point  = -margin_left
    right turning point = (X - 1) + margin_right
    """
    left = -margin_left
    right = (X - 1) + margin_right

    x0 = 0.5 * (left + right)
    A = 0.5 * (right - left)

    return x0, A


def joint_phase_covariance_model(
    x,
    lag,
    X,
    margin_left,
    margin_right,
    alpha,
    tau,
    sigma,
):
    x0, A = resonant_geometry_from_margins(
        X,
        margin_left,
        margin_right,
    )

    theta1 = np.arcsin((x - x0) / A)

    theta2 = np.arcsin((x + lag - x0) / A)

    dtheta = theta2 - theta1

    return damped_sinc_kernel(
        dtheta,
        alpha,
        tau,
        sigma,
    )


def fit_joint_resonant_covariance(
    corr_even,
    corr_odd,
    max_fit_lag=15,
    x_margin=20,
    x0_init=254.0,
    A_init=271.67,
    alpha_init=0.94,
    tau_init=0.0157,
    sigma_init=0.037,
):
    corr_even = np.asarray(corr_even)
    X = corr_even.shape[1]

    x, lag, y = correlation_observations(
        corr_even,
        corr_odd,
        max_fit_lag=max_fit_lag,
        x_margin=x_margin,
    )

    # Convert initial x0/A into turning-point margins
    left0 = x0_init - A_init
    right0 = x0_init + A_init

    margin_left0 = max(-left0, 0.1)
    margin_right0 = max(right0 - (X - 1), 0.1)

    p0 = [
        margin_left0,
        margin_right0,
        alpha_init,
        tau_init,
        sigma_init,
    ]

    lower = [
        1e-3,  # left turning point just outside FOV
        1e-3,  # right
        0.0,
        1e-4,
        1e-3,
    ]

    upper = [
        4 * X,
        4 * X,
        1.0,
        0.1,
        0.5,
    ]

    def residuals(p):

        (
            margin_left,
            margin_right,
            alpha,
            tau,
            sigma,
        ) = p

        pred = joint_phase_covariance_model(
            x,
            lag,
            X,
            margin_left,
            margin_right,
            alpha,
            tau,
            sigma,
        )

        return pred - y

    result = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.02,
        max_nfev=3000,
    )

    (
        margin_left,
        margin_right,
        alpha,
        tau,
        sigma,
    ) = result.x

    x0, A = resonant_geometry_from_margins(
        X,
        margin_left,
        margin_right,
    )

    pred = joint_phase_covariance_model(
        x,
        lag,
        X,
        *result.x,
    )

    residual = y - pred

    rss = np.sum(residual**2)

    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - rss / ss_tot

    return {
        "x0": x0,
        "A": A,
        "left_turn": x0 - A,
        "right_turn": x0 + A,
        "margin_left": margin_left,
        "margin_right": margin_right,
        "alpha": alpha,
        "tau": tau,
        "sigma": sigma,
        "rss": rss,
        "r2": r2,
        "x": x,
        "lag": lag,
        "y": y,
        "prediction": pred,
        "residual": residual,
        "result": result,
    }


def fit_kernel_at_fixed_geometry(
    x,
    lag,
    y,
    x0,
    A,
    p0=(0.94, 0.0157, 0.037),
    max_nfev=200,
):
    # geometry validity
    u1 = (x - x0) / A
    u2 = (x + lag - x0) / A

    if np.any(np.abs(u1) >= 1) or np.any(np.abs(u2) >= 1):
        return np.nan, None

    theta1 = np.arcsin(u1)
    theta2 = np.arcsin(u2)

    dtheta = theta2 - theta1

    def residuals(p):
        alpha, tau, sigma = p

        return (
            damped_sinc_kernel(
                dtheta,
                alpha,
                tau,
                sigma,
            )
            - y
        )

    result = least_squares(
        residuals,
        p0,
        bounds=(
            [0, 1e-4, 1e-3],
            [1.0, 0.1, 0.5],
        ),
        loss="soft_l1",
        f_scale=0.02,
        max_nfev=max_nfev,
    )

    pred = damped_sinc_kernel(dtheta, *result.x)

    rss = np.sum((y - pred) ** 2)

    return rss, result.x


def geometry_profile_grid(
    corr_even,
    corr_odd,
    x0_range=(245, 263),
    A_range=(250, 310),
    n_x0=37,
    n_A=61,
    max_fit_lag=15,
    x_margin=20,
    x_stride=4,
):
    x, lag, y = correlation_observations(
        corr_even,
        corr_odd,
        max_fit_lag=max_fit_lag,
        x_margin=x_margin,
    )

    # Subsample x to make the profile calculation much faster.
    select = (x.astype(int) % x_stride) == 0

    x = x[select]
    lag = lag[select]
    y = y[select]

    x0_values = np.linspace(*x0_range, n_x0)

    A_values = np.linspace(*A_range, n_A)

    rss = np.full(
        (len(A_values), len(x0_values)),
        np.nan,
    )

    kernel_params = np.full(
        (len(A_values), len(x0_values), 3),
        np.nan,
    )

    # Good starting point
    p0 = np.array(
        [
            0.94,
            0.0157,
            0.037,
        ]
    )

    for ia, A in enumerate(A_values):

        for ix, x0 in enumerate(x0_values):

            score, pars = fit_kernel_at_fixed_geometry(
                x,
                lag,
                y,
                x0=x0,
                A=A,
                p0=p0,
            )

            rss[ia, ix] = score

            if pars is not None:
                kernel_params[ia, ix] = pars

    return {
        "x0": x0_values,
        "A": A_values,
        "rss": rss,
        "kernel_params": kernel_params,
    }


def nonparametric_collapse_score(
    x,
    lag,
    corr,
    x0,
    A,
    n_bins=100,
    min_count=20,
):
    """
    Score how well C(x, dx) collapses to a single function C(dtheta).

    Lower is better.

    No parametric assumption is made about the kernel shape.
    """

    u1 = (x - x0) / A
    u2 = (x + lag - x0) / A

    valid = np.isfinite(corr) & (np.abs(u1) < 1) & (np.abs(u2) < 1)

    if np.sum(valid) < 100:
        return np.nan

    u1 = u1[valid]
    u2 = u2[valid]
    c = corr[valid]

    dtheta = np.arcsin(u2) - np.arcsin(u1)

    edges = np.linspace(
        np.nanmin(dtheta),
        np.nanmax(dtheta),
        n_bins + 1,
    )

    bin_index = np.digitize(dtheta, edges) - 1

    sse = 0.0
    n_total = 0

    for b in range(n_bins):

        select = bin_index == b

        n = np.sum(select)

        if n < min_count:
            continue

        values = c[select]

        mean = np.mean(values)

        sse += np.sum((values - mean) ** 2)

        n_total += n

    if n_total == 0:
        return np.nan

    return sse / n_total
