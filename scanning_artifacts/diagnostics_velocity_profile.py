import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares


def resonant_velocity_profile(x, baseline, amplitude, x0, A):
    """
    Correlation-width proxy based on resonant scanner velocity.

        y(x) = baseline
             + amplitude * sqrt(1 - ((x - x0) / A)^2)

    A is the distance from x0 to the resonant turning point.
    """

    u = (x - x0) / A

    inside = np.maximum(1.0 - u**2, 0.0)

    return baseline + amplitude * np.sqrt(inside)


def fit_resonant_profile(
    x,
    y,
    edge_margin=15,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    X = x.max() - x.min()

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > x.min() + edge_margin)
        & (x < x.max() - edge_margin)
    )

    xx = x[valid]
    yy = y[valid]

    xmid = 0.5 * (x.min() + x.max())

    # Initial guesses
    baseline0 = np.nanpercentile(yy, 10)
    amplitude0 = np.nanmax(yy) - baseline0

    # Turning point initially slightly outside recorded FOV
    A0 = 0.6 * X

    p0 = np.array(
        [
            baseline0,
            amplitude0,
            xmid,
            A0,
        ]
    )

    # A must extend far enough to contain all fitted points.
    minimum_A = 0.51 * X

    lower = [
        0,
        0,
        xmid - 0.2 * X,
        minimum_A,
    ]

    upper = [
        10 * np.nanmax(yy),
        10 * np.nanmax(yy),
        xmid + 0.2 * X,
        5 * X,
    ]

    def residuals(p):
        return resonant_velocity_profile(xx, *p) - yy

    fit = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.05,
    )

    pred = resonant_velocity_profile(xx, *fit.x)

    residual = yy - pred

    ss_res = np.sum(residual**2)
    ss_tot = np.sum((yy - np.mean(yy)) ** 2)

    r2 = 1 - ss_res / ss_tot

    return {
        "params": fit.x,
        "x": xx,
        "y": yy,
        "prediction": pred,
        "residuals": residual,
        "r2": r2,
        "fit": fit,
    }


def quadratic_profile(x, baseline, amplitude, x0, scale):
    u = (x - x0) / scale
    return baseline + amplitude * (1 - u**2)


def fit_quadratic_profile(
    x,
    y,
    edge_margin=15,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    X = x.max() - x.min()

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > x.min() + edge_margin)
        & (x < x.max() - edge_margin)
    )

    xx = x[valid]
    yy = y[valid]

    xmid = 0.5 * (x.min() + x.max())

    p0 = [
        np.nanpercentile(yy, 10),
        np.nanmax(yy) - np.nanmin(yy),
        xmid,
        X,
    ]

    lower = [
        -10,
        0,
        xmid - 0.2 * X,
        0.3 * X,
    ]

    upper = [
        10 * np.nanmax(yy),
        10 * np.nanmax(yy),
        xmid + 0.2 * X,
        10 * X,
    ]

    def residuals(p):
        return quadratic_profile(xx, *p) - yy

    fit = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.05,
    )

    pred = quadratic_profile(xx, *fit.x)

    rss = np.sum((yy - pred) ** 2)

    return {
        "params": fit.x,
        "x": xx,
        "y": yy,
        "prediction": pred,
        "residuals": yy - pred,
        "rss": rss,
    }


def fit_dataset(results, edge_margin=20):

    even = results["metric_even"]["corr_area"]
    odd = results["metric_odd"]["corr_area"]

    width = 0.5 * (even + odd)

    x = np.arange(len(width))

    resonant = fit_resonant_profile(
        x,
        width,
        edge_margin=edge_margin,
    )

    quadratic = fit_quadratic_profile(
        x,
        width,
        edge_margin=edge_margin,
    )

    return x, width, resonant, quadratic


def resonant_velocity_profile_b1(
    x,
    amplitude,
    x0,
    A,
):
    u = (x - x0) / A

    inside = np.maximum(1.0 - u**2, 0.0)

    return 1.0 + amplitude * np.sqrt(inside)


def fit_resonant_profile_b1(
    x,
    y,
    edge_margin=20,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    X = x.max() - x.min()

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > x.min() + edge_margin)
        & (x < x.max() - edge_margin)
    )

    xx = x[valid]
    yy = y[valid]

    xmid = 0.5 * (x.min() + x.max())

    p0 = [
        np.nanmax(yy) - 1,
        xmid,
        0.6 * X,
    ]

    lower = [
        0,
        xmid - 0.2 * X,
        0.51 * X,
    ]

    upper = [
        20,
        xmid + 0.2 * X,
        5 * X,
    ]

    def residuals(p):
        return resonant_velocity_profile_b1(xx, *p) - yy

    fit = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.05,
    )

    pred = resonant_velocity_profile_b1(xx, *fit.x)

    residual = yy - pred

    rss = np.sum(residual**2)

    ss_tot = np.sum((yy - np.mean(yy)) ** 2)

    return {
        "params": fit.x,
        "x": xx,
        "prediction": pred,
        "residuals": residual,
        "rss": rss,
        "r2": 1 - rss / ss_tot,
    }


def cosine_velocity_profile(
    x,
    baseline,
    amplitude,
    x0,
    k,
):
    return baseline + amplitude * np.cos(k * (x - x0))


def fit_cosine_velocity(
    x,
    y,
    edge_margin=20,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    X = x.max() - x.min()

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > x.min() + edge_margin)
        & (x < x.max() - edge_margin)
    )

    xx = x[valid]
    yy = y[valid]

    xmid = 0.5 * (x.min() + x.max())

    p0 = [
        1.0,
        np.nanmax(yy) - 1.0,
        xmid,
        np.pi / X,
    ]

    lower = [
        0,
        0,
        xmid - 0.2 * X,
        0,
    ]

    upper = [
        10,
        10,
        xmid + 0.2 * X,
        2 * np.pi / X,
    ]

    def residuals(p):
        return cosine_velocity_profile(xx, *p) - yy

    fit = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.05,
    )

    pred = cosine_velocity_profile(xx, *fit.x)

    residual = yy - pred
    rss = np.sum(residual**2)
    ss_tot = np.sum((yy - np.mean(yy)) ** 2)

    return {
        "params": fit.x,
        "x": xx,
        "prediction": pred,
        "residuals": residual,
        "rss": rss,
        "r2": 1 - rss / ss_tot,
    }
