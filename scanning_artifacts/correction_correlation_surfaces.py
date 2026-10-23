import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter


def plot_correlation_surface(
    surface,
    shifts,
    metrics=None,
):
    plt.figure(figsize=(6, 5))

    plt.imshow(
        surface,
        origin="lower",
        extent=[
            shifts[0] - 0.5,
            shifts[-1] + 0.5,
            shifts[0] - 0.5,
            shifts[-1] + 0.5,
        ],
        aspect="equal",
    )

    plt.colorbar(label="normalized correlation")

    if metrics is not None:
        plt.scatter(
            metrics["dx"],
            metrics["dy"],
            marker="x",
            s=100,
        )

        plt.title(
            "dx={:.2f}, dy={:.2f}, "
            "C={:.3f}, margin={:.3f}".format(
                metrics["dx"],
                metrics["dy"],
                metrics["peak"],
                metrics["peak_margin"],
            )
        )

    plt.xlabel("x shift applied to moving image [px]")
    plt.ylabel("y shift applied to moving image [px]")

    plt.show()


def analyze_movie_correlation_surfaces(
    movie,
    frame_lag=1,
    max_shift=5,
    lowpass_sigma=1.0,
    highpass_sigma=20.0,
):
    """
    Analyze translation identifiability through a movie.

    `movie` should preferably already be temporally binned.
    """

    T = movie.shape[0]

    results = []

    for t in range(T - frame_lag):

        surface, shifts = correlation_surface(
            movie[t],
            movie[t + frame_lag],
            max_shift=max_shift,
            lowpass_sigma=lowpass_sigma,
            highpass_sigma=highpass_sigma,
        )

        metrics = correlation_surface_metrics(
            surface,
            shifts,
        )

        metrics["frame"] = t
        metrics["surface"] = surface

        results.append(metrics)

    return results, shifts


def preprocess_corr_image(
    image,
    lowpass_sigma=1.0,
    highpass_sigma=20.0,
):
    """
    Preprocessing only for estimating spatial translation.

    Parameters
    ----------
    image : (Y, X)

    lowpass_sigma :
        Small smoothing to suppress pixel-scale noise.

    highpass_sigma :
        Remove broad illumination/background gradients.
        Set to None to disable.
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if lowpass_sigma is not None:
        image = gaussian_filter(
            image,
            lowpass_sigma,
        )

    if highpass_sigma is not None:
        image = image - gaussian_filter(
            image,
            highpass_sigma,
        )

    return image


def normalized_corr(a, b):
    """
    Pearson-like normalized correlation of two image patches.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    valid = np.isfinite(a) & np.isfinite(b)

    a = a[valid]
    b = b[valid]

    if len(a) < 10:
        return np.nan

    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))

    if denom == 0:
        return np.nan

    return np.sum(a * b) / denom


def correlation_surface(
    reference,
    moving,
    max_shift=5,
    lowpass_sigma=1.0,
    highpass_sigma=20.0,
):
    """
    Calculate normalized correlation for all integer translations
    in [-max_shift, +max_shift] in x and y.

    Sign convention
    ---------------
    dx, dy are the shifts that would be APPLIED TO `moving`
    to align it to `reference`.

    Returns
    -------
    surface : (2*max_shift+1, 2*max_shift+1)
        rows = dy
        cols = dx

    shifts : array
        [-max_shift, ..., +max_shift]
    """

    A = preprocess_corr_image(
        reference,
        lowpass_sigma=lowpass_sigma,
        highpass_sigma=highpass_sigma,
    )

    B = preprocess_corr_image(
        moving,
        lowpass_sigma=lowpass_sigma,
        highpass_sigma=highpass_sigma,
    )

    if A.shape != B.shape:
        raise ValueError("reference and moving image must have same shape")

    Y, X = A.shape
    m = max_shift

    # Fixed reference crop.
    ref = A[
        m : Y - m,
        m : X - m,
    ]

    shifts = np.arange(
        -m,
        m + 1,
    )

    surface = np.full(
        (len(shifts), len(shifts)),
        np.nan,
    )

    for iy, dy in enumerate(shifts):
        for ix, dx in enumerate(shifts):

            # If (dy, dx) is the displacement applied to B,
            # B_shifted(y,x) = B(y-dy, x-dx)
            test = B[
                m - dy : Y - m - dy,
                m - dx : X - m - dx,
            ]

            surface[iy, ix] = normalized_corr(
                ref,
                test,
            )

    return surface, shifts


def correlation_surface_metrics(
    surface,
    shifts,
    exclusion_radius=1,
):
    """
    Extract peak position and several measures of identifiability.
    """

    surface = np.asarray(surface)

    iy, ix = np.unravel_index(
        np.nanargmax(surface),
        surface.shape,
    )

    dy = float(shifts[iy])
    dx = float(shifts[ix])

    peak = surface[iy, ix]

    # -----------------------------------------
    # Subpixel parabolic interpolation
    # -----------------------------------------

    dx_sub = dx
    dy_sub = dy

    if 0 < ix < surface.shape[1] - 1:
        ym = surface[iy, ix - 1]
        y0 = surface[iy, ix]
        yp = surface[iy, ix + 1]

        denom = ym - 2 * y0 + yp

        if np.isfinite(denom) and denom != 0:
            dx_sub += 0.5 * (ym - yp) / denom

    if 0 < iy < surface.shape[0] - 1:
        ym = surface[iy - 1, ix]
        y0 = surface[iy, ix]
        yp = surface[iy + 1, ix]

        denom = ym - 2 * y0 + yp

        if np.isfinite(denom) and denom != 0:
            dy_sub += 0.5 * (ym - yp) / denom

    # -----------------------------------------
    # Best unrelated competitor
    # -----------------------------------------

    yy, xx = np.indices(surface.shape)

    dist = np.sqrt((yy - iy) ** 2 + (xx - ix) ** 2)

    competitor = surface.copy()

    competitor[dist <= exclusion_radius] = np.nan

    second_peak = np.nanmax(competitor)

    peak_margin = peak - second_peak

    # -----------------------------------------
    # Local curvature
    # -----------------------------------------

    curvature_x = np.nan
    curvature_y = np.nan

    if 0 < ix < surface.shape[1] - 1:
        curvature_x = 2 * peak - surface[iy, ix - 1] - surface[iy, ix + 1]

    if 0 < iy < surface.shape[0] - 1:
        curvature_y = 2 * peak - surface[iy - 1, ix] - surface[iy + 1, ix]

    return {
        "dx": dx_sub,
        "dy": dy_sub,
        "peak": peak,
        "second_peak": second_peak,
        "peak_margin": peak_margin,
        "curvature_x": curvature_x,
        "curvature_y": curvature_y,
    }
