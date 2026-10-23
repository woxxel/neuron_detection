import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation


def preprocess_registration_image(
    image,
    highpass_sigma=12.0,
    lowpass_sigma=1.0,
):
    """
    Prepare one 2D frame for image registration.

    Removes broad spatial background and slightly suppresses pixel noise.
    """
    image = np.asarray(image, dtype=np.float32)

    if lowpass_sigma is not None:
        image = gaussian_filter(image, lowpass_sigma)

    if highpass_sigma is not None:
        background = gaussian_filter(image, highpass_sigma)
        image = image - background

    image -= np.nanmean(image)

    std = np.nanstd(image)
    if std > 0:
        image /= std

    return image


def temporal_bin_movie(movie, bin_size=3):
    """
    Average consecutive frames.

    movie : (T, Y, X)
    """
    movie = np.asarray(movie)

    if bin_size <= 1:
        return movie

    n = movie.shape[0] // bin_size

    movie = movie[: n * bin_size]

    return movie.reshape(n, bin_size, *movie.shape[1:]).mean(axis=1)


def make_registration_reference(movie):
    """
    Robust structural reference image.
    """
    return np.nanmedian(movie, axis=0)


def estimate_global_motion(
    movie,
    reference,
    upsample_factor=20,
):
    """
    Estimate rigid 2D shift of every frame relative to reference.

    Returns
    -------
    dy, dx : arrays of shape (T,)
        Shift that needs to be applied to each frame to align it
        with the reference.
    error : registration error estimate
    """

    T = movie.shape[0]

    dy = np.full(T, np.nan)
    dx = np.full(T, np.nan)
    error = np.full(T, np.nan)

    ref = preprocess_registration_image(reference)

    for t in range(T):

        frame = preprocess_registration_image(movie[t])

        shift, err, _ = phase_cross_correlation(
            ref,
            frame,
            upsample_factor=upsample_factor,
        )

        dy[t], dx[t] = shift
        error[t] = err

    return dy, dx, error


def estimate_local_x_motion(
    movie,
    reference,
    strip_width=96,
    step=24,
    y_margin=20,
    max_abs_shift=10,
    upsample_factor=20,
):
    """
    Estimate x-motion independently in overlapping vertical strips.

    Returns
    -------
    x_centers : (Nstrips,)
    dx_local  : (T, Nstrips)
    errors    : (T, Nstrips)
    """

    T, Y, X = movie.shape

    x_starts = np.arange(0, X - strip_width + 1, step)

    x_centers = x_starts + strip_width / 2

    dx_local = np.full((T, len(x_starts)), np.nan)

    errors = np.full_like(dx_local, np.nan)

    for j, x0 in enumerate(x_starts):

        x1 = x0 + strip_width

        ref_patch = reference[y_margin : Y - y_margin, x0:x1]

        ref_patch = preprocess_registration_image(ref_patch)

        for t in range(T):

            patch = movie[t, y_margin : Y - y_margin, x0:x1]

            patch = preprocess_registration_image(patch)

            shift, err, _ = phase_cross_correlation(
                ref_patch,
                patch,
                upsample_factor=upsample_factor,
            )

            dy, dx = shift

            if abs(dx) <= max_abs_shift and abs(dy) <= max_abs_shift:
                dx_local[t, j] = dx
                errors[t, j] = err

    return x_centers, dx_local, errors


def robust_linear_fit(x, y, n_iter=4, clip=3.5):
    """
    Robust linear regression y = intercept + slope*x
    using iterative MAD clipping.
    """

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 10:
        return np.nan, np.nan, np.nan, 0

    keep = np.ones(len(x), dtype=bool)

    for _ in range(n_iter):

        slope, intercept = np.polyfit(x[keep], y[keep], 1)

        residual = y - (intercept + slope * x)

        median = np.median(residual[keep])

        mad = np.median(np.abs(residual[keep] - median))

        sigma = 1.4826 * mad

        if sigma == 0:
            break

        keep = np.abs(residual - median) < clip * sigma

    slope, intercept = np.polyfit(x[keep], y[keep], 1)

    pred = intercept + slope * x[keep]

    ss_res = np.sum((y[keep] - pred) ** 2)

    ss_tot = np.sum((y[keep] - np.mean(y[keep])) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return slope, intercept, r2, keep.sum()


def estimate_local_scale(
    dx_global,
    dx_local,
):
    """
    Regress local displacement against global displacement.

    Returns relative local x-scale.
    """

    n_strips = dx_local.shape[1]

    slope = np.full(n_strips, np.nan)
    intercept = np.full(n_strips, np.nan)
    r2 = np.full(n_strips, np.nan)
    n = np.zeros(n_strips, dtype=int)

    for j in range(n_strips):

        (
            slope[j],
            intercept[j],
            r2[j],
            n[j],
        ) = robust_linear_fit(dx_global, dx_local[:, j])

    # Relative scale: median interior value = 1
    good = np.isfinite(slope) & (r2 > 0.3)

    scale = slope.copy()

    scale /= np.nanmedian(scale[good])

    return {
        "scale": scale,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "n": n,
    }


def resonant_motion_scale_diagnostic(
    movie,
    bin_size=3,
    strip_width=96,
    step=24,
    y_margin=20,
    upsample_factor=20,
):
    """
    Measure x-dependent geometric scale using natural tissue motion.

    movie must have shape (T, Y, X).
    """

    # Temporal binning
    m = temporal_bin_movie(movie, bin_size=bin_size)

    # Reference
    reference = make_registration_reference(m)

    # Global motion
    dy_global, dx_global, global_error = estimate_global_motion(
        m,
        reference,
        upsample_factor=upsample_factor,
    )

    # Local motion
    x_centers, dx_local, local_error = estimate_local_x_motion(
        m,
        reference,
        strip_width=strip_width,
        step=step,
        y_margin=y_margin,
        upsample_factor=upsample_factor,
    )

    # Local geometric scale
    fit = estimate_local_scale(dx_global, dx_local)

    return {
        "movie_binned": m,
        "reference": reference,
        "dx_global": dx_global,
        "dy_global": dy_global,
        "global_error": global_error,
        "x_centers": x_centers,
        "dx_local": dx_local,
        "local_error": local_error,
        **fit,
    }
