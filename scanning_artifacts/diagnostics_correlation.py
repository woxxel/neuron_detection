import numpy as np
from scipy.ndimage import gaussian_filter


def diagnose_resonant_scan(
    movie,
    max_lag=15,
    corr_threshold=0.2,
    phase_window=64,
    phase_step=16,
    phase_max_shift=8,
):
    z = frame_difference_noise(movie)

    # ---------------------------
    # Noise correlations
    # ---------------------------

    corr_even = spatial_noise_correlation(z, parity=0, max_lag=max_lag)

    corr_odd = spatial_noise_correlation(z, parity=1, max_lag=max_lag)

    metric_even = correlation_width_metrics(corr_even, threshold=corr_threshold)

    metric_odd = correlation_width_metrics(corr_odd, threshold=corr_threshold)

    # ---------------------------
    # Noise variance
    # ---------------------------

    noise_even = noise_std_profile(z, parity=0)
    noise_odd = noise_std_profile(z, parity=1)

    # ---------------------------
    # Bidirectional mismatch
    # ---------------------------

    template = structural_template(movie)

    phase_x, phase_shift, phase_quality = local_bidirectional_shift(
        template,
        window=phase_window,
        step=phase_step,
        max_shift=phase_max_shift,
    )

    # ---------------------------
    # Structural anisotropy
    # ---------------------------

    structure_x, structure_ratio = structural_anisotropy(movie)

    results = {
        "corr_even": corr_even,
        "corr_odd": corr_odd,
        "metric_even": metric_even,
        "metric_odd": metric_odd,
        "noise_even": noise_even,
        "noise_odd": noise_odd,
        "phase_x": phase_x,
        "phase_shift": phase_shift,
        "phase_quality": phase_quality,
        "structure_x": structure_x,
        "structure_ratio": structure_ratio,
    }

    return results


def frame_difference_noise(movie, remove_row_common_mode=True):
    """
    movie : (T, Y, X)

    Frame differences suppress slow calcium activity and emphasize
    detector noise / interpolation correlations.
    """
    z = np.diff(np.asarray(movie, dtype=np.float32), axis=0)

    if remove_row_common_mode:
        # Suppress frame-to-frame / line-to-line common fluctuations.
        z -= np.nanmedian(z, axis=2, keepdims=True)

    return z


def _temporal_corr(a, b):
    """
    Correlation over time.

    a, b shape: (T, rows, X)
    Returns:     (rows, X)
    """
    a = a - np.nanmean(a, axis=0, keepdims=True)
    b = b - np.nanmean(b, axis=0, keepdims=True)

    num = np.nansum(a * b, axis=0)

    den = np.sqrt(np.nansum(a * a, axis=0) * np.nansum(b * b, axis=0))

    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def _fisher_average(r, axis=0):
    """
    Average correlation coefficients using Fisher z transform.
    """
    r = np.clip(r, -0.9999, 0.9999)
    return np.tanh(np.nanmean(np.arctanh(r), axis=axis))


def spatial_noise_correlation(z, parity=0, max_lag=15):
    """
    Calculate C(x, dx) separately for even/odd rows.

    Parameters
    ----------
    z : (T, Y, X)
        Preferably output of frame_difference_noise().
    parity : 0 or 1
        0 = even rows
        1 = odd rows
    max_lag : int

    Returns
    -------
    corr : (2*max_lag+1, X)
        First axis corresponds to dx =
        [-max_lag, ..., 0, ..., max_lag].
    """

    rows = np.arange(parity, z.shape[1], 2)
    zz = z[:, rows, :]

    X = zz.shape[2]
    L = max_lag

    corr = np.full((2 * L + 1, X), np.nan)
    corr[L] = 1.0

    for lag in range(1, L + 1):

        # correlation x -> x+lag
        r = _temporal_corr(zz[:, :, :-lag], zz[:, :, lag:])

        corr[L + lag, :-lag] = _fisher_average(r, axis=0)

        # same pixel pairs but referenced from the other side
        corr[L - lag, lag:] = corr[L + lag, :-lag]

    return corr


def correlation_width_metrics(corr, threshold=0.2):
    """
    Returns two local measures of spatial correlation.

    threshold_width:
        first lag where correlation falls below threshold

    corr_area:
        integrated positive correlation around the pixel
    """

    L = (corr.shape[0] - 1) // 2
    X = corr.shape[1]

    # symmetrize +lag and -lag
    local_corr = np.full((L, X), np.nan)

    for lag in range(1, L + 1):
        a = corr[L + lag]
        b = corr[L - lag]

        local_corr[lag - 1] = np.nanmean(np.stack([a, b]), axis=0)

    threshold_width = np.full(X, np.nan)

    for x in range(X):
        c = local_corr[:, x]
        valid = np.isfinite(c)

        if not np.any(valid):
            continue

        below = np.where(valid & (c < threshold))[0]

        if len(below):
            threshold_width[x] = below[0] + 1
        else:
            threshold_width[x] = L

    # Similar to an integrated autocorrelation length
    corr_area = 1 + 2 * np.nansum(np.maximum(local_corr, 0), axis=0)

    return {
        "local_corr": local_corr,
        "threshold_width": threshold_width,
        "corr_area": corr_area,
    }


def structural_template(movie, sigma=1.0):
    """
    Time-averaged structural image, converted to horizontal gradients.
    """
    img = np.nanmean(movie, axis=0)

    img = gaussian_filter(img, sigma=sigma)

    # Horizontal gradient emphasizes features useful for x registration
    gx = np.gradient(img, axis=1)

    return gx


def _normalized_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    a = a - np.nanmean(a)
    b = b - np.nanmean(b)

    num = np.nansum(a * b)

    den = np.sqrt(np.nansum(a * a) * np.nansum(b * b))

    if den == 0:
        return np.nan

    return num / den


def local_bidirectional_shift(
    template,
    window=64,
    step=16,
    max_shift=8,
):
    """
    Compare odd rows to interpolated neighboring even rows.

    Returns a local odd/even displacement as function of x.
    """

    Y, X = template.shape

    odd_rows = np.arange(1, Y - 1, 2)

    odd = template[odd_rows]

    # Estimate what each odd line should look like from the
    # adjacent even lines.
    reference = (template[odd_rows - 1] + template[odd_rows + 1]) / 2

    centers = []
    best_shifts = []
    best_corrs = []

    margin = max_shift

    for x0 in range(margin, X - window - margin + 1, step):
        x1 = x0 + window

        ref = reference[:, x0:x1]

        shifts = np.arange(-max_shift, max_shift + 1)
        values = []

        for s in shifts:

            test = odd[:, x0 + s : x1 + s]

            values.append(_normalized_corr(ref, test))

        values = np.asarray(values)

        j = np.nanargmax(values)
        best = float(shifts[j])

        # Subpixel refinement with a parabola
        if 0 < j < len(shifts) - 1:
            ym = values[j - 1]
            y0 = values[j]
            yp = values[j + 1]

            denom = ym - 2 * y0 + yp

            if denom != 0:
                offset = 0.5 * (ym - yp) / denom
                best += offset

        centers.append((x0 + x1) / 2)
        best_shifts.append(best)
        best_corrs.append(values[j])

    return (
        np.asarray(centers),
        np.asarray(best_shifts),
        np.asarray(best_corrs),
    )


def structural_anisotropy(movie: np.ndarray[np.float32], bin_width=32, sigma=1.5):
    """
    Crude diagnostic of x-dependent geometric stretching.

    Returns an x/y structural scale proxy for vertical stripes.
    """
    assert movie.ndim == 3, "Input movie must be a 3D array (time, y, x)"

    img = np.nanmean(movie, axis=0)
    img = gaussian_filter(img, sigma=sigma)
    assert len(img.shape) == 2, "Averaged image must be 2D"

    gy, gx = np.gradient(img)

    X = img.shape[1]

    centers = []
    ratio = []

    for x0 in range(0, X - bin_width + 1, bin_width):
        x1 = x0 + bin_width

        px = np.nanmean(gx[:, x0:x1] ** 2)
        py = np.nanmean(gy[:, x0:x1] ** 2)

        centers.append((x0 + x1) / 2)

        ratio.append(np.sqrt(py / px) if px > 0 else np.nan)

    ratio = np.asarray(ratio)

    # Relative rather than absolute anisotropy is more useful here.
    ratio /= np.nanmedian(ratio)

    return np.asarray(centers), ratio


def noise_std_profile(z, parity):
    rows = np.arange(parity, z.shape[1], 2)

    return np.sqrt(np.nanmean(z[:, rows, :] ** 2, axis=(0, 1)))
