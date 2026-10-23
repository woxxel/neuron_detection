import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def _norm_corr(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    good = np.isfinite(a) & np.isfinite(b)
    a = a[good]
    b = b[good]

    if len(a) < 10:
        return np.nan

    a -= a.mean()
    b -= b.mean()

    den = np.sqrt(np.sum(a * a) * np.sum(b * b))

    if den == 0:
        return np.nan

    return np.sum(a * b) / den


def estimate_bidi_shift_frame(
    image,
    max_shift=5,
    x_margin=20,
    use_gradient=True,
):
    """
    Estimate odd/even relative x displacement in one frame.

    Assumes saved odd/even rows are already in common image orientation.
    """

    image = np.asarray(image, dtype=np.float32)

    if use_gradient:
        image = np.gradient(gaussian_filter(image, sigma=(0, 1)), axis=1)

    Y, X = image.shape

    odd_rows = np.arange(1, Y - 1, 2)

    odd = image[odd_rows]

    reference = (image[odd_rows - 1] + image[odd_rows + 1]) / 2

    shifts = np.arange(-max_shift, max_shift + 1)

    corr = []

    for s in shifts:

        if s < 0:
            ref = reference[:, x_margin - s : X - x_margin]

            test = odd[:, x_margin : X - x_margin + s]

        elif s > 0:
            ref = reference[:, x_margin : X - x_margin - s]

            test = odd[:, x_margin + s : X - x_margin]

        else:
            ref = reference[:, x_margin : X - x_margin]

            test = odd[:, x_margin : X - x_margin]

        corr.append(_norm_corr(ref, test))

    corr = np.asarray(corr)

    j = np.nanargmax(corr)
    shift = float(shifts[j])

    # parabolic subpixel interpolation
    if 0 < j < len(shifts) - 1:

        ym = corr[j - 1]
        y0 = corr[j]
        yp = corr[j + 1]

        denom = ym - 2 * y0 + yp

        if denom != 0:
            shift += 0.5 * (ym - yp) / denom

    # How sharp is the correlation maximum?
    quality = corr[j]

    return shift, quality


def bidi_shift_timeseries(
    movie,
    max_shift=5,
    x_margin=20,
):
    T = movie.shape[0]

    shifts = np.full(T, np.nan)
    quality = np.full(T, np.nan)

    for t in range(T):

        shifts[t], quality[t] = estimate_bidi_shift_frame(
            movie[t],
            max_shift=max_shift,
            x_margin=x_margin,
        )

    return shifts, quality


def bidi_shift_x_time(
    movie,
    window=96,
    step=32,
    max_shift=5,
    x_margin=8,
):
    """
    Time-resolved local odd/even alignment.

    Returns
    -------
    x_centers : (Nx,)
    shifts : (T, Nx)
    quality : (T, Nx)
    """

    T, Y, X = movie.shape

    starts = np.arange(x_margin, X - window - x_margin + 1, step)

    centers = starts + window / 2

    shifts_out = np.full((T, len(starts)), np.nan)

    quality_out = np.full_like(shifts_out, np.nan)

    for t in range(T):

        image = np.gradient(gaussian_filter(movie[t], sigma=(0, 1)), axis=1)

        odd_rows = np.arange(1, Y - 1, 2)

        odd = image[odd_rows]

        reference = (image[odd_rows - 1] + image[odd_rows + 1]) / 2

        for k, x0 in enumerate(starts):

            x1 = x0 + window

            values = []
            candidate_shifts = np.arange(-max_shift, max_shift + 1)

            for s in candidate_shifts:

                ref = reference[:, x0:x1]

                test = odd[:, x0 + s : x1 + s]

                values.append(_norm_corr(ref, test))

            values = np.asarray(values)

            j = np.nanargmax(values)

            dx = float(candidate_shifts[j])

            if 0 < j < len(values) - 1:

                ym = values[j - 1]
                y0 = values[j]
                yp = values[j + 1]

                denom = ym - 2 * y0 + yp

                if denom != 0:
                    dx += 0.5 * (ym - yp) / denom

            shifts_out[t, k] = dx
            quality_out[t, k] = values[j]

    return centers, shifts_out, quality_out


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def bidi_shift_y_time(
    movie,
    pairs_per_block=8,
    step_pairs=4,
    max_shift=5,
    x_margin=32,
    smooth_x=1.0,
):
    """
    Estimate bidirectional odd/even phase displacement as function of
    frame and y-position.

    Parameters
    ----------
    movie : ndarray, shape (T, Y, X)

    pairs_per_block : int
        Number of odd rows (and their neighboring even rows) pooled
        into each local y estimate.

        8 means a block spans approximately 16 image rows.

    step_pairs : int
        Step between blocks in odd-row units.
        4 gives ~50% overlap for pairs_per_block=8.

    max_shift : int
        Maximum tested odd/even displacement in pixels.

    x_margin : int
        Pixels excluded at left/right image edge.

    smooth_x : float
        Gaussian smoothing along x before taking x gradient.

    Returns
    -------
    y_centers : ndarray, shape (Ny,)
        Center y-coordinate for every block.

    shifts : ndarray, shape (T, Ny)
        Estimated odd/even displacement.

    quality : ndarray, shape (T, Ny)
        Maximum normalized correlation.

    peak_margin : ndarray, shape (T, Ny)
        Difference between best and second-best integer-shift
        correlation. Useful confidence measure.
    """

    movie = np.asarray(movie, dtype=np.float32)

    T, Y, X = movie.shape

    # Odd rows that have both neighboring even rows available
    odd_rows = np.arange(1, Y - 1, 2)

    starts = np.arange(0, len(odd_rows) - pairs_per_block + 1, step_pairs)

    y_centers = np.array([np.mean(odd_rows[s : s + pairs_per_block]) for s in starts])

    shifts_out = np.full((T, len(starts)), np.nan, dtype=float)

    quality_out = np.full_like(shifts_out, np.nan)

    peak_margin_out = np.full_like(shifts_out, np.nan)

    candidate_shifts = np.arange(-max_shift, max_shift + 1)

    for t in range(T):

        # Only smooth along fast scanning axis.
        image = gaussian_filter(movie[t], sigma=(0, smooth_x))

        # Horizontal gradients are more useful for x registration
        image = np.gradient(image, axis=1)

        for k, start in enumerate(starts):

            rows = odd_rows[start : start + pairs_per_block]

            odd = image[rows]

            # Structural estimate at odd-row locations from
            # neighboring even rows.
            reference = (image[rows - 1] + image[rows + 1]) / 2

            corr_values = []

            for s in candidate_shifts:

                if s < 0:
                    ref = reference[:, x_margin - s : X - x_margin]

                    test = odd[:, x_margin : X - x_margin + s]

                elif s > 0:
                    ref = reference[:, x_margin : X - x_margin - s]

                    test = odd[:, x_margin + s : X - x_margin]

                else:
                    ref = reference[:, x_margin : X - x_margin]

                    test = odd[:, x_margin : X - x_margin]

                corr_values.append(_norm_corr(ref, test))

            corr_values = np.asarray(corr_values)

            if not np.any(np.isfinite(corr_values)):
                continue

            j = np.nanargmax(corr_values)

            dx = float(candidate_shifts[j])

            # ---------------------------------------
            # Subpixel parabolic interpolation
            # ---------------------------------------
            if 0 < j < len(candidate_shifts) - 1:

                ym = corr_values[j - 1]
                y0 = corr_values[j]
                yp = corr_values[j + 1]

                denom = ym - 2 * y0 + yp

                if np.isfinite(denom) and denom != 0:
                    dx += 0.5 * (ym - yp) / denom

            shifts_out[t, k] = dx
            quality_out[t, k] = corr_values[j]

            # Difference from second-best integer solution.
            finite_corr = corr_values[np.isfinite(corr_values)]

            if len(finite_corr) >= 2:
                best_two = np.partition(finite_corr, -2)[-2:]

                peak_margin_out[t, k] = best_two[-1] - best_two[-2]

    return (
        y_centers,
        shifts_out,
        quality_out,
        peak_margin_out,
    )
