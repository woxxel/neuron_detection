import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def build_alignment_template(movie, N_iter=2, k=200, plot=False, **kwargs):

    assert N_iter > 0, "N_iter must be positive"

    features, frame_ids = make_similarity_features(
        movie,  # e.g. your stripe-free 256 movie
        temporal_stride=kwargs.get("temporal_stride", 2),
    )

    scores, neighbors = recurrent_similarity_scores(
        features,
        frame_ids,
        k=k,
        exclude_frames=kwargs.get("exclude_frames", 45),
    )

    seed_feature_idx = np.argmax(scores)
    seed_frame = frame_ids[seed_feature_idx]

    candidate_frames = frame_ids[neighbors[seed_feature_idx]]

    print("seed frame:", seed_frame)
    print("similarity score:", scores[seed_feature_idx])

    template = movie[seed_frame]

    candidate_frames = np.concatenate(
        [
            [seed_frame],
            candidate_frames[:200],
        ]
    )

    if plot:
        fig, axes = plt.subplots(1, N_iter + 1, figsize=(12, 6))
        im = axes[0].imshow(
            template, origin="lower", aspect="auto", cmap="gray", clim=(0.1, 0.8)
        )
        plt.colorbar(im, ax=axes[0])

    for n in range(N_iter):
        template, aligned, good = refine_template(movie, candidate_frames, template)

        if plot:
            im = axes[n + 1].imshow(
                template, origin="lower", aspect="auto", cmap="gray", clim=(0.1, 0.8)
            )

    return template, aligned[good]


def refine_template(movie, candidate_frames, template_in):

    aligned, metrics = align_template_candidates(
        movie,
        candidate_frames,
        template_in,
    )

    peak = np.array([m["peak"] for m in metrics])

    margin = np.array([m["peak_margin"] for m in metrics])

    good = (peak >= np.percentile(peak, 25)) & (margin >= np.percentile(margin, 25))

    template_out = np.mean(
        aligned[good],
        axis=0,
    )
    return template_out, aligned, good


from scipy.ndimage import shift
from tqdm.auto import tqdm


def align_template_candidates(
    movie,
    candidate_frames,
    template,
    max_shift=5,
):
    aligned = []
    metrics = []

    for t in tqdm(candidate_frames):

        surface, shifts = correlation_surface(
            template,
            movie[t],
            max_shift=max_shift,
            lowpass_sigma=1.0,
            highpass_sigma=20.0,
        )

        m = correlation_surface_metrics(
            surface,
            shifts,
        )

        frame_aligned = shift(
            movie[t],
            shift=(
                m["dy"],
                m["dx"],
            ),
            order=1,
            mode="nearest",
        )

        aligned.append(frame_aligned)

        metrics.append(m)

    return (
        np.asarray(aligned),
        metrics,
    )


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


from scipy.ndimage import gaussian_filter


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


def make_similarity_features(movie, temporal_stride=1, by=8, bx=4, n_pca=64):
    frame_ids = np.arange(0, len(movie), temporal_stride)

    small = block_mean_movie(
        np.asarray(movie[frame_ids], dtype=np.float32), by=by, bx=bx
    )

    X = small.reshape(len(small), -1)

    # Remove brightness differences.
    X -= X.mean(axis=1, keepdims=True)

    X /= X.std(axis=1, keepdims=True) + 1e-8

    # PCA greatly accelerates the all-to-all comparison.
    n_pca = min(n_pca, len(X) - 1, X.shape[1])

    X = PCA(
        n_components=n_pca,
        svd_solver="randomized",
    ).fit_transform(X)

    # Normalize so dot product = cosine similarity.
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12

    return X.astype(np.float32), frame_ids


def recurrent_similarity_scores(
    features,
    frame_ids,
    k=100,
    exclude_frames=30,
    block_size=256,
):
    """
    For each frame, find its k most similar frames elsewhere
    in the movie.

    Nearby times are explicitly excluded.
    """

    N = len(features)

    scores = np.empty(N, dtype=np.float32)

    neighbors = np.empty((N, k), dtype=np.int32)

    for start in range(0, N, block_size):

        stop = min(start + block_size, N)

        C = features[start:stop] @ features.T

        for local_i, global_i in enumerate(range(start, stop)):

            dt = np.abs(frame_ids - frame_ids[global_i])

            C[local_i, dt <= exclude_frames] = -np.inf

        idx = np.argpartition(C, kth=N - k, axis=1)[:, -k:]
        vals = np.take_along_axis(C, idx, axis=1)

        # sort strongest first
        order = np.argsort(vals, axis=1)[:, ::-1]

        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)

        neighbors[start:stop] = idx

        # Median/mean of top matches says:
        # "does this image state recur often?"
        scores[start:stop] = np.median(vals, axis=1)

    return scores, neighbors


def block_mean_movie(movie, by=8, bx=4):
    """
    Spatial reduction used ONLY for finding similar frames.

    For a (T, 512, 256) movie and by=8, bx=4:
        -> (T, 64, 64)
    """

    T, Y, X = movie.shape

    Y2 = (Y // by) * by
    X2 = (X // bx) * bx

    x = movie[:, :Y2, :X2]

    x = x.reshape(T, Y2 // by, by, X2 // bx, bx)

    return x.mean(axis=(2, 4))
