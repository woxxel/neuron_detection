import numpy as np
from sklearn.decomposition import PCA


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


from .correction_correlation_surfaces import (
    correlation_surface,
    correlation_surface_metrics,
)
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


import numpy as np
from scipy.interpolate import interp1d


def split_half_theta_coherence(
    aligned_frames,
    x_positions,
    x0,
    A,
    seed=0,
    n_theta=256,
):
    """
    Split-half reproducibility spectrum in uniform resonant phase.

    aligned_frames : (N, Y, Xout)
        Already spatially aligned candidate frames.

    x_positions : (Xout,)
        Physical x coordinates represented by the reduced movie,
        e.g. resampler["x_out"].
    """

    rng = np.random.default_rng(seed)

    N = len(aligned_frames)
    order = rng.permutation(N)

    ia = order[: N // 2]
    ib = order[N // 2 :]

    image_a = np.mean(
        aligned_frames[ia],
        axis=0,
    )

    image_b = np.mean(
        aligned_frames[ib],
        axis=0,
    )

    # -----------------------------------
    # x -> theta
    # -----------------------------------

    theta_x = np.arcsin((x_positions - x0) / A)

    theta = np.linspace(
        theta_x[0],
        theta_x[-1],
        n_theta,
    )

    interp_a = interp1d(
        theta_x,
        image_a,
        axis=-1,
        kind="linear",
        assume_sorted=True,
    )

    interp_b = interp1d(
        theta_x,
        image_b,
        axis=-1,
        kind="linear",
        assume_sorted=True,
    )

    image_a = interp_a(theta)
    image_b = interp_b(theta)

    # -----------------------------------
    # Remove row DC
    # -----------------------------------

    image_a -= image_a.mean(
        axis=-1,
        keepdims=True,
    )

    image_b -= image_b.mean(
        axis=-1,
        keepdims=True,
    )

    # -----------------------------------
    # IMPORTANT: taper boundaries
    # -----------------------------------

    window = np.hanning(n_theta)

    image_a *= window[None, :]
    image_b *= window[None, :]

    # -----------------------------------
    # Spectrum
    # -----------------------------------

    FA = np.fft.rfft(
        image_a,
        axis=-1,
    )

    FB = np.fft.rfft(
        image_b,
        axis=-1,
    )

    cross = np.mean(
        FA * np.conj(FB),
        axis=0,
    )

    PA = np.mean(
        np.abs(FA) ** 2,
        axis=0,
    )

    PB = np.mean(
        np.abs(FB) ** 2,
        axis=0,
    )

    coherence = np.abs(cross) ** 2 / (PA * PB + 1e-20)

    dtheta = theta[1] - theta[0]

    freq = np.fft.rfftfreq(
        n_theta,
        d=dtheta,
    )

    return (
        freq,
        coherence,
        image_a,
        image_b,
        PA,
        PB,
    )


# def split_half_x_coherence(
#     aligned_frames,
#     dx=1.0,
#     seed=0,
# ):
#     """
#     Frequency-wise reproducibility of anatomy along x.

#     Returns magnitude-squared coherence between two independent
#     averages of the selected/aligned frames.
#     """

#     rng = np.random.default_rng(seed)

#     N = len(aligned_frames)

#     order = rng.permutation(N)

#     A_idx = order[: N // 2]
#     B_idx = order[N // 2 :]

#     A = np.mean(
#         aligned_frames[A_idx],
#         axis=0,
#     )

#     B = np.mean(
#         aligned_frames[B_idx],
#         axis=0,
#     )

#     # Remove row DC.
#     A = A - A.mean(
#         axis=-1,
#         keepdims=True,
#     )

#     B = B - B.mean(
#         axis=-1,
#         keepdims=True,
#     )

#     FA = np.fft.rfft(
#         A,
#         axis=-1,
#     )

#     FB = np.fft.rfft(
#         B,
#         axis=-1,
#     )

#     cross = np.mean(
#         FA * np.conj(FB),
#         axis=0,
#     )

#     P_A = np.mean(
#         np.abs(FA) ** 2,
#         axis=0,
#     )

#     P_B = np.mean(
#         np.abs(FB) ** 2,
#         axis=0,
#     )

#     coherence = np.abs(cross) ** 2 / (P_A * P_B + 1e-20)

#     freq = np.fft.rfftfreq(
#         A.shape[-1],
#         d=dx,
#     )

#     return (
#         freq,
#         coherence,
#         A,
#         B,
#     )
