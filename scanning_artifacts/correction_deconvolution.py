import numpy as np


def make_regularized_inverse_filter(
    freq_fft,
    freq_H,
    H,
    regularization=0.15,
    max_gain=1.7,
    clip_H_at_one=True,
    cutoff=None,
    cutoff_width=5.0,
):
    """
    Regularized inverse filter for a measured transfer magnitude H(f).

    Parameters
    ----------
    freq_fft : ndarray
        Frequencies required by the FFT [cycles/rad].

    freq_H, H : ndarray
        Estimated acquisition transfer magnitude.

    regularization : float
        lambda in

            G = (1 + lambda) H / (H^2 + lambda)

        Larger -> more conservative.

    max_gain : float or None
        Hard upper bound on inverse gain.

    clip_H_at_one : bool
        For a conservative first test, do not interpret estimated
        H > 1 as genuine passband amplification.

    cutoff : float or None
        Optional high-frequency cutoff [cycles/rad].

    cutoff_width : float
        Smooth rolloff width before cutoff.

    Returns
    -------
    G : ndarray
        Regularized inverse filter.

    H_interp : ndarray
        Transfer magnitude interpolated to FFT frequencies.
    """

    freq_H = np.asarray(freq_H)
    H = np.asarray(H)

    good = np.isfinite(freq_H) & np.isfinite(H)

    freq_H = freq_H[good]
    H = H[good]

    order = np.argsort(freq_H)
    freq_H = freq_H[order]
    H = H[order]

    # Normalize DC
    H = H / H[0]

    H_interp = np.interp(
        freq_fft,
        freq_H,
        H,
        left=H[0],
        right=0.0,
    )

    H_interp = np.maximum(H_interp, 0.0)

    if clip_H_at_one:
        H_interp = np.minimum(H_interp, 1.0)

    lam = regularization

    G = (1.0 + lam) * H_interp / (H_interp**2 + lam)

    if max_gain is not None:
        G = np.minimum(G, max_gain)

    # Optional smooth high-frequency cutoff
    if cutoff is not None:

        start = cutoff - cutoff_width

        rolloff = np.ones_like(freq_fft)

        transition = (freq_fft > start) & (freq_fft < cutoff)

        u = (freq_fft[transition] - start) / cutoff_width

        rolloff[transition] = 0.5 * (1 + np.cos(np.pi * u))

        rolloff[freq_fft >= cutoff] = 0

        G *= rolloff

    return G, H_interp


def deconvolve_theta_movie(
    movie_theta,
    theta_uniform,
    freq_H,
    H,
    regularization=0.15,
    max_gain=1.7,
    cutoff=50.0,
    cutoff_width=5.0,
    pad_fraction=0.5,
    chunk_frames=32,
):
    """
    Regularized phase-domain deconvolution along the fast axis.

    movie_theta
        Movie already represented on a uniform resonant-phase grid.
        Shape (T, Y, Ntheta).

    Returns
    -------
    movie_out
        Deconvolved movie on the same theta grid.

    freq_fft, G, H_interp
        Diagnostics for the applied filter.
    """

    movie_theta = np.asarray(movie_theta)

    T, Y, N = movie_theta.shape

    dtheta = np.mean(np.diff(theta_uniform))

    pad = int(np.ceil(pad_fraction * N))

    N_pad = N + 2 * pad

    freq_fft = np.fft.rfftfreq(
        N_pad,
        d=dtheta,
    )

    G, H_interp = make_regularized_inverse_filter(
        freq_fft,
        freq_H,
        H,
        regularization=regularization,
        max_gain=max_gain,
        cutoff=cutoff,
        cutoff_width=cutoff_width,
    )

    out = np.empty(
        movie_theta.shape,
        dtype=np.float32,
    )

    for start in range(
        0,
        T,
        chunk_frames,
    ):

        stop = min(start + chunk_frames, T)

        block = np.asarray(
            movie_theta[start:stop],
            dtype=np.float32,
        )

        block_pad = np.pad(
            block,
            (
                (0, 0),
                (0, 0),
                (pad, pad),
            ),
            mode="reflect",
        )

        F = np.fft.rfft(
            block_pad,
            axis=-1,
        )

        F *= G[None, None, :]

        restored = np.fft.irfft(
            F,
            n=N_pad,
            axis=-1,
        )

        out[start:stop] = restored[..., pad : pad + N]

    return (
        out,
        freq_fft,
        G,
        H_interp,
    )
