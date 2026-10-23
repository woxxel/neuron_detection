import numpy as np


def raised_cosine_lowpass(
    freq,
    f_pass,
    f_stop,
):
    """
    Unit gain below f_pass, smooth cosine rolloff,
    zero at and above f_stop.
    """
    freq = np.asarray(freq)

    H = np.ones_like(
        freq,
        dtype=float,
    )

    transition = (freq > f_pass) & (freq < f_stop)

    u = (freq[transition] - f_pass) / (f_stop - f_pass)

    H[transition] = 0.5 * (1 + np.cos(np.pi * u))

    H[freq >= f_stop] = 0.0

    return H


def make_target_equalizer(
    freq,
    freq_H,
    H_old,
    H_target,
    regularization=0.15,
    max_gain=1.7,
):
    """
    Regularized filter transforming H_old toward H_target.

    E(f) =
        (1 + lambda)
        H_target(f) H_old(f)
        ---------------------
          H_old(f)^2 + lambda

    Assumes zero-phase / real nonnegative H_old.
    """

    freq_H = np.asarray(freq_H)
    H_old = np.asarray(H_old)

    good = np.isfinite(freq_H) & np.isfinite(H_old)

    freq_H = freq_H[good]
    H_old = H_old[good]

    order = np.argsort(freq_H)

    freq_H = freq_H[order]
    H_old = H_old[order]

    # normalize DC
    H_old = H_old / H_old[0]

    H = np.interp(
        freq,
        freq_H,
        H_old,
        left=H_old[0],
        right=0.0,
    )

    H = np.clip(
        H,
        0,
        1,
    )

    lam = regularization

    E = (1 + lam) * H_target * H / (H**2 + lam)

    if max_gain is not None:
        E = np.minimum(
            E,
            max_gain,
        )

    H_effective = H * E

    return E, H, H_effective


def trapezoid_weights(x):
    x = np.asarray(x)

    w = np.empty_like(x)

    w[1:-1] = 0.5 * (x[2:] - x[:-2])

    w[0] = 0.5 * (x[1] - x[0])

    w[-1] = 0.5 * (x[-1] - x[-2])

    return w


def make_equalized_resonant_resampler(
    X,
    x0,
    A,
    n_out,
    freq_H,
    H_old,
    regularization=0.15,
    max_gain=1.7,
    pass_fraction=0.75,
    stop_fraction=0.95,
    kernel_oversample=4,
):
    """
    Construct a direct operator

        original uniform x
            ->
        phase-domain equalization
            ->
        reduced uniform x

    without creating intermediate resampled images.

    Parameters
    ----------
    X :
        Original number of x pixels, e.g. 512.

    x0, A :
        Resonant geometry.

    n_out :
        Final number of uniform spatial samples, e.g. 256.

    freq_H, H_old :
        Previously estimated phase-domain acquisition
        transfer magnitude.

    regularization, max_gain :
        Strength of inverse/equalization.

    pass_fraction, stop_fraction :
        Define target low-pass relative to the globally
        safe phase Nyquist of the output grid.

    Returns
    -------
    dict containing W and filter diagnostics.
    """

    # ------------------------------------------------
    # Input coordinates
    # ------------------------------------------------

    x_in = np.arange(
        X,
        dtype=float,
    )

    theta_in = np.arcsin((x_in - x0) / A)

    # ------------------------------------------------
    # Desired reduced, uniform physical x
    # ------------------------------------------------

    x_out = np.linspace(
        0,
        X - 1,
        n_out,
    )

    theta_out = np.arcsin((x_out - x0) / A)

    # ------------------------------------------------
    # Conservative phase Nyquist of final x grid
    #
    # theta spacing is largest near scan edges.
    # ------------------------------------------------

    dtheta_out = np.diff(theta_out)

    f_nyquist_safe = 1.0 / (2.0 * np.max(dtheta_out))

    f_pass = pass_fraction * f_nyquist_safe

    f_stop = stop_fraction * f_nyquist_safe

    # ------------------------------------------------
    # Make a fine theta grid ONLY to construct the
    # impulse response of our filter.
    #
    # The movie itself is never interpolated onto it.
    # ------------------------------------------------

    min_dtheta = np.min(np.diff(theta_in))

    dtheta_kernel = min_dtheta / kernel_oversample

    theta_span = theta_in[-1] - theta_in[0]

    # generous extent to avoid FFT wraparound
    required_length = 4.0 * theta_span

    N_kernel = int(np.ceil(required_length / dtheta_kernel))

    # next power of two
    N_kernel = 1 << int(np.ceil(np.log2(N_kernel)))

    freq = np.fft.rfftfreq(
        N_kernel,
        d=dtheta_kernel,
    )

    # ------------------------------------------------
    # Desired transfer function
    # ------------------------------------------------

    H_target = raised_cosine_lowpass(
        freq,
        f_pass=f_pass,
        f_stop=f_stop,
    )

    E, H_interp, H_effective = make_target_equalizer(
        freq,
        freq_H,
        H_old,
        H_target,
        regularization=regularization,
        max_gain=max_gain,
    )

    # ------------------------------------------------
    # Convert frequency-domain equalizer to its
    # zero-phase impulse response
    # ------------------------------------------------

    kernel = np.fft.irfft(
        E,
        n=N_kernel,
    )

    kernel = np.fft.fftshift(kernel)

    lag = (np.arange(N_kernel) - N_kernel // 2) * dtheta_kernel

    # ------------------------------------------------
    # Evaluate the impulse response directly between
    # every desired output theta and every actual
    # input theta.
    # ------------------------------------------------

    delta_theta = theta_out[:, None] - theta_in[None, :]

    K = np.interp(
        delta_theta.ravel(),
        lag,
        kernel,
        left=0.0,
        right=0.0,
    ).reshape(delta_theta.shape)

    # Correct integration weights because input samples
    # are nonuniform in theta.
    q = trapezoid_weights(theta_in)

    W = K * q[None, :]

    # Preserve constant intensity exactly.
    row_sum = np.sum(
        W,
        axis=1,
        keepdims=True,
    )

    W /= row_sum + 1e-15

    return {
        "W": W,
        "x_in": x_in,
        "theta_in": theta_in,
        "x_out": x_out,
        "theta_out": theta_out,
        "freq": freq,
        "H_old": H_interp,
        "H_target": H_target,
        "equalizer": E,
        "H_effective": H_effective,
        "f_nyquist_safe": f_nyquist_safe,
        "f_pass": f_pass,
        "f_stop": f_stop,
        "kernel": kernel,
        "lag": lag,
    }


def apply_resampling_operator(
    movie,
    resampler,
    chunk_frames=64,
    dtype=np.float32,
):
    W = resampler["W"]

    T, Y, X = movie.shape

    if X != W.shape[1]:
        raise ValueError(f"movie has X={X}, operator expects {W.shape[1]}")

    n_out = W.shape[0]

    out = np.empty(
        (T, Y, n_out),
        dtype=dtype,
    )

    for start in range(
        0,
        T,
        chunk_frames,
    ):

        stop = min(
            T,
            start + chunk_frames,
        )

        block = np.asarray(
            movie[start:stop],
            dtype=np.float32,
        )

        out[start:stop] = block @ W.T

    return out
