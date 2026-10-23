import numpy as np


def make_resonant_phase_grids(
    X,
    x0,
    A,
    n_theta=None,
):
    """
    Construct mappings between the saved linear x-coordinate and
    uniform resonant phase theta.

    Returns
    -------
    x : (X,)
        Original x coordinates.

    theta_x : (X,)
        Resonant phase corresponding to every original x pixel.

    theta_uniform : (Ntheta,)
        Uniform phase grid.

    x_of_theta : (Ntheta,)
        Physical x coordinate corresponding to theta_uniform.
    """

    if n_theta is None:
        n_theta = X

    x = np.arange(X, dtype=float)

    u = (x - x0) / A

    if np.any(np.abs(u) >= 1):
        raise ValueError("x0/A imply resonant turning points inside the image.")

    theta_x = np.arcsin(u)

    theta_uniform = np.linspace(
        theta_x[0],
        theta_x[-1],
        n_theta,
    )

    x_of_theta = x0 + A * np.sin(theta_uniform)

    return {
        "x": x,
        "theta_x": theta_x,
        "theta_uniform": theta_uniform,
        "x_of_theta": x_of_theta,
    }


def movie_x_to_theta(movie, grids):
    """
    movie : (T, Y, X)

    Returns movie on a uniform resonant-phase grid:
        (T, Y, Ntheta)
    """

    movie = np.asarray(movie)

    T, Y, X = movie.shape

    x = grids["x"]
    x_of_theta = grids["x_of_theta"]

    out = np.empty(
        (T, Y, len(x_of_theta)),
        dtype=np.float32,
    )

    for t in range(T):
        for y in range(Y):

            out[t, y] = np.interp(
                x_of_theta,
                x,
                movie[t, y],
            )

    return out


def movie_theta_to_x(movie_theta, grids):
    """
    Uniform theta -> original uniform physical-x grid.
    """

    movie_theta = np.asarray(movie_theta)

    T, Y, N = movie_theta.shape

    theta_uniform = grids["theta_uniform"]
    theta_x = grids["theta_x"]

    out = np.empty(
        (T, Y, len(theta_x)),
        dtype=np.float32,
    )

    for t in range(T):
        for y in range(Y):

            out[t, y] = np.interp(
                theta_x,
                theta_uniform,
                movie_theta[t, y],
            )

    return out


def make_partial_whitening_filter(
    freq_fft,
    freq_psd,
    psd,
    beta=0.5,
    regularization=0.1,
    max_gain=2.0,
    cutoff=None,
):
    """
    Construct a regularized partial whitening filter.

    beta:
        0   -> no whitening
        1   -> full 1/sqrt(S) whitening
        ~0.3-0.6 is a conservative starting point.
    """

    S = np.interp(
        freq_fft,
        freq_psd,
        psd,
        left=psd[0],
        right=psd[-1],
    )

    S = np.maximum(S, 0)

    S /= np.nanmax(S)

    lam = regularization

    W = ((S + lam) / (1 + lam)) ** (-beta / 2)

    W = np.minimum(W, max_gain)

    if cutoff is not None:
        W = np.where(
            freq_fft <= cutoff,
            W,
            0.0,
        )

    return W


def whiten_theta_movie(
    movie_theta,
    theta_uniform,
    freq_psd,
    psd,
    beta=0.4,
    regularization=0.15,
    max_gain=1.7,
    cutoff=45,
):
    """
    Filter each scan line along the uniform theta axis.
    """

    movie_theta = np.asarray(movie_theta, dtype=np.float32)

    dtheta = np.mean(np.diff(theta_uniform))

    N = movie_theta.shape[-1]

    freq_fft = np.fft.rfftfreq(N, d=dtheta)

    W = make_partial_whitening_filter(
        freq_fft,
        freq_psd,
        psd,
        beta=beta,
        regularization=regularization,
        max_gain=max_gain,
        cutoff=cutoff,
    )

    F = np.fft.rfft(movie_theta, axis=-1)

    F *= W[None, None, :]

    out = np.fft.irfft(F, n=N, axis=-1)

    return out.astype(np.float32), W, freq_fft
