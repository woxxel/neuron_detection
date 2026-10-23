import numpy as np
from scipy.signal import firwin
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d

import numpy as np
from scipy.signal import firwin

import numpy as np


def trapezoid_weights(x):
    """
    Quadrature weights for samples at nonuniform positions x.
    """
    x = np.asarray(x)

    w = np.empty_like(x)

    w[1:-1] = 0.5 * (x[2:] - x[:-2])

    w[0] = 0.5 * (x[1] - x[0])

    w[-1] = 0.5 * (x[-1] - x[-2])

    return w


def make_direct_theta_resampler(
    X,
    x0,
    A,
    n_out,
    antialias_fraction=0.90,
    phase_bandlimit=None,
    num_lobes=6,
    kaiser_beta=8.0,
):
    """
    Direct resonant-phase low-pass + downsampling operator.

    Maps:
        X uniform-x input samples
            ->
        n_out uniform-x output samples

    without constructing any intermediate resampled image.

    Filtering is defined in resonant phase theta.
    """

    # -----------------------------------
    # Input coordinates
    # -----------------------------------

    x_in = np.arange(
        X,
        dtype=float,
    )

    theta_in = np.arcsin((x_in - x0) / A)

    # -----------------------------------
    # Desired output: uniform physical x
    # -----------------------------------

    x_out = np.linspace(
        0,
        X - 1,
        n_out,
    )

    theta_out = np.arcsin((x_out - x0) / A)

    # -----------------------------------
    # Determine safe phase bandwidth
    # -----------------------------------

    dtheta_in = np.diff(theta_in)
    dtheta_out = np.diff(theta_out)

    input_nyquist = 1.0 / (2.0 * np.max(dtheta_in))

    output_nyquist = 1.0 / (2.0 * np.max(dtheta_out))

    cutoff = antialias_fraction * min(
        input_nyquist,
        output_nyquist,
    )

    if phase_bandlimit is not None:
        cutoff = min(
            cutoff,
            phase_bandlimit,
        )

    # -----------------------------------
    # Quadrature weights because theta_in
    # is nonuniformly sampled
    # -----------------------------------

    q = trapezoid_weights(theta_in)

    # theta difference between every
    # output and input sample
    dtheta = theta_out[:, None] - theta_in[None, :]

    # -----------------------------------
    # Windowed ideal low-pass kernel
    #
    # h(theta) =
    #   2 fc sinc(2 fc theta)
    # -----------------------------------

    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * dtheta)

    # Keep several sinc lobes
    support = num_lobes / (2.0 * cutoff)

    u = dtheta / support

    inside = np.abs(u) <= 1

    window = np.zeros_like(u)

    window[inside] = np.i0(kaiser_beta * np.sqrt(1.0 - u[inside] ** 2)) / np.i0(
        kaiser_beta
    )

    # Integration on nonuniform theta samples
    W = h * window * q[None, :]

    # Preserve constant intensity
    W /= (
        np.sum(
            W,
            axis=1,
            keepdims=True,
        )
        + 1e-15
    )

    return {
        "W": W,
        "x_in": x_in,
        "theta_in": theta_in,
        "x_out": x_out,
        "theta_out": theta_out,
        "cutoff": cutoff,
        "input_nyquist": input_nyquist,
        "output_nyquist": output_nyquist,
    }


def direct_theta_downsample_movie(
    movie,
    resampler,
    chunk_frames=64,
    dtype=np.float32,
):
    """
    movie:
        (T, Y, X)

    output:
        (T, Y, n_out)
    """

    W = resampler["W"]

    T, Y, X = movie.shape

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

        # (T,Y,X) @ (X,N)
        # -> (T,Y,N)
        out[start:stop] = block @ W.T

    return out


def make_resonant_downsample_plan_v2(
    X,
    x0,
    A,
    n_out,
    n_dense=None,
    antialias_fraction=0.90,
    phase_bandlimit=None,
    numtaps=63,
):
    """
    Resonant-aware downsampling plan without an intermediate
    low-resolution uniform-theta grid.

    Pipeline:
        x -> dense uniform theta
          -> theta low-pass
          -> directly sample at theta(x_out)
    """

    if n_dense is None:
        n_dense = X

    if numtaps % 2 == 0:
        numtaps += 1

    # Original uniform-x positions
    x_in = np.arange(
        X,
        dtype=float,
    )

    theta_x_in = np.arcsin((x_in - x0) / A)

    theta_min = theta_x_in[0]
    theta_max = theta_x_in[-1]

    # Dense, uniform theta representation
    theta_dense = np.linspace(
        theta_min,
        theta_max,
        n_dense,
    )

    x_theta_dense = x0 + A * np.sin(theta_dense)

    dtheta_dense = np.mean(np.diff(theta_dense))

    fs_dense = 1 / dtheta_dense

    # Final uniform physical-x grid
    x_out = np.linspace(
        0,
        X - 1,
        n_out,
    )

    theta_x_out = np.arcsin((x_out - x0) / A)

    # ---------------------------------------
    # Important:
    #
    # output points are nonuniform in theta.
    #
    # For a globally safe anti-alias filter,
    # use the largest theta spacing.
    # ---------------------------------------

    dtheta_out = np.diff(theta_x_out)

    max_dtheta_out = np.max(dtheta_out)

    min_nyquist_theta = 1.0 / (2 * max_dtheta_out)

    cutoff = antialias_fraction * min_nyquist_theta

    if phase_bandlimit is not None:
        cutoff = min(
            cutoff,
            phase_bandlimit,
        )

    cutoff = min(
        cutoff,
        0.499 * fs_dense,
    )

    fir = firwin(
        numtaps,
        cutoff=cutoff,
        fs=fs_dense,
        window=("kaiser", 8.0),
    )

    return {
        "X": X,
        "x0": x0,
        "A": A,
        "x_in": x_in,
        "theta_dense": theta_dense,
        "x_theta_dense": x_theta_dense,
        "x_out": x_out,
        "theta_x_out": theta_x_out,
        "fir": fir,
        "cutoff": cutoff,
        "min_nyquist_theta": min_nyquist_theta,
        "n_dense": n_dense,
        "n_out": n_out,
    }


from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d


def resonant_downsample_movie_v2(
    movie,
    plan,
    chunk_frames=32,
    dtype=np.float32,
):
    """
    x -> dense theta -> low-pass -> reduced uniform x

    No intermediate sparse theta representation.
    """

    T, Y, X = movie.shape

    out = np.empty(
        (
            T,
            Y,
            plan["n_out"],
        ),
        dtype=dtype,
    )

    for start in range(
        0,
        T,
        chunk_frames,
    ):

        stop = min(start + chunk_frames, T)

        block = np.asarray(
            movie[start:stop],
            dtype=np.float32,
        )

        # --------------------------------
        # 1. uniform x -> uniform theta
        # --------------------------------

        interp_x = interp1d(
            plan["x_in"],
            block,
            axis=-1,
            kind="linear",
            assume_sorted=True,
        )

        theta_dense = interp_x(plan["x_theta_dense"])

        # --------------------------------
        # 2. pool / low-pass in theta
        # --------------------------------

        theta_filtered = convolve1d(
            theta_dense,
            weights=plan["fir"],
            axis=-1,
            mode="reflect",
        )

        # --------------------------------
        # 3. DIRECTLY evaluate at theta
        #    corresponding to uniform
        #    reduced x positions
        # --------------------------------

        interp_theta = interp1d(
            plan["theta_dense"],
            theta_filtered,
            axis=-1,
            kind="linear",
            assume_sorted=True,
        )

        out[start:stop] = interp_theta(plan["theta_x_out"])

    return out


def temporal_column_stats(movie):

    # temporal + y averages
    mean_x = np.mean(
        movie,
        axis=(0, 1),
    )

    std_x = np.std(
        movie,
        axis=(0, 1),
    )

    diff = np.diff(
        movie,
        axis=0,
    )

    noise_std_x = np.std(
        diff,
        axis=(0, 1),
    ) / np.sqrt(2)

    return (
        mean_x,
        std_x,
        noise_std_x,
    )


def make_resonant_downsample_plan(
    X,
    x0,
    A,
    n_out,
    n_theta=None,
    n_dense=None,
    antialias_fraction=0.90,
    phase_bandlimit=None,
    numtaps=63,
):
    """
    Prepare coordinates and anti-alias filter for resonant-aware
    fast-axis downsampling.

    Parameters
    ----------
    X : int
        Original number of x pixels.

    x0, A : float
        Estimated resonant geometry:
            x = x0 + A sin(theta)

    n_out : int
        Final number of uniformly spaced physical-x pixels.

    n_theta : int or None
        Number of samples retained on uniform theta grid before
        mapping back to x. Defaults to n_out.

    n_dense : int or None
        Dense uniform-theta representation before low-pass filtering.
        Defaults to X.

    antialias_fraction : float
        Fraction of target theta Nyquist frequency used as FIR cutoff.

    phase_bandlimit : float or None
        Optional additional cutoff in cycles/radian.
        E.g. 40-45 based on our inferred bandwidth.
        None means only apply the anti-alias requirement.

    numtaps : int
        Length of symmetric FIR filter. Should be odd.

    Returns
    -------
    dict
    """

    if n_theta is None:
        n_theta = n_out

    if n_dense is None:
        n_dense = X

    if numtaps % 2 == 0:
        numtaps += 1

    # ------------------------------------------
    # Original linear-x coordinate
    # ------------------------------------------

    x_in = np.arange(X, dtype=float)

    u = (x_in - x0) / A

    if np.any(np.abs(u) >= 1):
        raise ValueError(
            "The inferred resonant turning points lie " "inside the recorded FOV."
        )

    theta_x_in = np.arcsin(u)

    theta_min = theta_x_in[0]
    theta_max = theta_x_in[-1]
    theta_span = theta_max - theta_min

    # ------------------------------------------
    # Dense uniform phase coordinate
    # ------------------------------------------

    theta_dense = np.linspace(theta_min, theta_max, n_dense)

    x_of_theta_dense = x0 + A * np.sin(theta_dense)

    dtheta_dense = np.mean(np.diff(theta_dense))

    fs_dense = 1.0 / dtheta_dense

    # ------------------------------------------
    # Reduced uniform phase coordinate
    # ------------------------------------------

    theta_down = np.linspace(theta_min, theta_max, n_theta)

    dtheta_target = theta_span / (n_theta - 1)

    fs_target = 1.0 / dtheta_target

    target_nyquist = 0.5 * fs_target

    cutoff = antialias_fraction * target_nyquist

    if phase_bandlimit is not None:
        cutoff = min(cutoff, phase_bandlimit)

    # Cannot exceed dense-grid Nyquist
    cutoff = min(cutoff, 0.499 * fs_dense)

    fir = firwin(
        numtaps,
        cutoff=cutoff,
        fs=fs_dense,
        window=("kaiser", 8.0),
    )

    # ------------------------------------------
    # Final uniform physical-x grid
    # ------------------------------------------

    x_out = np.linspace(0, X - 1, n_out)

    theta_x_out = np.arcsin((x_out - x0) / A)

    return {
        "X": X,
        "x0": x0,
        "A": A,
        "n_dense": n_dense,
        "n_theta": n_theta,
        "n_out": n_out,
        "x_in": x_in,
        "theta_x_in": theta_x_in,
        "theta_dense": theta_dense,
        "x_of_theta_dense": x_of_theta_dense,
        "theta_down": theta_down,
        "x_out": x_out,
        "theta_x_out": theta_x_out,
        "fir": fir,
        "cutoff": cutoff,
        "target_nyquist": target_nyquist,
        "fs_dense": fs_dense,
        "fs_target": fs_target,
    }


def resonant_downsample_movie(
    movie,
    plan,
    chunk_frames=32,
    dtype=np.float32,
):
    """
    Resonant-aware fast-axis downsampling.

    Input
    -----
    movie : (T, Y, X)

    Output
    ------
    movie_ds : (T, Y, n_out)

    Pipeline
    --------
    linear x
        -> uniform theta
        -> anti-alias filtering in theta
        -> reduced theta sampling
        -> uniform reduced physical x
    """

    movie = np.asarray(movie)

    T, Y, X = movie.shape

    if X != plan["X"]:
        raise ValueError(f"Movie X={X}, but plan expects X={plan['X']}")

    out = np.empty(
        (
            T,
            Y,
            plan["n_out"],
        ),
        dtype=dtype,
    )

    for start in range(
        0,
        T,
        chunk_frames,
    ):

        stop = min(start + chunk_frames, T)

        block = np.asarray(
            movie[start:stop],
            dtype=np.float32,
        )

        # --------------------------------------
        # 1. linear x -> uniform theta
        # --------------------------------------

        interp_x = interp1d(
            plan["x_in"],
            block,
            axis=-1,
            kind="linear",
            assume_sorted=True,
            bounds_error=False,
        )

        block_theta_dense = interp_x(plan["x_of_theta_dense"])

        # --------------------------------------
        # 2. Anti-alias / pooling filter
        #
        # Unit-sum symmetric FIR means DC/intensity
        # is preserved while neighboring redundant
        # samples are combined.
        # --------------------------------------

        block_theta_filt = convolve1d(
            block_theta_dense,
            weights=plan["fir"],
            axis=-1,
            mode="reflect",
        )

        # --------------------------------------
        # 3. Dense theta -> reduced theta
        # --------------------------------------

        interp_theta = interp1d(
            plan["theta_dense"],
            block_theta_filt,
            axis=-1,
            kind="linear",
            assume_sorted=True,
            bounds_error=False,
        )

        block_theta_down = interp_theta(plan["theta_down"])

        # --------------------------------------
        # 4. Reduced theta -> reduced linear x
        # --------------------------------------

        interp_back = interp1d(
            plan["theta_down"],
            block_theta_down,
            axis=-1,
            kind="linear",
            assume_sorted=True,
            bounds_error=False,
        )

        block_x_down = interp_back(plan["theta_x_out"])

        out[start:stop] = block_x_down.astype(
            dtype,
            copy=False,
        )

    return out
