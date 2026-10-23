import numpy as np
import cv2

from scipy.ndimage import gaussian_filter


def preprocess_registration_image(
    image,
    lowpass_sigma=1.0,
    highpass_sigma=20.0,
):
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


def template_correlation_surfaces(
    movie,
    template,
    max_shift=8,
    lowpass_sigma=1.0,
    highpass_sigma=20.0,
):
    """
    Normalized correlation surface of every frame against a fixed template.

    Parameters
    ----------
    movie : (T, Y, X)

    template : (Y, X)

    max_shift : int
        Search range in pixels.

    Returns
    -------
    surfaces : (T, 2*m+1, 2*m+1)

    shifts :
        Shift values corresponding to each axis.

    Sign convention
    ---------------
    dx/dy are shifts to APPLY TO THE FRAME
    to align it to the template.
    """

    T, Y, X = movie.shape
    m = max_shift

    template_p = preprocess_registration_image(
        template,
        lowpass_sigma,
        highpass_sigma,
    )

    # Cropping this allows matchTemplate to explore +/- m
    # while always comparing exactly the same template region.
    template_crop = template_p[
        m : Y - m,
        m : X - m,
    ]

    surfaces = np.empty(
        (
            T,
            2 * m + 1,
            2 * m + 1,
        ),
        dtype=np.float32,
    )

    for t in range(T):

        frame = preprocess_registration_image(
            movie[t],
            lowpass_sigma,
            highpass_sigma,
        )

        corr = cv2.matchTemplate(
            frame,
            template_crop,
            cv2.TM_CCOEFF_NORMED,
        )

        # OpenCV gives displacement of template within moving image.
        # Reverse axes so our coordinates are instead the correction
        # to APPLY TO the moving image.
        surfaces[t] = corr[
            ::-1,
            ::-1,
        ]

    shifts = np.arange(
        -m,
        m + 1,
        dtype=float,
    )

    return surfaces, shifts


def correlation_to_likelihood(
    surfaces,
    corr_temperature=0.01,
):
    """
    Convert normalized correlation surfaces into discrete likelihoods.

    Larger temperature -> trust each individual frame less.
    """

    surfaces = np.asarray(
        surfaces,
        dtype=np.float64,
    )

    peak = np.nanmax(
        surfaces,
        axis=(1, 2),
        keepdims=True,
    )

    logL = (surfaces - peak) / corr_temperature

    # numerical safety
    logL = np.clip(
        logL,
        -50,
        0,
    )

    likelihood = np.exp(logL)

    likelihood[~np.isfinite(likelihood)] = 0

    likelihood /= (
        likelihood.sum(
            axis=(1, 2),
            keepdims=True,
        )
        + 1e-30
    )

    return likelihood


def make_motion_transition_kernel(
    radius=6,
    sigma_smooth=0.6,
    sigma_fast=2.5,
    fast_weight=0.05,
):
    """
    Transition probability for frame-to-frame motion.

    A mixture gives:
      - strong preference for smooth motion
      - non-negligible probability of rapid motion
    """

    d = np.arange(
        -radius,
        radius + 1,
        dtype=float,
    )

    dx, dy = np.meshgrid(
        d,
        d,
    )

    r2 = dx**2 + dy**2

    K_smooth = np.exp(-0.5 * r2 / sigma_smooth**2)

    K_fast = np.exp(-0.5 * r2 / sigma_fast**2)

    K = (1 - fast_weight) * K_smooth + fast_weight * K_fast

    K /= K.sum()

    return K


from scipy.signal import convolve2d


def smooth_registration_likelihoods(
    likelihood,
    transition_kernel,
):
    """
    Forward-backward inference for discrete 2-D displacement states.

    Returns
    -------
    posterior :
        P(shift_t | all frames)

    alpha :
        forward-filtered distributions

    beta :
        backward messages
    """

    likelihood = np.asarray(
        likelihood,
        dtype=np.float64,
    )

    T, ny, nx = likelihood.shape

    alpha = np.empty_like(likelihood)

    beta = np.empty_like(likelihood)

    # ------------------------------------
    # Initial state: uniform prior
    # ------------------------------------

    alpha[0] = likelihood[0]

    alpha[0] /= alpha[0].sum() + 1e-30

    # ------------------------------------
    # Forward pass
    # ------------------------------------

    for t in range(
        1,
        T,
    ):

        predicted = convolve2d(
            alpha[t - 1],
            transition_kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        alpha[t] = likelihood[t] * predicted

        alpha[t] /= alpha[t].sum() + 1e-30

    # ------------------------------------
    # Backward pass
    # ------------------------------------

    beta[-1] = 1.0
    beta[-1] /= beta[-1].sum()

    # kernel is symmetric here
    Kback = transition_kernel[
        ::-1,
        ::-1,
    ]

    for t in range(
        T - 2,
        -1,
        -1,
    ):

        future = likelihood[t + 1] * beta[t + 1]

        beta[t] = convolve2d(
            future,
            Kback,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        beta[t] /= beta[t].sum() + 1e-30

    # ------------------------------------
    # Smoothed posterior
    # ------------------------------------

    posterior = alpha * beta

    posterior /= (
        posterior.sum(
            axis=(1, 2),
            keepdims=True,
        )
        + 1e-30
    )

    return posterior, alpha, beta


def trajectory_from_posterior(
    posterior,
    shifts,
):
    shifts = np.asarray(
        shifts,
        dtype=float,
    )

    DX, DY = np.meshgrid(
        shifts,
        shifts,
    )

    # ------------------------------------
    # posterior mean
    # ------------------------------------

    dx_mean = np.sum(
        posterior * DX[None, :, :],
        axis=(1, 2),
    )

    dy_mean = np.sum(
        posterior * DY[None, :, :],
        axis=(1, 2),
    )

    # ------------------------------------
    # posterior uncertainty
    # ------------------------------------

    var_x = np.sum(
        posterior * (DX[None, :, :] - dx_mean[:, None, None]) ** 2,
        axis=(1, 2),
    )

    var_y = np.sum(
        posterior * (DY[None, :, :] - dy_mean[:, None, None]) ** 2,
        axis=(1, 2),
    )

    # ------------------------------------
    # marginal MAP
    # ------------------------------------

    flat = posterior.reshape(
        len(posterior),
        -1,
    )

    idx = np.argmax(
        flat,
        axis=1,
    )

    iy, ix = np.unravel_index(
        idx,
        posterior.shape[1:],
    )

    dx_map = shifts[ix]
    dy_map = shifts[iy]

    return {
        "dx_mean": dx_mean,
        "dy_mean": dy_mean,
        "dx_std": np.sqrt(var_x),
        "dy_std": np.sqrt(var_y),
        "dx_map": dx_map,
        "dy_map": dy_map,
    }


def independent_peak_trajectory(
    surfaces,
    shifts,
):
    flat = surfaces.reshape(
        len(surfaces),
        -1,
    )

    idx = np.argmax(
        flat,
        axis=1,
    )

    iy, ix = np.unravel_index(
        idx,
        surfaces.shape[1:],
    )

    return (
        shifts[ix],
        shifts[iy],
    )


def correlation_along_trajectory(
    surfaces,
    shifts,
    dx,
    dy,
):
    scores = np.empty(len(surfaces))

    for t in range(len(surfaces)):
        ix = np.argmin(np.abs(shifts - dx[t]))

        iy = np.argmin(np.abs(shifts - dy[t]))

        scores[t] = surfaces[
            t,
            iy,
            ix,
        ]

    return scores
