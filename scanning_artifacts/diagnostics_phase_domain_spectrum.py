import numpy as np
from scipy.ndimage import gaussian_filter1d


def empirical_phase_covariance(
    collapsed,
    theta_max=None,
    n_positive=256,
    smooth_sigma=1.0,
):
    """
    Construct an evenly sampled, symmetric phase-domain
    correlation function K(dtheta).

    Returns
    -------
    theta : symmetric phase-lag axis
    K     : correlation kernel, with K(0)=1
    """

    dtheta = np.asarray(collapsed["dtheta"], dtype=float)

    corr = np.asarray(collapsed["corr"], dtype=float)

    good = np.isfinite(dtheta) & np.isfinite(corr)

    dtheta = dtheta[good]
    corr = corr[good]

    if theta_max is None:
        theta_max = np.nanpercentile(dtheta, 99)

    edges = np.linspace(0, theta_max, n_positive + 1)

    centers = 0.5 * (edges[:-1] + edges[1:])

    Kpos = np.full(n_positive, np.nan)

    for i in range(n_positive):

        select = (dtheta >= edges[i]) & (dtheta < edges[i + 1])

        if np.any(select):
            Kpos[i] = np.nanmean(corr[select])

    # interpolate empty bins
    valid = np.isfinite(Kpos)

    Kpos[~valid] = np.interp(
        centers[~valid],
        centers[valid],
        Kpos[valid],
    )

    if smooth_sigma is not None:
        Kpos = gaussian_filter1d(Kpos, smooth_sigma)

    # Add exact zero-lag correlation
    dtheta_step = centers[1] - centers[0]

    theta_pos = np.concatenate(
        [
            [0.0],
            centers,
        ]
    )

    Kpos = np.concatenate(
        [
            [1.0],
            Kpos,
        ]
    )

    # Symmetrize
    theta = np.concatenate(
        [
            -theta_pos[:0:-1],
            theta_pos,
        ]
    )

    K = np.concatenate(
        [
            Kpos[:0:-1],
            Kpos,
        ]
    )

    return theta, K, dtheta_step


def covariance_to_spectrum(
    theta,
    K,
    taper=True,
    pad_factor=8,
):
    """
    something's flawed in here:
    ------------------
    Fourier transform a symmetric covariance kernel.

    Frequency output is in cycles / radian of scanner phase.
    """

    theta = np.asarray(theta)
    K = np.asarray(K)

    dtheta = np.mean(np.diff(theta))

    k = K.copy()

    if taper:
        window = np.hanning(len(k))
        k *= window

    nfft = int(2 ** np.ceil(np.log2(len(k) * pad_factor)))

    # Move zero lag to index 0 before FFT
    spectrum = np.real(np.fft.rfft(np.fft.ifftshift(k), n=nfft))

    freq = np.fft.rfftfreq(nfft, d=dtheta)

    # Numerical/truncation artifacts can make tiny negative values.
    spectrum = np.maximum(spectrum, 0)

    # normalize
    spectrum /= np.nanmax(spectrum)

    return freq, spectrum


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


def covariance_cosine_spectrum(
    theta_bin,
    K_empirical,
    K0,
    theta_max=None,
    n_theta=4096,
    f_max=100.0,
    n_freq=2000,
    taper_fraction=0.15,
):
    """
    Continuous cosine transform of an empirical even covariance kernel.

    Parameters
    ----------
    theta_bin : array
        Positive phase-lag coordinates.

    K_empirical : array
        Empirical covariance/correlation at theta_bin.

    K0 : float
        Smooth correlated covariance at zero lag.
        For the correlated component this can be alpha from the
        damped-sinc fit, rather than total C(0)=1.

    theta_max : float or None
        Maximum measured theta to use. If larger than the available
        data range, it is automatically clipped.

    Returns
    -------
    freq : cycles / rad
    psd  : normalized spectral power
    H    : sqrt(psd), approximate relative transfer amplitude
    """

    theta_bin = np.asarray(theta_bin, dtype=float)
    K_empirical = np.asarray(K_empirical, dtype=float)

    # ----------------------------------------
    # Remove invalid entries
    # ----------------------------------------
    valid = np.isfinite(theta_bin) & np.isfinite(K_empirical) & (theta_bin > 0)

    theta_bin = theta_bin[valid]
    K_empirical = K_empirical[valid]

    # ----------------------------------------
    # Sort
    # ----------------------------------------
    order = np.argsort(theta_bin)

    theta_bin = theta_bin[order]
    K_empirical = K_empirical[order]

    # Remove duplicate theta values if present
    theta_bin, unique_idx = np.unique(theta_bin, return_index=True)
    K_empirical = K_empirical[unique_idx]

    # ----------------------------------------
    # Limit to requested range
    # ----------------------------------------
    available_max = theta_bin[-1]

    if theta_max is None:
        theta_max_eff = available_max
    else:
        theta_max_eff = min(float(theta_max), available_max)

    use = theta_bin <= theta_max_eff

    theta_data = np.concatenate(
        [
            [0.0],
            theta_bin[use],
        ]
    )

    K_data = np.concatenate(
        [
            [float(K0)],
            K_empirical[use],
        ]
    )

    # IMPORTANT:
    # integrate only as far as the last actual interpolation point,
    # not the nominal theta_max.
    theta_max_eff = theta_data[-1]

    # ----------------------------------------
    # Uniform integration grid
    # ----------------------------------------
    theta = np.linspace(0.0, theta_max_eff, n_theta)

    interp = PchipInterpolator(theta_data, K_data, extrapolate=False)

    k = interp(theta)

    # Sanity check
    if np.any(~np.isfinite(k)):
        raise RuntimeError(
            "Interpolation still produced non-finite values. "
            f"theta range = [{theta[0]}, {theta[-1]}], "
            f"data range = [{theta_data[0]}, {theta_data[-1]}]"
        )

    # ----------------------------------------
    # Gently taper only the far end
    # ----------------------------------------
    if taper_fraction > 0:

        n_taper = int(taper_fraction * len(theta))

        if n_taper > 1:

            window = np.ones_like(theta)

            u = np.linspace(0, np.pi / 2, n_taper)

            window[-n_taper:] = np.cos(u) ** 2

            k *= window

    # ----------------------------------------
    # Continuous cosine transform
    # ----------------------------------------
    freq = np.linspace(0.0, f_max, n_freq)

    psd = np.empty_like(freq)

    for i, f in enumerate(freq):

        psd[i] = 2.0 * np.trapezoid(k * np.cos(2 * np.pi * f * theta), theta)

    # ----------------------------------------
    # Normalize
    # ----------------------------------------
    if not np.isfinite(psd[0]) or psd[0] <= 0:
        raise RuntimeError(f"Invalid zero-frequency PSD: {psd[0]}")

    psd /= psd[0]

    # Tiny negative values may result from finite/noisy covariance.
    H = np.sqrt(np.clip(psd, 0, None))

    return freq, psd, H


def crossing_frequency(
    freq,
    response,
    threshold,
):
    below = np.where(response <= threshold)[0]

    if len(below) == 0:
        return np.nan

    i = below[0]

    if i == 0:
        return freq[0]

    # linear interpolation
    f0, f1 = freq[i - 1 : i + 1]
    r0, r1 = response[i - 1 : i + 1]

    return f0 + (threshold - r0) * (f1 - f0) / (r1 - r0)


def local_bandwidth_from_phase_cutoff(
    X,
    x0,
    A,
    f_phase_cutoff,
):
    x = np.arange(X)

    dx_dtheta = np.sqrt(np.maximum(A**2 - (x - x0) ** 2, 0))

    spatial_cutoff = f_phase_cutoff / dx_dtheta

    nyquist_spacing = dx_dtheta / (2 * f_phase_cutoff)

    return (
        spatial_cutoff,
        nyquist_spacing,
    )
