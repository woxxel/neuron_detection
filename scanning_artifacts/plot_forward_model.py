import matplotlib.pyplot as plt
import numpy as np

from .diagnostics_forward_model import predicted_correlation_map


def plot_covariance_forward_model(
    corr_even,
    corr_odd,
    fit,
):
    measured = 0.5 * (corr_even + corr_odd)

    L = (measured.shape[0] - 1) // 2
    X = measured.shape[1]

    predicted = predicted_correlation_map(
        X=X,
        max_lag=L,
        x0=fit["x0"],
        A=fit["A"],
        tau_theta=fit["tau_theta"],
        alpha=fit["alpha"],
    )

    residual = measured - predicted

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)

    extent = [
        0,
        X - 1,
        -L,
        L,
    ]

    im = axes[0].imshow(
        measured,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-0.25,
        vmax=1,
    )

    axes[0].set_title("Measured noise correlation")
    axes[0].set_ylabel(r"$\Delta x$")

    fig.colorbar(im, ax=axes[0], label="correlation")

    im = axes[1].imshow(
        predicted,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-0.25,
        vmax=1,
    )

    axes[1].set_title("Predicted resonant phase-sinc correlation")
    axes[1].set_ylabel(r"$\Delta x$")

    fig.colorbar(im, ax=axes[1], label="correlation")

    vmax_res = np.nanpercentile(np.abs(residual), 99)

    im = axes[2].imshow(
        residual,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-vmax_res,
        vmax=vmax_res,
    )

    axes[2].set_title("Measured - predicted")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel(r"$\Delta x$")

    fig.colorbar(im, ax=axes[2], label="residual")

    # plt.show()

    return predicted, residual


def plot_measured_vs_predicted_kernels(
    corr_even,
    corr_odd,
    predicted,
    xs=(30, 100, 256, 410, 480),
):
    measured = 0.5 * (corr_even + corr_odd)

    L = (measured.shape[0] - 1) // 2
    lags = np.arange(-L, L + 1)

    fig, axes = plt.subplots(
        len(xs), 1, figsize=(9, 2.3 * len(xs)), sharex=True, constrained_layout=True
    )

    for ax, x in zip(axes, xs):

        ax.plot(
            lags,
            measured[:, x],
            "o-",
            label="measured",
        )

        ax.plot(
            lags,
            predicted[:, x],
            "-",
            label="predicted",
        )

        ax.axhline(
            0,
            linestyle="--",
        )

        ax.set_ylabel(f"x={x}\ncorrelation")

    axes[0].legend()

    axes[-1].set_xlabel(r"$\Delta x$ [pixels]")

    # plt.show()


from .diagnostics_forward_model import estimate_phase_kernel


def plot_phase_kernel_by_x(
    collapsed, ax=None, n_x_bins=5, n_theta_bins=80, disp_legend=True
):
    x = collapsed["x"]

    x_edges = np.linspace(
        np.min(x),
        np.max(x),
        n_x_bins + 1,
    )

    if ax is None:
        plt.figure(figsize=(9, 5))
        ax = plt.gca()

    for i in range(n_x_bins):

        select = (x >= x_edges[i]) & (x < x_edges[i + 1])

        subset = {
            "dtheta": collapsed["dtheta"][select],
            "corr": collapsed["corr"][select],
        }

        theta, K, _, count = estimate_phase_kernel(
            subset,
            n_bins=n_theta_bins,
            min_count=5,
        )

        good = np.isfinite(K)

        label = f"x = {x_edges[i]:.0f}" f"–{x_edges[i+1]:.0f}"

        ax.plot(
            theta[good],
            K[good],
            label=label,
        )

    ax.axhline(0, ls="--")

    ax.set_xlabel(r"$\Delta\theta$ [rad]")
    ax.set_ylabel("correlation")

    ax.set_title("Phase-domain covariance by x region")

    if disp_legend:
        ax.legend(fontsize=8)
