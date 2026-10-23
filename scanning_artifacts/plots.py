from matplotlib import pyplot as plt
import numpy as np

from turnover_dynamics.neuron_detection_caiman.scanning_artifacts.diagnostics_forward_model import (
    phase_collapse_data,
)
from turnover_dynamics.neuron_detection_caiman.scanning_artifacts.diagnostics_framewise import (
    bidi_shift_timeseries,
    bidi_shift_x_time,
)

from turnover_dynamics.neuron_detection_caiman.scanning_artifacts.plot_forward_model import (
    plot_phase_kernel_by_x,
)


def plot_correlation_heatmaps(subfigure, corr_even, corr_odd):
    L = (corr_even.shape[0] - 1) // 2
    X = corr_even.shape[1]

    fig, axes = subfigure.subplots(1, 2, figsize=(12, 6))

    im = axes[0].imshow(
        corr_even,
        origin="lower",
        aspect="auto",
        extent=[0, X - 1, -L, L],
        vmin=-0.2,
        vmax=1,
    )

    axes[0].set_title("Even rows: noise correlation")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$\Delta x$")

    fig.colorbar(im, ax=axes[0], label="correlation")

    im = axes[1].imshow(
        corr_odd,
        origin="lower",
        aspect="auto",
        extent=[0, X - 1, -L, L],
        vmin=-0.2,
        vmax=1,
    )

    axes[1].set_title("Odd rows: noise correlation")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$\Delta x$")

    fig.colorbar(im, ax=axes[1], label="correlation")


def plot_resonant_diagnostics(
    diagnostics, joint_fit, with_collapse=False, save_path=None
):

    corr_even = diagnostics["corr_even"]
    corr_odd = diagnostics["corr_odd"]

    L = (corr_even.shape[0] - 1) // 2
    X = corr_even.shape[1]

    fig = plt.figure(figsize=(12, 3))  # , constrained_layout=True)

    gs = fig.add_gridspec(1, 3)
    gs_fine = fig.add_gridspec(2, 3)
    # plot_correlation_heatmaps(fig, diagnostics["corr_even"], diagnostics["corr_odd"])

    # --------------------------------------------------
    # correlation heatmaps
    # --------------------------------------------------

    ax_corr = fig.add_subplot(gs[0, 0])
    im = ax_corr.imshow(
        corr_even,
        origin="lower",
        aspect="auto",
        extent=[0, X - 1, -L, L],
        vmin=-0.2,
        vmax=1,
    )

    ax_corr.set_title("Even rows: noise correlation")
    ax_corr.set_xlabel("x")
    ax_corr.set_ylabel(r"$\Delta x$")

    fig.colorbar(im, ax=ax_corr, label="correlation")

    if with_collapse:
        ax = fig.add_subplot(gs_fine[0, 1])

        collapsed = phase_collapse_data(
            diagnostics["corr_even"],
            diagnostics["corr_odd"],
            x0=joint_fit["x0"],
            A=joint_fit["A"],
        )

        ax_collapsed = fig.add_subplot(gs_fine[1, 1])
        plot_phase_kernel_by_x(collapsed, ax=ax_collapsed, disp_legend=False)
        ax_collapsed.spines[["top", "right"]].set_visible(False)
    else:
        ax = fig.add_subplot(gs[0, 1])
    plot_local_corr_profiles(
        diagnostics, ax=ax, xs=(30, 100, 256, 410, 480), disp_legend=not with_collapse
    )
    ax.spines[["top", "right"]].set_visible(False)

    # im = axes[0, 1].imshow(
    #     corr_odd,
    #     origin="lower",
    #     aspect="auto",
    #     extent=[0, X - 1, -L, L],
    #     vmin=-0.2,
    #     vmax=1,
    # )

    # axes[0, 1].set_title("Odd rows: noise correlation")
    # axes[0, 1].set_xlabel("x")
    # axes[0, 1].set_ylabel(r"$\Delta x$")

    # fig.colorbar(im, ax=axes[0, 1], label="correlation")

    # --------------------------------------------------
    # correlation width
    # --------------------------------------------------

    ax_corr_width = fig.add_subplot(gs[0, 2])
    ax_corr_width.plot(diagnostics["metric_even"]["corr_area"], label="even")

    ax_corr_width.plot(diagnostics["metric_odd"]["corr_area"], label="odd")

    ax_corr_width.set_ylim([0, 5.0])
    ax_corr_width.set_title("Integrated local noise-correlation width")

    ax_corr_width.set_xlabel("x")
    ax_corr_width.set_ylabel("correlation width [px]")
    ax_corr_width.legend()

    ax_corr_width.spines[["top", "right"]].set_visible(False)

    # --------------------------------------------------
    # noise variance
    # --------------------------------------------------

    # ax_noise = fig.add_subplot(gs[1, 1])
    # ax_noise.plot(diagnostics["noise_even"], label="even")
    # ax_noise.plot(diagnostics["noise_odd"], label="odd")

    # ax_noise.set_title("Frame-difference noise amplitude")

    # ax_noise.set_xlabel("x")
    # ax_noise.set_ylabel("noise SD")
    # ax_noise.legend()

    # --------------------------------------------------
    # odd/even displacement
    # --------------------------------------------------

    # axes[2, 0].plot(results["phase_x"], results["phase_shift"], "o-")

    # axes[2, 0].axhline(0, linestyle="--")

    # axes[2, 0].set_title("Local odd/even scan-direction mismatch")

    # axes[2, 0].set_xlabel("x")
    # axes[2, 0].set_ylabel("relative displacement [px]")

    # # --------------------------------------------------
    # # structural scale
    # # --------------------------------------------------

    # valid = (results["structure_x"] > 40) & (
    #     results["structure_x"] < results["corr_even"].shape[1] - 40
    # )

    # axes[2, 1].plot(
    #     results["structure_x"][valid], results["structure_ratio"][valid], "o-"
    # )

    # axes[2, 1].axhline(1, linestyle="--")

    # axes[2, 1].set_title("Relative x/y structural-scale proxy")

    # axes[2, 1].set_xlabel("x")
    # axes[2, 1].set_ylabel("relative anisotropy")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show()


def plot_local_corr_profiles(
    results, ax=None, xs=(25, 100, 256, 410, 485), disp_legend=True
):
    corr_even = results["corr_even"]
    corr_odd = results["corr_odd"]

    L = (corr_even.shape[0] - 1) // 2
    lags = np.arange(-L, L + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    for x in xs:
        c = 0.5 * (corr_even[:, x] + corr_odd[:, x])
        ax.plot(lags, c, marker="o", markersize=2, linewidth=1.0, label=f"x={x}")

    ax.axhline(0, linestyle="--")
    ax.set_xlabel(r"$\Delta x$ [pixels]")
    ax.set_ylabel("noise correlation")
    ax.set_title("Local noise-correlation kernels")
    if disp_legend:
        ax.legend(fontsize=8)


def plot_motion_scale_diagnostic(
    results,
    example_strips=None,
):
    dx_global = results["dx_global"]
    dy_global = results["dy_global"]

    x_centers = results["x_centers"]
    dx_local = results["dx_local"]

    scale = results["scale"]
    r2 = results["r2"]

    if example_strips is None:
        n = len(x_centers)
        example_strips = [
            1,
            n // 4,
            n // 2,
            3 * n // 4,
            n - 2,
        ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # ------------------------------------------------
    # Global motion
    # ------------------------------------------------

    axes[0, 0].plot(dx_global, label="x")

    axes[0, 0].plot(dy_global, label="y")

    axes[0, 0].set_title("Global estimated tissue motion")

    axes[0, 0].set_xlabel("temporally binned frame")

    axes[0, 0].set_ylabel("registration shift [px]")

    axes[0, 0].legend()

    # ------------------------------------------------
    # Local vs global shift
    # ------------------------------------------------

    for j in example_strips:

        valid = np.isfinite(dx_global) & np.isfinite(dx_local[:, j])

        axes[0, 1].scatter(
            dx_global[valid],
            dx_local[valid, j],
            s=8,
            alpha=0.35,
            label=f"x={x_centers[j]:.0f}",
        )

        slope = results["slope"][j]
        intercept = results["intercept"][j]

        xx = np.array(
            [
                np.nanmin(dx_global),
                np.nanmax(dx_global),
            ]
        )

        axes[0, 1].plot(xx, intercept + slope * xx)

    axes[0, 1].set_title("Local motion vs global motion")

    axes[0, 1].set_xlabel("global dx [px]")

    axes[0, 1].set_ylabel("local dx [px]")

    axes[0, 1].legend()

    # ------------------------------------------------
    # Main result: relative scale
    # ------------------------------------------------

    axes[1, 0].plot(x_centers, scale, "o-")

    axes[1, 0].axhline(1, linestyle="--")

    axes[1, 0].set_title("Motion-derived relative x scale")

    axes[1, 0].set_xlabel("x")

    axes[1, 0].set_ylabel("relative local x scale")

    # ------------------------------------------------
    # Fit quality
    # ------------------------------------------------

    axes[1, 1].plot(x_centers, r2, "o-")

    axes[1, 1].set_ylim(0, 1)

    axes[1, 1].set_title("Local motion-fit quality")

    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel(r"$R^2$")

    plt.show()


def plot_bidi_shift_time(img):
    bidi_shift, bidi_quality = bidi_shift_timeseries(img)

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(bidi_shift)
    ax.axhline(0, ls="--")

    ax.set_xlabel("frame")
    ax.set_ylabel("odd/even x displacement [px]")
    ax.set_title("Time-resolved bidirectional phase offset")

    plt.show()


def plot_bidi_shift_x_time(img):
    x, bidi_xt, q_xt = bidi_shift_x_time(img)

    plt.figure(figsize=(11, 6))

    plt.imshow(
        bidi_xt,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], 0, img.shape[0]],
    )

    plt.colorbar(label="odd/even displacement [px]")

    plt.xlabel("x")
    plt.ylabel("frame")
    plt.title("Time-dependent bidirectional displacement")

    plt.show()


def plot_bidi_y_time(
    y,
    shifts,
    quality=None,
    peak_margin=None,
    vmax=None,
):
    if vmax is None:
        vmax = np.nanpercentile(np.abs(shifts), 99)

    fig, axes = plt.subplots(
        1 if quality is None else 3, 1, figsize=(12, 10), constrained_layout=True
    )

    if quality is None:
        axes = [axes]

    im = axes[0].imshow(
        shifts,
        origin="lower",
        aspect="auto",
        extent=[y[0], y[-1], 0, shifts.shape[0]],
        vmin=-vmax,
        vmax=vmax,
    )

    axes[0].set_xlabel("y")
    axes[0].set_ylabel("frame")
    axes[0].set_title("Y-resolved bidirectional displacement")

    fig.colorbar(im, ax=axes[0], label="odd/even displacement [px]")

    if quality is not None:

        im = axes[1].imshow(
            quality,
            origin="lower",
            aspect="auto",
            extent=[y[0], y[-1], 0, quality.shape[0]],
        )

        axes[1].set_xlabel("y")
        axes[1].set_ylabel("frame")
        axes[1].set_title("Maximum registration correlation")

        fig.colorbar(im, ax=axes[1], label="correlation")

        im = axes[2].imshow(
            peak_margin,
            origin="lower",
            aspect="auto",
            extent=[y[0], y[-1], 0, peak_margin.shape[0]],
        )

        axes[2].set_xlabel("y")
        axes[2].set_ylabel("frame")
        axes[2].set_title("Correlation-peak margin")

        fig.colorbar(im, ax=axes[2], label="best - second best correlation")

    # plt.show()
