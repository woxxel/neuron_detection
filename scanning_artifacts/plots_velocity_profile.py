from matplotlib import pyplot as plt
import numpy as np

from .diagnostics_velocity_profile import resonant_velocity_profile


def plot_resonant_fit(x, width_even, width_odd, fit):
    baseline, amplitude, x0, A = fit["params"]

    x_dense = np.linspace(fit["x"].min(), fit["x"].max(), 1000)

    model_dense = resonant_velocity_profile(
        x_dense,
        baseline,
        amplitude,
        x0,
        A,
    )

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        height_ratios=[3, 1],
        constrained_layout=True,
    )

    ax[0].plot(
        x,
        width_even,
        alpha=0.5,
        label="even",
    )

    ax[0].plot(
        x,
        width_odd,
        alpha=0.5,
        label="odd",
    )

    ax[0].plot(
        x_dense,
        model_dense,
        linewidth=2,
        label="resonant velocity fit",
    )

    ax[0].axvline(
        x0,
        linestyle="--",
        label=f"fit center = {x0:.1f}",
    )

    ax[0].set_ylabel("integrated correlation width [px]")

    ax[0].set_title(f"Resonant-velocity fit, R² = {fit['r2']:.4f}")

    ax[0].legend()

    ax[1].axhline(
        0,
        linestyle="--",
    )

    ax[1].plot(
        fit["x"],
        fit["residuals"],
    )

    ax[1].set_xlabel("x [pixel]")
    ax[1].set_ylabel("residual")

    plt.show()


def compare_models(
    x,
    width,
    first_fit,
    second_fit,
    first_label="resonant velocity",
    second_label="quadratic",
):

    xx = first_fit["x"]

    yr = first_fit["prediction"]
    yq = second_fit["prediction"]

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        constrained_layout=True,
    )

    ax[0].plot(
        x,
        width,
        label="measured",
    )

    ax[0].plot(
        xx,
        yr,
        label=first_label,
    )

    ax[0].plot(
        xx,
        yq,
        label=second_label,
    )

    ax[0].set_ylabel("correlation width [px]")

    ax[0].legend()

    ax[1].plot(
        xx,
        first_fit["residuals"],
        label=first_label,
    )

    ax[1].plot(
        xx,
        second_fit["residuals"],
        label=second_label,
    )

    ax[1].axhline(
        0,
        linestyle="--",
    )

    ax[1].set_xlabel("x")
    ax[1].set_ylabel("residual")

    ax[1].legend()

    # plt.show()
