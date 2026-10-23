import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Optional
import tifffile as tiff
from tifffile import imread, imwrite
from scipy.ndimage import median_filter

from ..utils.build_alignment_template import build_alignment_template

import caiman, caiman.cluster
from caiman.source_extraction import cnmf as cnmf
from caiman.motion_correction import MotionCorrect


def motion_correct(
    fname,
    params,
    path_destination: Optional[str | Path] = None,
    fname_stem: Optional[str] = None,
    n_processes: Optional[int] = None,
    template: Optional[np.ndarray] = None,
    plot: bool = False,
):
    """
    Runs the motion correction algorithm NoRMCorr and returns the path to the output file

    n_processes shouldnt be too large, as
    """
    use_parallel = n_processes is not None and n_processes > 1
    fname_stem = fname_stem if fname_stem is not None else Path(fname).stem

    print(f"\tNow running motion correction @t = {time.ctime()}")

    t_start = time.time()  # start time measurement from here

    print(f"Building alignment template @t = {time.ctime()}")
    movie = tiff.imread(fname).astype("float32")
    k = max(int(movie.shape[0] * 0.05), 100)
    template, aligned = build_alignment_template(movie, N_iter=1, k=k, plot=False)

    # normalize the template to the range [0, 1]
    template = (template - template.min()) / (template.max() - template.min())
    import cv2

    figure_dir = Path(path_destination) / "figures"
    figure_dir.mkdir(exist_ok=True, parents=True)
    background_path = figure_dir / f"template_{fname_stem}.png"
    cv2.imwrite(str(background_path), template * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print(
        f"Alignment template built @t = {time.ctime()} and saved to {background_path}"
    )
    # print(f"Loading data from {fname} for median filtering...")
    # data = imread(fname)
    # data = data.astype(np.float32)
    # data_filtered = median_filter(data, size=(5, 1, 1))
    # fname_short_median = fname.parent / f"{fname.stem}_median{fname.suffix}"
    # print(f"Saving filtered data to {fname_short_median}...")
    # imwrite(fname_short_median, data_filtered.astype(data.dtype))

    ## initialize parameters and settings for running OnACID
    #
    opts = cnmf.params.CNMFParams(params_dict=params)
    if use_parallel:
        c, dview, n_processes = caiman.cluster.setup_cluster(
            n_processes=n_processes, single_thread=False
        )
    else:
        dview = None
        n_processes = 1

    print("Now running the actual motion correction...")
    ## run the actual motion correction algorithm
    mc = MotionCorrect(fname, dview=dview, **opts.get_group("motion"))
    mc.motion_correct(template=template, save_movie=True)
    # mc = MotionCorrect(fname_short_median, dview=dview, **opts.get_group("motion"))
    # mc.motion_correct(save_movie=False)
    # print("Applying shifts to original data")
    # path_to_motion_correct = mc.apply_shifts_movie(fname, save_memmap=True)
    # mc.mmap_file = [path_to_motion_correct]

    if use_parallel:
        caiman.stop_server(dview=dview)  ## restart server to clean up memory

    if path_destination is None:
        return mc.mmap_file[0]

    ## calculate and store shifts
    shifts = calculate_shifts(mc)

    m_rig = caiman.load(mc.mmap_file)
    Cn = caiman.local_correlations(m_rig, swap_dim=False)

    results = {
        "motion_corrected_file": mc.mmap_file[0],
        "shifts": shifts,
        "Cn": Cn,
        "template": template,  # mc.total_template_els if mc.pw_rigid else mc.total_template_rig,
    }

    with open(
        f"{path_destination}/motion_correction_summary_{fname_stem}.pkl", "wb"
    ) as f:
        pickle.dump(results, f)

    ## create plots
    if plot:
        figure_dir = Path(path_destination) / "figures"
        figure_dir.mkdir(exist_ok=True)
        plot_results_motion_correction(
            results["shifts"],
            results["Cn"],
            results["template"],
            figure_path=figure_dir / f"motion_correction_summary_{fname_stem}.png",
        )

    print(
        "\tMotion correction done @t = %s, (time passed: %s)"
        % (time.ctime(), str(time.time() - t_start))
    )

    return mc.mmap_file[0]


def calculate_shifts(mc: MotionCorrect):
    shifts = {}
    if mc.pw_rigid:
        shifts["piecewise"] = {
            "x_shifts": mc.x_shifts_els,
            "y_shifts": mc.y_shifts_els,
            "x_shifts_max": np.percentile(
                np.concatenate(mc.x_shifts_els), [1, 99], axis=0
            ),
            "y_shifts_max": np.percentile(
                np.concatenate(mc.y_shifts_els), [1, 99], axis=0
            ),
        }
    else:
        x_shifts_rig, y_shifts_rig = np.percentile(mc.shifts_rig, [1, 99], axis=0).T
        shifts["rigid"] = {
            "x_shifts": np.array(mc.shifts_rig)[:, 0],
            "y_shifts": np.array(mc.shifts_rig)[:, 1],
            "x_shifts_max": x_shifts_rig,
            "y_shifts_max": y_shifts_rig,
        }

    return shifts


def plot_results_motion_correction(
    shifts, Cn, template, figure_path: Optional[str | Path] = None
):

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(template, origin="lower", aspect="auto", cmap="gray")
    plt.title("motion correction template")
    plt.subplot(2, 2, 2)
    # plt.imshow(mc.total_template_els, origin="lower",cmap="gray")
    plt.imshow(Cn, origin="lower", aspect="auto", cmap="viridis")
    plt.title("local correlations after rigid motion correction")

    plt.subplot(4, 4, (9, 11))
    if "piecewise" in shifts:
        plt.plot(shifts["piecewise"]["x_shifts"], linewidth=0.2)
    else:
        plt.plot(shifts["rigid"]["x_shifts"], linewidth=0.5)

    # plt.hist2d(np.concatenate(mc.x_shifts_els), np.concatenate(mc.y_shifts_els), bins=50, cmap='hot')
    if "piecewise" in shifts:
        text_str = f"\n x shifts (pw): {shifts['piecewise']['x_shifts_max']}"
    else:
        text_str = f"x shifts (rig): {shifts['rigid']['x_shifts_max']}"
    plt.text(
        0.5,
        0.8,
        text_str,
        transform=plt.gca().transAxes,
        ha="center",
    )
    plt.ylabel("x shifts")
    plt.gca().spines[["top", "right"]].set_visible(False)

    plt.subplot(4, 4, (13, 15))
    if "piecewise" in shifts:
        plt.plot(shifts["piecewise"]["y_shifts"], linewidth=0.2)
    else:
        plt.plot(shifts["rigid"]["y_shifts"], linewidth=0.5)
    plt.xlabel("frames")
    plt.ylabel("y shifts")
    if "piecewise" in shifts:
        text_str = f"\n y shifts (pw): {shifts['piecewise']['y_shifts_max']}"
    else:
        text_str = f"y shifts (rig): {shifts['rigid']['y_shifts_max']}"
    plt.text(
        0.5,
        0.8,
        text_str,
        transform=plt.gca().transAxes,
        ha="center",
    )
    plt.gca().spines[["top", "right"]].set_visible(False)

    ax_scatter = plt.subplot(2, 4, 8)

    ax = ax_scatter.twinx()
    if "piecewise" in shifts:
        hist_data = np.concatenate(shifts["piecewise"]["x_shifts"])
    else:
        hist_data = shifts["rigid"]["x_shifts"].copy()
    ax.hist(hist_data, bins=50, alpha=0.5, color="C0")

    plt.setp(ax, yticks=[])
    ax.spines[["top", "right"]].set_visible(False)

    ax = ax_scatter.twiny()
    if "piecewise" in shifts:
        hist_data = np.concatenate(shifts["piecewise"]["y_shifts"])
    else:
        hist_data = shifts["rigid"]["y_shifts"].copy()
    ax.hist(
        hist_data,
        bins=50,
        alpha=0.5,
        color="C0",
        orientation="horizontal",
    )
    plt.setp(ax, xticks=[])
    ax.spines[["top", "right"]].set_visible(False)

    key = "piecewise" if "piecewise" in shifts else "rigid"
    ax_scatter.plot(
        shifts[key]["x_shifts"],
        shifts[key]["y_shifts"],
        marker=".",
        markersize=1.0,
        markeredgecolor=None,
        linestyle="None",
        alpha=1.0,
    )
    plt.setp(ax_scatter, xlabel="x shifts (pixels)", ylabel="y shifts (pixels)")
    ax_scatter.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if figure_path is not None:
        plt.savefig(figure_path, dpi=150)
    plt.show()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
