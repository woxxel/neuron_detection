import time
from pathlib import Path

import numpy as np
import caiman, caiman.cluster
from caiman.source_extraction import cnmf as cnmf
from caiman.source_extraction.cnmf.params import CNMFParams

import caiman.utils.visualization as cm_viz
from matplotlib import pyplot as plt
from typing import Optional, List

"""
    TODO:
     - store all parameters properly (in yml file or something?)
"""


def init_dview(n_processes, dview=None):

    if dview:
        caiman.stop_server(dview=dview)

    use_parallel = n_processes is not None and n_processes > 1
    if use_parallel:
        c, dview, n_processes = caiman.cluster.setup_cluster(
            n_processes=n_processes, single_thread=False
        )
    else:
        dview = None
        n_processes = 1
    return dview, n_processes


def neuron_detection(
    fname,
    params_dict,
    seeded_init: Optional[dict] = None,
    path_destination: str | Path = ".",
    prefix: str = "results_CaImAn",
    fname_stem: Optional[str] = None,
    save_type: str = "hdf5",
    n_processes: Optional[int] = 1,
    plot: bool = False,
):
    """
    Runs the neuron detection algorithm OnACID and returns the path to the output file
    """

    result_path = str(Path(path_destination) / f"{prefix}_{fname_stem}.{save_type}")
    print(f"in the end, saving results to: {result_path}")

    t_start = time.time()  # start time measurement from here

    params_dict["fnames"] = [fname]
    params_dict["n_processes"] = n_processes
    params_dict["p"] = 2
    params_dict["optimize_g"] = False

    params = CNMFParams(params_dict=params_dict)
    f = params.get("data", "fr")

    Yr, dims, num_frames = caiman.load_memmap(str(fname))  # loading file handle only
    images = np.reshape(Yr.T, [num_frames] + list(dims), order="F")
    if seeded_init is None:
        print(f"\tNow running neuron detection from scratch @t = {time.ctime()}")
        ## analyzing data from scratch
        params.set("online", {"epochs": 1, "init_method": "cnmf"})

        m_rig = caiman.load(str(fname))
        Cn = caiman.local_correlations(m_rig, swap_dim=False)
        del m_rig

        dview, n_processes = init_dview(n_processes)
        cnmf_obj = cnmf.online_cnmf.OnACID(params=params, dview=dview)
        cnmf_obj.fit_online()

        cnmf_obj.estimates.evaluate_components(images, cnmf_obj.params, dview=dview)
        setattr(cnmf_obj.estimates, "Cn", Cn)
        print(
            f"Num accepted/rejected: {len(cnmf_obj.estimates.idx_components)}, {len(cnmf_obj.estimates.idx_components_bad)}"  # type: ignore
        )
        del images

        if plot:
            try:
                plot_contours(cnmf_obj.estimates)
            except Exception as e:
                print("Could not create plots:", e)

        cnmf_obj.estimates.select_components(
            use_object=True, save_discarded_components=False
        )
        cnmf_obj.save(str(result_path))
        _ = init_dview(0, dview=dview)

        cnmf_obj = cnmf.cnmf.load_CNMF(str(result_path))
    else:
        ## analyzing data when footprint initialization is provided
        ## (e.g. from earlier runs, other sessions or channels)
        print(
            f"\tNow running neuron detection with seeded initialization @t = {time.ctime()}"
        )
        assert all(
            key in seeded_init for key in ["A", "C", "b", "f", "Cn"]
        ), "seeded_init must contain keys 'A', 'C', 'b', 'f', 'Cn'"
        cnmf_obj = cnmf.cnmf.CNMF(
            n_processes=12,
            Ain=seeded_init["A"],
            Cin=seeded_init["C"],
            b_in=seeded_init["b"],
            f_in=seeded_init["f"],
            params=params,
        )
        setattr(cnmf_obj.estimates, "Cn", seeded_init["Cn"])

        # if seeded_init is not None:

        Y = np.transpose(images, list(range(1, len(dims) + 1)) + [0])
        # Y_F = np.transpose(np.reshape(images, (num_frames, -1), order="F"))
        Yr = np.transpose(np.reshape(images, (num_frames, -1), order="F"))
        del images

        Y.filename = str(fname)
        # Y_F.filename = str(fname)
        Yr.filename = str(fname)

        ## run initial temporal update to initialize calcium traces for seeded neurons
        ## (refit starts with spatial update and produces crap)

        # run this non-parallel, as n_pixels_per_process is not set
        # Y_F = cnmf_obj.preprocess(Y_F)
        Yr = cnmf_obj.preprocess(Yr)
        print("preprocessing done")
        # cnmf_obj.update_temporal(Y_F)
        cnmf_obj.update_temporal(Yr)
        print("temporal update done")

    cnmf_obj.params.set("temporal", {"p": 2, "optimize_g": False})
    cnmf_obj.mmap_file = str(fname)

    dview, n_processes = init_dview(n_processes)

    images = np.reshape(Yr.T, [num_frames] + list(dims), order="C")
    cnmf_obj = cnmf_obj.refit(images, dview)

    dview, n_processes = init_dview(n_processes, dview=dview)

    cnmf_obj.estimates.detrend_df_f(
        quantileMin=8, frames_window=int(10 * f), flag_auto=False, use_residuals=False
    )
    print("Detrended df/f")

    setattr(cnmf_obj.estimates, "dview", dview)
    # cnmf_obj.estimates.compute_residuals(Y_F)
    cnmf_obj.estimates.compute_residuals(Yr)
    # del Y_F
    del Yr
    print("Computed residuals")

    cnmf_obj.estimates.evaluate_components(images, cnmf_obj.params, dview=dview)
    print("Evaluated components")
    del images

    print(
        f"Num accepted/rejected: {len(cnmf_obj.estimates.idx_components)}, {len(cnmf_obj.estimates.idx_components_bad)}"  # type: ignore
    )

    figure_dir = Path(path_destination) / "figures"
    if plot:
        figure_dir.mkdir(exist_ok=True)
        try:
            plot_contours(
                cnmf_obj.estimates,
                figure_path=figure_dir / f"neuron_contours_{fname_stem}.png",
            )
            plot_quality(
                cnmf_obj, figure_path=figure_dir / f"neuron_quality_{fname_stem}.png"
            )
        except Exception as e:
            print("Could not create plots:", e)
    else:
        cnmf_obj.estimates.coordinates = cm_viz.get_contours(
            cnmf_obj.estimates.A, dims, thr=0.2, thr_method="max"
        )

    cnmf_obj.estimates.select_components(
        use_object=True, save_discarded_components=False
    )

    ## run this after selection, as selection doesnt properly clean up S_dff, F_dff_dec
    try:
        cnmf_obj.estimates.deconvolve(cnmf_obj.params, dview=dview, dff_flag=False)
        cnmf_obj.estimates.deconvolve(cnmf_obj.params, dview=dview, dff_flag=True)
        print("Deconvolved traces")
    except Exception as e:
        print("Could not deconvolve traces:", e)

    _ = init_dview(0, dview=dview)

    retain_keys = [
        "A",
        "C",
        "S",
        "S_dff",
        "F_dff",
        "F_dff_dec",
        "b",
        "f",
        "R",
        "Cn",
        "dims",
        "coordinates",
        "center",
        "SNR_comp",
        "r_values",
        "cnn_preds",
        "sn",
        "bl",
        "c1",
        "neurons_sn",
        "g",
        "YrA",
    ]

    cnmf_obj.estimates = clear_cnm(cnmf_obj.estimates, retain=retain_keys)  # type: ignore

    if plot:
        try:
            plot_traces(
                cnmf_obj.estimates,
                f=f,
                figure_path=figure_dir / f"neuron_traces_{fname_stem}.png",
                n_plots=(5, 5),
            )
        except Exception as e:
            print("Could not create plots:", e)

    print(f"Saving results to {result_path}")
    cnmf_obj.save(str(result_path))

    print(
        "\tNeuron detection done @t = %s, (time passed: %s)"
        % (time.ctime(), str(time.time() - t_start))
    )

    return result_path


def clear_cnm(dic, retain=None, remove=None):

    if retain is None and remove is None:
        print("Please provide a list of keys to obtain in the structure")
        return

    if not (remove is None):
        keys = list(dic.__dict__.keys())
        for key in keys:
            if key in remove:
                dic.__dict__.pop(key)

    if not (retain is None):
        keys = list(dic.__dict__.keys())
        for key in keys:
            if key not in retain:
                dic.__dict__.pop(key)

    return dic


# -------------- PLOT FUNCTIONS ----------------- #


def plot_contours(
    est,
    idx_good=None,
    idx_bad=None,
    display_numbers=False,
    thr=0.2,
    thr_method="max",
    figure_path=None,
):

    idx_good = getattr(est, "idx_components", None) if idx_good is None else idx_good
    idx_bad = getattr(est, "idx_components_bad", None) if idx_bad is None else idx_bad

    if idx_good is None:
        idx_good = np.arange(est.A.shape[1])

    img = est.Cn
    # if est.coordinates is None:  # not hasattr(self, 'coordinates'):
    est.coordinates = cm_viz.get_contours(
        est.A, img.shape, thr=thr, thr_method=thr_method
    )

    fig = plt.figure(figsize=(20, 10))
    coor_g = [est.coordinates[cr] for cr in idx_good]
    plt.subplot(1, 2, 1)
    cm_viz.plot_contours(
        est.A[:, idx_good],
        img,
        coordinates=coor_g,
        display_numbers=display_numbers,
        cmap="viridis",
        origin="lower",
    )
    plt.setp(plt.gca(), xlim=(0, img.shape[1]), ylim=(img.shape[0], 0))
    plt.title(f"Accepted Components (n={len(idx_good)})")

    if idx_bad is not None:
        # bad = list(set(range(self.A.shape[1])) - set(cnmf_obj.estimates.idx_components))
        coor_b = [est.coordinates[cr] for cr in idx_bad]
        plt.subplot(1, 2, 2)
        cm_viz.plot_contours(
            est.A[:, idx_bad],
            img,
            coordinates=coor_b,
            display_numbers=display_numbers,
            cmap="viridis",
            colors="red",
        )
        plt.setp(plt.gca(), xlim=(0, img.shape[1]), ylim=(img.shape[0], 0))
        plt.title(f"Rejected Components (n={len(idx_bad)})")
    if figure_path is not None:
        plt.savefig(figure_path, dpi=150)

    plt.show()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def plot_traces(est, f, idx_good=None, n_plots=(5, 5), figure_path=None):

    T = est.C.shape[1]
    T_range = np.array([T // 2, T])
    t = np.linspace(*(T_range / f), T_range[1] - T_range[0])  # type: ignore

    idx_good = getattr(est, "idx_components", None) if idx_good is None else idx_good
    if idx_good is None:
        idx_good = np.arange(est.C.shape[0])
    # print(idx_good)
    idxes = np.random.choice(idx_good, size=np.prod(n_plots), replace=False)
    idxes.sort()

    fig = plt.figure(figsize=(10, 10))
    for i, n in enumerate(idxes):

        plt.subplot(*n_plots, i + 1)
        plt.plot(
            t,
            2 + est.S_dff[n, slice(*T_range)] / np.max(est.S_dff[n, :]),
            linewidth=0.5,
            color="k",
            label="S",
        )

        if est.F_dff is not None:
            plt.plot(
                t,
                1 + est.F_dff[n, slice(*T_range)] / np.max(est.F_dff[n, :]),
                linewidth=0.5,
                color="tab:green",
                label="dF/F",
            )

        max_C = np.max(est.C[n, :])
        plt.plot(
            t,
            est.C[n, slice(*T_range)] / max_C,
            linewidth=0.5,
            color="tab:blue",
            label="C",
        )
        if est.R is not None:
            plt.plot(
                t,
                (est.C[n, slice(*T_range)] + est.R[n, slice(*T_range)]) / max_C,
                linewidth=0.5,
                color="tab:orange",
                alpha=0.7,
                label="C+R",
            )

        plt.text(0.6, 0.9, f"{n=}", transform=plt.gca().transAxes, fontsize="small")

        plt.setp(plt.gca(), ylim=(-0.5, 5.0), yticks=[], xlim=(t[0], t[-1]))
        if i < (n_plots[0] - 1) * n_plots[1]:
            plt.gca().set_xticklabels([])
        else:
            plt.xlabel("time (s)")
        if i == 0:
            plt.legend(loc="upper left", fontsize="small")
        plt.gca().spines[["top", "right"]].set_visible(False)

    if figure_path is not None:
        plt.savefig(figure_path, dpi=150)

    plt.show()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def plot_key(
    ax,
    est,
    key,
    nbins=30,
    idx_good=None,
    idx_bad=None,
    flipped: bool = False,
    styles: List = list(),
):
    vals = getattr(est, key)

    idx_good = getattr(est, "idx_components", None) if idx_good is None else idx_good
    idx_bad = getattr(est, "idx_components_bad", None) if idx_bad is None else idx_bad
    val_range = np.percentile(vals[np.isfinite(vals)], [1, 99])

    for idxes, style in zip([idx_bad, idx_good], styles):
        if len(idxes) == 0:
            continue

        ax.hist(
            vals[idxes],
            bins=np.linspace(*val_range, nbins),  # type: ignore
            orientation="horizontal" if flipped else "vertical",
            **style,
        )

        inf_mask = np.isinf(vals[idxes])
        if inf_mask.any():
            inf_count = inf_mask.sum()
            inf_pos = val_range[1] + (val_range[1] - val_range[0]) * 0.1
            if flipped:
                ax.barh(
                    inf_pos,
                    inf_count,
                    height=(val_range[1] - val_range[0]) / nbins,
                    **style,
                )
            else:
                ax.bar(
                    inf_pos,
                    inf_count,
                    width=(val_range[1] - val_range[0]) / nbins,
                    **style,
                )
            ax.text(
                inf_pos * 0.95,
                inf_count / 2 if flipped else inf_count,
                "inf",
                ha="center",
                va="center",
                fontsize=8,
            )

    if flipped:
        ax.invert_xaxis()
    ax.spines[["top", "left" if flipped else "right"]].set_visible(False)


def plot_scatter(ax, est, key_x, key_y, idx_good=None, idx_bad=None):
    x_vals = getattr(est, key_x)
    y_vals = getattr(est, key_y)

    idx_good = getattr(est, "idx_components", None) if idx_good is None else idx_good
    idx_bad = getattr(est, "idx_components_bad", None) if idx_bad is None else idx_bad

    ax.scatter(
        x_vals[idx_bad],
        y_vals[idx_bad],
        color="tab:red",
        s=4,
        alpha=0.7,
        label="bad",
    )
    ax.scatter(
        x_vals[idx_good],
        y_vals[idx_good],
        color="tab:green",
        s=2,
        alpha=0.7,
        label="good",
    )
    plt.setp(ax, xlabel=key_x, ylabel=key_y)
    ax.legend()


def plot_quality(
    cnmf_obj, idx_good=None, idx_bad=None, border_thr=5.0, figure_path=None
):

    idx_good = (
        getattr(cnmf_obj.estimates, "idx_components", None)
        if idx_good is None
        else idx_good
    )
    idx_bad = (
        getattr(cnmf_obj.estimates, "idx_components_bad", None)
        if idx_bad is None
        else idx_bad
    )

    if idx_good is None:
        idx_good = np.arange(cnmf_obj.estimates.A.shape[1])
    if idx_bad is None:
        idx_bad = np.setdiff1d(
            np.arange(cnmf_obj.estimates.A.shape[1]), idx_good
        ).astype(int)

    print(f"Plotting quality for {len(idx_good)} good and {len(idx_bad)} bad neurons")

    est = cnmf_obj.estimates
    fig = plt.figure(figsize=(6, 6))
    styles = [{"color": "tab:red", "alpha": 0.7}, {"color": "tab:green", "alpha": 0.7}]

    bad_shape = est.cnn_preds < cnmf_obj.params.get("quality", "cnn_lowest")
    idx_bad_shape = np.where(bad_shape)[0]
    idx_bad_quality = np.setdiff1d(idx_bad, idx_bad_shape)
    # idx_bad = idx_quality_bad

    # ## calculating the distance to the next border
    # b_dist = np.zeros(est.A.shape[1])
    # for n in range(est.A.shape[1]):
    #     border_distance_lower = est.coordinates[n]["CoM"]
    #     border_distance_upper = est.dims[0] - est.coordinates[n]["CoM"]
    #     b_dist[n] = np.min(np.minimum(border_distance_lower, border_distance_upper))

    # bad_position = b_dist < border_thr

    # bad_additional = bad_position | bad_shape
    # idx_bad_additional = np.where(bad_additional)[0]
    # idx_bad = np.union1d(idx_bad, idx_bad_additional)
    # idx_good = np.setdiff1d(idx_good, idx_bad)

    # idx_bad_quality = np.where(bad_shape & ~bad_position)[0]
    # idx_add = idx_bad_additional
    # idx_bad_filtered = np.setdiff1d(idx_bad, idx_bad_additional)

    # print(idx_bad_filtered)

    ## plotting single quality statistics
    ax_cnn = plt.subplot(2, 2, 1)
    plot_key(
        ax_cnn, est, "cnn_preds", idx_good=idx_good, idx_bad=idx_bad, styles=styles
    )
    ax_cnn.axvline(
        cnmf_obj.params.get("quality", "cnn_lowest"), color="k", linestyle="--"
    )
    plt.setp(ax_cnn, xlabel="CNN Prediction", ylabel="Count")
    ax_cnn.text(
        0.7,
        0.9,
        f"n={bad_shape.sum()} discarded from cnn",
        transform=ax_cnn.transAxes,
        ha="center",
    )

    ax_snr = plt.subplot(2, 2, 2)
    plot_key(
        ax_snr,
        est,
        "SNR_comp",
        idx_good=idx_good,
        idx_bad=idx_bad_quality,
        styles=styles,
    )
    plt.setp(ax_snr, xlabel="SNR", ylabel="Count")

    ax_rval = plt.subplot(2, 2, 3)
    plot_key(
        ax_rval,
        est,
        "r_values",
        idx_good=idx_good,
        idx_bad=idx_bad_quality,
        flipped=True,
        styles=styles,
    )
    plt.setp(ax_rval, xlabel="Count", yticklabels=[])
    ax_rval.yaxis.tick_right()

    # ax = fig.add_subplot([0.25, 0.74, 0.15, 0.15])
    # ax.hist(b_dist[idx_bad_quality], np.logspace(0, 2.5, 51), **styles[0])
    # ax.hist(b_dist[idx_good], np.logspace(0, 2.5, 51), **styles[1])
    # ax.axvline(border_thr, color="k", linestyle="--", linewidth=0.5)
    # ax.spines[["top", "right", "left"]].set_visible(False)
    # ax.set_xscale("log")
    # plt.setp(ax, xlabel="border distance [px]", yticks=[])
    # ax.text(
    #     0.7,
    #     0.9,
    #     f"n={(b_dist<border_thr).sum()} discarded from border",
    #     transform=ax.transAxes,
    #     ha="center",
    # )

    ax = plt.subplot(2, 2, 4)
    plot_scatter(
        ax, est, "SNR_comp", "r_values", idx_good=idx_good, idx_bad=idx_bad_quality
    )
    plt.setp(
        ax,
        xlim=ax_snr.get_xlim(),
        ylim=ax_rval.get_ylim(),
        xlabel="SNR",
        ylabel="r value",
    )
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if figure_path is not None:
        plt.savefig(figure_path, dpi=150)

    plt.show()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
