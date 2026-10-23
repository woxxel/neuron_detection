from tifffile import TiffFile, memmap
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import pickle

from .diagnostics_forward_model import (
    fit_joint_resonant_covariance,
    damped_sinc_kernel,
    fit_damped_sinc,
    phase_collapse_data,
    estimate_phase_kernel,
)
from .diagnostics_correlation import diagnose_resonant_scan

from .correction_fix_convolution import (
    make_equalized_resonant_resampler,
)

from .build_alignment_template import (
    make_similarity_features,
    recurrent_similarity_scores,
    align_template_candidates,
    split_half_theta_coherence,
)


def run_diagnostics(movie, save_path: str | Path):

    path_diagnostics = Path(save_path)
    path_diagnostics.mkdir(parents=True, exist_ok=True)

    fname = f"diagnostics_and_fit.pkl"
    path_file = path_diagnostics / fname
    print(
        f"Running diagnostics and fitting joint resonant covariance for save path: {path_file}..."
    )
    if path_file.exists():
        # create an empty file or handle accordingly
        with open(path_file, "rb") as f:
            results = pickle.load(f)
    else:
        diagnostics = diagnose_resonant_scan(movie)
        joint_fit = fit_joint_resonant_covariance(
            diagnostics["corr_even"],
            diagnostics["corr_odd"],
        )
        results = {"diagnostics": diagnostics, "joint_fit": joint_fit}
        with open(path_file, "wb") as f:
            pickle.dump(results, f)

    return results


def resample_and_equalize(diagnostics, fit, n=256, **kwargs):
    """
    TODO:
    * include diagnostic plots for checking proper bandwidth, etc
    """
    collapsed = phase_collapse_data(
        diagnostics["corr_even"],
        diagnostics["corr_odd"],
        x0=fit["x0"],
        A=fit["A"],
    )
    damped_fit = fit_damped_sinc(collapsed)

    theta_model = np.linspace(
        0,
        max(0.20, 6 * damped_fit["sigma"]),
        10000,
    )

    K_model = damped_sinc_kernel(
        theta_model,
        damped_fit["alpha"],
        damped_fit["tau"],
        damped_fit["sigma"],
    )

    freq_model = np.linspace(0, 80, 2000)
    psd_model = np.empty_like(freq_model)

    for i, f in enumerate(freq_model):
        psd_model[i] = 2 * np.trapezoid(
            K_model * np.cos(2 * np.pi * f * theta_model), theta_model
        )

    psd_model /= psd_model[0]

    H_model = np.sqrt(np.maximum(psd_model, 0))

    resampler_eq = make_equalized_resonant_resampler(
        X=512,
        x0=fit["x0"],
        A=fit["A"],
        n_out=n,
        freq_H=freq_model,
        H_old=H_model,
        regularization=kwargs.get("regularization", 0.15),
        max_gain=kwargs.get("max_gain", 1.7),
        pass_fraction=kwargs.get("pass_fraction", 0.85),
        stop_fraction=kwargs.get("stop_fraction", 0.98),
    )
    return resampler_eq


def refine_template(movie, candidate_frames, template_in):

    aligned, metrics = align_template_candidates(
        movie,
        candidate_frames,
        template_in,
    )

    peak = np.array([m["peak"] for m in metrics])

    margin = np.array([m["peak_margin"] for m in metrics])

    good = (peak >= np.percentile(peak, 25)) & (margin >= np.percentile(margin, 25))

    template_out = np.mean(
        aligned[good],
        axis=0,
    )
    return template_out, aligned, good


def build_alignment_template(movie, N_iter=2, k=200, plot=False, **kwargs):
    features, frame_ids = make_similarity_features(
        movie,  # e.g. your stripe-free 256 movie
        temporal_stride=kwargs.get("temporal_stride", 2),
    )

    scores, neighbors = recurrent_similarity_scores(
        features,
        frame_ids,
        k=k,
        exclude_frames=kwargs.get("exclude_frames", 45),
    )

    seed_feature_idx = np.argmax(scores)
    seed_frame = frame_ids[seed_feature_idx]

    candidate_frames = frame_ids[neighbors[seed_feature_idx]]

    print("seed frame:", seed_frame)
    print("similarity score:", scores[seed_feature_idx])

    template = movie[seed_frame]

    candidate_frames = np.concatenate(
        [
            [seed_frame],
            candidate_frames[:200],
        ]
    )

    if plot:
        fig, axes = plt.subplots(1, N_iter + 1, figsize=(12, 6))
        im = axes[0].imshow(
            template, origin="lower", aspect="auto", cmap="gray", clim=(0.1, 0.8)
        )
        plt.colorbar(im, ax=axes[0])

    for n in range(N_iter):
        template, aligned, good = refine_template(movie, candidate_frames, template)

        if plot:
            im = axes[n + 1].imshow(
                template, origin="lower", aspect="auto", cmap="gray", clim=(0.1, 0.8)
            )

    return template, aligned[good]


def check_coherence(aligned, joint_fit):

    freq_theta, coh_theta, A_half, B_half, PA, PB = split_half_theta_coherence(
        aligned,
        x_positions=np.linspace(0, 511, aligned.shape[-1]),
        x0=joint_fit["x0"],
        A=joint_fit["A"],
    )

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    axes[0].plot(
        freq_theta,
        coh_theta,
    )
    axes[0].axvline(32, ls="--")
    axes[0].axvline(40, ls=":")

    axes[0].set_xlim(0, 70)
    axes[0].set_ylim(0, 1)

    axes[0].set_xlabel("phase frequency [cycles / rad]")
    axes[0].set_ylabel("split-half coherence")

    axes[1].semilogy(freq_theta, PA, label="half A")
    axes[1].semilogy(freq_theta, PB, label="half B")
    axes[1].set_ylabel("power")
    axes[1].set_xlabel("phase frequency [cycles/rad]")
    axes[1].legend()

    axes[2].imshow(A_half, aspect="auto", origin="lower")
    axes[3].imshow(B_half, aspect="auto", origin="lower")

    plt.show()


### =============================================================== ###
### ===================== HELPER FUNCTIONS ======================== ###
### =============================================================== ###


def save_data_as_tif(out_path, data, normalize=True):

    dims = data.shape
    T = dims[0]
    dtype = data.dtype

    normalize = True
    normalize = normalize and (not str(dtype).startswith("float"))
    max_val = np.iinfo(dtype).max if normalize else None

    def get_frame(t, normalize_by=None):
        return data[t, ...] / normalize_by if normalize_by else data[t, ...]

    memmap_image = memmap(out_path, shape=dims, dtype=dtype)

    for t in tqdm(range(T)):
        memmap_image[t, ...] = get_frame(t, normalize_by=max_val if normalize else None)

        # commit changes to file every so often to avoid memory overuse
        if not ((t + 1) % 2000) or (t + 1) == T:
            memmap_image.flush()

    del memmap_image
