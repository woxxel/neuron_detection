# from __future__ import annotations

import os, shutil, copy
from pathlib import Path
from typing import Optional

from neuron_detection.preprocessing.file_structures.make_tif_stack import make_tif_stack
from neuron_detection.image_processing.motion_correct import motion_correct
from neuron_detection.image_processing.neuron_detection import neuron_detection

from neuron_detection.utils.parameters import CaImAn as CaImAn_params
import argparse


def run_detection(
    source: str | Path,
    destination: Optional[str] = None,
    tmp: Optional[str] = None,
    prefix: str = "results_CaImAn",
    suffix: str = "",
    n_processes: int = 12,
    save_type: str = "hdf5",
    **kwargs,
):
    """
    runs the full CaImAn neuron detection pipeline
    assumes source to provide the full path to the data (either to file or folder with images)
    """
    print("in run_detection with source:", source, flush=True)
    path_source = Path(source)
    fname_stem = Path(source).stem

    # print(f"USING FNAME STEM {fname_stem} TO GENERATE OUTPUT FILE NAMES - VIA SUFFIX!")
    path_destination = Path(destination) if destination else path_source.parent
    os.makedirs(path_destination, exist_ok=True)

    path_tmp = (
        Path(
            os.environ.get("TMPDIR", tmp if tmp is not None else str(path_destination))
        )
        / "tmp"
    )

    os.environ["CAIMAN_DATA"] = str(Path.home() / "caiman_data")
    os.environ["CAIMAN_TEMP"] = str(path_tmp.resolve())
    print("path tmp:", path_tmp, flush=True)

    os.makedirs(path_tmp, exist_ok=True)

    ## ensures that the file is in tif-stack format,
    ## and that it is available at path_tmp location for faster access
    path_to_stack = ensure_stack(path_source, path_tmp)

    ## change parameters according to input
    params = copy.deepcopy(CaImAn_params)
    params.update(kwargs)
    print("using parameters:")
    for key, val in params.items():
        print(f"  {key}: {val}")

    path_to_motion_correct = motion_correct(
        path_to_stack,
        params,
        path_destination=path_destination,
        fname_stem=fname_stem,
        n_processes=4,
        plot=True,
    )
    print("path to motion_correct:", path_to_motion_correct)

    path_to_neuron_detection = neuron_detection(
        path_to_motion_correct,
        params,
        path_destination=path_destination,
        prefix=prefix,
        fname_stem=f"{fname_stem}{suffix}",
        save_type=save_type,
        n_processes=n_processes,
        plot=True,
    )
    print("path to neuron_detection:", path_to_neuron_detection)

    ## need to clean up after!
    if path_tmp.is_dir() and str(path_tmp) != str(path_destination):
        print("cleaning up temporary files...")
        shutil.rmtree(path_tmp)

    print("Finished processing!")


def ensure_stack(path_source: str | Path, path_tmp: str | Path) -> str:
    """
    ensures that a memmapped tif-stack is available for further processing

    checks how images are present:
        1. as .tif file
        2. as .h5 file
        3. as image files in source folder
    """
    path_source = Path(path_source)
    path_tmp = Path(path_tmp)

    path_to_stack = Path(path_source)
    if path_source.suffix.startswith(".tif"):
        if path_tmp != path_source.parent:
            path_to_stack = path_tmp / path_source.name
            print("copying ", path_source.name, " to ", path_to_stack)
            shutil.copy(path_source, path_to_stack)

    elif path_source.suffix == ".h5":
        path_to_stack = make_tif_stack(
            path_source, path_tmp, dtype="float16", clean_after_stacking=False
        )
    elif path_source.is_dir():
        print(
            "provided folder path as source - searching for .tif files inside and stacking them into single tif-stack for further processing"
        )
        tifFiles = [
            file for file in path_source.iterdir() if file.suffix.startswith(".tif")
        ]
        if len(tifFiles):
            path_to_stack = make_tif_stack(
                path_source, path_tmp, dtype="float16", clean_after_stacking=False
            )
    else:
        assert False, "No image files found in source!"

    return str(path_to_stack)


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="neuron-detection", description="neuron detection pipeline using CaImAn"
    )

    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--destination", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="results_CaImAn")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--n_processes", type=int, default=12)
    parser.add_argument("--save_type", type=str, default="hdf5")
    parser.add_argument("--fr", type=float, default=15.0)
    parser.add_argument("--decay_time", type=float, default=0.47)
    parser.add_argument("--dxy", type=float, default=512.0 / 520)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(args)
    run_detection(**args.__dict__)

    return 0
