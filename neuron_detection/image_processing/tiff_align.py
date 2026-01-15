
"""
tiff_align.py

Python translation of the MATLAB class `tiff_align` (tiff_align.m).

Implements:
- Recursive rigid x-y alignment by maximizing cross-correlation
- Non-rigid warping via a Lucas–Kanade style optimization with 1D B-spline basis
- Frame-by-frame export of aligned stacks to TIFF files

Dependencies:
    numpy
    tifffile
    scipy

Example:
    from tiff_align import TiffAlign

    # Single channel, one file or directory containing *.tif
    TiffAlign(path_in="raw/", path_out="aligned/")

    # Multi-channel: rows = file number, cols = channel
    files = [
        ["ch1_file1.tif", "ch2_file1.tif"],
        ["ch1_file2.tif", "ch2_file2.tif"],
    ]
    TiffAlign(path_in=files, path_out="aligned/")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile

from scipy import sparse
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import spsolve


Array = np.ndarray
PathLike = Union[str, Path]


def _as_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _list_tifs_in_dir(d: Path) -> List[Path]:
    # mimic MATLAB dir(pathcat(pathIn,'*.tif')) behaviour; include .tif/.tiff
    files = sorted(list(d.glob("*.tif")) + list(d.glob("*.tiff")))
    return files


def _nanquantile01(x: Array) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.quantile(x, 0.01))


def _shift_with_nan(img: Array, shift_y: int, shift_x: int) -> Array:
    """
    Shift image by (shift_y, shift_x) filling empty pixels with NaN.
    Positive shift_y moves content down (towards increasing row index),
    positive shift_x moves content right (towards increasing col index).
    """
    if shift_y == 0 and shift_x == 0:
        return img
    h, w = img.shape
    out = np.full((h, w), np.nan, dtype=float)

    y0_dst = max(0, shift_y)
    y1_dst = h + min(0, shift_y)
    x0_dst = max(0, shift_x)
    x1_dst = w + min(0, shift_x)

    y0_src = max(0, -shift_y)
    y1_src = h - max(0, shift_y)
    x0_src = max(0, -shift_x)
    x1_src = w - max(0, shift_x)

    if y1_dst > y0_dst and x1_dst > x0_dst:
        out[y0_dst:y1_dst, x0_dst:x1_dst] = img[y0_src:y1_src, x0_src:x1_src]
    return out


def constructH(Hd: Array, ns: int) -> sparse.csr_matrix:
    """
    Helper for Lucas–Kanade: construct tri-diagonal sparse matrix from Hd vector.
    Translation of MATLAB constructH(Hd, ns).
    """
    Hd = np.asarray(Hd).reshape(-1)
    H2d1 = Hd[:ns]
    H2d2 = np.concatenate([Hd[ns:], np.array([0.0])])
    H2d3 = np.concatenate([np.array([0.0]), Hd[ns:]])

    # diags: lower (-1), main (0), upper (1)
    lower = H2d2[: ns - 1]
    main = H2d1[:ns]
    upper = H2d3[1:ns]
    return sparse.diags(
        diagonals=[lower, main, upper],
        offsets=[-1, 0, 1],
        shape=(ns, ns),
        format="csr",
        dtype=float,
    )


def max_corr(img_input: Array, img_goal: Array, max_shift: Optional[int]) -> Tuple[int, int]:
    """
    Calculate rigid x-y shift by maximizing cross-correlation (FFT).

    Returns: (shift_y, shift_x) as integers.
    """
    img_input = np.asarray(img_input, dtype=float)
    img_goal = np.asarray(img_goal, dtype=float)

    # remove NaN edges using center row/col as reference (MATLAB logic)
    cy = img_input.shape[0] // 2
    cx = img_input.shape[1] // 2
    nan_y = np.isnan(img_input[:, cx] * img_goal[:, cx])
    nan_x = np.isnan(img_input[cy, :] * img_goal[cy, :])

    good_y = np.where(~nan_y)[0]
    good_x = np.where(~nan_x)[0]
    if good_y.size == 0 or good_x.size == 0:
        return 0, 0

    img_input = img_input[np.ix_(good_y, good_x)]
    img_goal = img_goal[np.ix_(good_y, good_x)]

    # square crop around center
    N = int(np.floor(min(img_input.shape) / 2))
    if N <= 1:
        return 0, 0

    y0 = img_input.shape[0] // 2 - N
    y1 = img_input.shape[0] // 2 + N
    x0 = img_input.shape[1] // 2 - N
    x1 = img_input.shape[1] // 2 + N

    img_input = img_input[y0:y1, x0:x1]
    img_goal = img_goal[y0:y1, x0:x1]

    # cross-correlation via FFT
    # MATLAB used fft2(img_input).*fft2(rot90(img_goal,2)) then ifft2, fftshift
    F1 = np.fft.fft2(img_input)
    F2 = np.fft.fft2(img_goal)
    C = np.fft.fftshift(np.real(np.fft.ifft2(F1 * np.conj(F2))))

    if max_shift is not None and not (isinstance(max_shift, float) and np.isnan(max_shift)):
        yy, xx = np.meshgrid(np.arange(2 * N) - N + 1, np.arange(2 * N) - N + 1, indexing="ij")
        mask = (xx * xx + yy * yy) < (max_shift * max_shift)
        C = C * mask

    # find maximum (median if multiple)
    maxv = np.nanmax(C)
    inds = np.argwhere(C == maxv)
    if inds.size == 0:
        return 0, 0
    ind_y0 = int(np.median(inds[:, 0]))
    ind_x0 = int(np.median(inds[:, 1]))

    # Translate MATLAB:
    # shift_y = N - ind_y (with ind_y 1-based). For 0-based -> shift = (N-1) - ind0
    shift_y = (N - 1) - ind_y0
    shift_x = (N - 1) - ind_x0
    return int(shift_y), int(shift_x)


@dataclass
class _XYRec:
    I: Array  # mean image
    n: int    # number of frames
    T: Array  # transforms (n_frames, 2)


def xy_align(obj: "TiffAlign", stacks_to_do: Sequence[int], max_shift: Optional[int]) -> _XYRec:
    """
    Recursive alignment (Scanbox-style).
    stacks_to_do: 1-based global stack indices (as in MATLAB).
    """
    stacks_to_do = list(stacks_to_do)
    if len(stacks_to_do) == 1:
        I = obj.read_stack(stacks_to_do[0], channel_to_extract=1)
        return _XYRec(I=I, n=1, T=np.array([[0, 0]], dtype=int))

    mid = len(stacks_to_do) // 2
    r_input = xy_align(obj, stacks_to_do[:mid], max_shift)
    r_goal = xy_align(obj, stacks_to_do[mid:], max_shift)

    shift_y, shift_x = max_corr(r_input.I, r_goal.I, max_shift)

    if abs(shift_y) + abs(shift_x) > 0:
        r_input.I = _shift_with_nan(r_input.I, shift_y, shift_x)

    I = (r_input.n * r_input.I + r_goal.n * r_goal.I) / (r_input.n + r_goal.n)
    T_input = r_input.T + np.array([shift_y, shift_x], dtype=int)
    T = np.vstack([T_input, r_goal.T])
    return _XYRec(I=I, n=r_input.n + r_goal.n, T=T.astype(int))


class TiffAlign:
    """
    Align frames in TIFF files (rigid shifts + Lucas–Kanade non-rigid warp).
    """

    def __init__(
        self,
        path_in: Optional[Union[PathLike, Sequence[Sequence[PathLike]]]] = None,
        path_out: PathLike = "aligned",
        *,
        max_shift_all: int = 10,
        n_basis: int = 16,
        mean_samples: int = 5,
        max_iter: int = 25,
        dcorr_goal: float = 5e-4,
        damping: float = 1.0,
        max_shift_block: int = 10,
        bigtiff: bool = True,
    ) -> None:
        self.path_out = _as_path(path_out)
        self.path_out.mkdir(parents=True, exist_ok=True)

        # Parameters
        self.max_shift_all = int(max_shift_all)
        self.n_basis = int(n_basis)
        self.mean_samples = int(mean_samples)
        self.max_iter = int(max_iter)
        self.dcorr_goal = float(dcorr_goal)
        self.damping = float(damping)
        self.max_shift_block = int(max_shift_block)
        self._bigtiff = bool(bigtiff)

        # State
        self.tiff_info: List[List[Dict[str, Any]]] = []
        self.width: int = 0
        self.height: int = 0
        self.num_channels: int = 0
        self.num_stacks: int = 0
        self.stack_to_file: Array = np.zeros((0, 3), dtype=int)  # [file_number(1-based), stack_ini, stack_end]
        self.xy_shift: Array = np.zeros((0, 2), dtype=int)

        # Lucas–Kanade precomputed
        self.Bf: Optional[Array] = None
        self.mask: Optional[Array] = None
        self.xi: Optional[Array] = None
        self.yi: Optional[Array] = None

        # Export
        self._writers: List[List[tifffile.TiffWriter]] = []
        self._out_dtype: Optional[np.dtype] = None

        file_names = self.make_file_structure(path_in)

        self._open_and_validate(file_names)

        self.xy_shift = np.zeros((self.num_stacks, 2), dtype=int)

        # Process immediately (matches MATLAB constructor behavior)
        self.process_tiff()

    # ---------------- I/O and file structure ----------------

    def make_file_structure(
        self, path_in: Optional[Union[PathLike, Sequence[Sequence[PathLike]]]]
    ) -> List[List[Path]]:
        """
        Construct list of file names.
        - None: raises (MATLAB opened a GUI)
        - str/Path:
            - *.tif file => one file, one channel
            - directory => all *.tif/*.tiff, one channel
        - 2D sequence: rows = file number, columns = channel
        """
        if path_in is None:
            raise ValueError(
                "path_in must be provided in Python (MATLAB version could open a GUI). "
                "Provide a .tif file, a directory, or a 2D list [files][channels]."
            )

        if isinstance(path_in, (str, Path)):
            p = _as_path(path_in)
            if p.is_dir():
                files = _list_tifs_in_dir(p)
                if not files:
                    raise FileNotFoundError(f"No TIFF files found in directory: {p}")
                return [[f] for f in files]
            if p.is_file():
                return [[p]]
            raise FileNotFoundError(f"Input path does not exist: {p}")

        # 2D sequence
        matrix: List[List[Path]] = []
        for row in path_in:
            matrix.append([_as_path(x) for x in row])
        if not matrix or not matrix[0]:
            raise ValueError("Empty file structure provided.")
        # validate rectangular
        ncols = len(matrix[0])
        if any(len(r) != ncols for r in matrix):
            raise ValueError("All rows in file structure must have the same number of channels (columns).")
        return matrix

    def _open_and_validate(self, file_names: List[List[Path]]) -> None:
        """
        Open all TIFF files, read metadata, compute mapping stack->file.
        """
        # validate same dimensions across all files
        first = tifffile.TiffFile(str(file_names[0][0]))
        page0 = first.pages[0]
        self.height, self.width = int(page0.imagelength), int(page0.imagewidth)
        first.close()

        for row in file_names:
            for f in row:
                with tifffile.TiffFile(str(f)) as tf:
                    p = tf.pages[0]
                    h, w = int(p.imagelength), int(p.imagewidth)
                    if (h != self.height) or (w != self.width):
                        raise ValueError("Not all files have the same x-y dimensions.")

        # open all and collect stacks
        self.tiff_info = []
        for row in file_names:
            row_info: List[Dict[str, Any]] = []
            for f in row:
                tf = tifffile.TiffFile(str(f))
                stacks = len(tf.pages)
                row_info.append({"tif": tf, "filename": f, "stacks": stacks})
            self.tiff_info.append(row_info)

        self.num_channels = len(self.tiff_info[0])
        # ensure each row has same stacks for every channel (MATLAB check intent)
        for r in self.tiff_info:
            stacks0 = r[0]["stacks"]
            for c in r[1:]:
                if c["stacks"] != stacks0:
                    raise ValueError("Not all channels have the same number of stacks within a file group.")

        # mapping: for each file group (row), stacks are concatenated
        self.num_stacks = 0
        mapping = []
        for i, r in enumerate(self.tiff_info, start=1):  # file_number 1-based
            stacks = int(r[0]["stacks"])
            ini = self.num_stacks + 1
            end = self.num_stacks + stacks
            mapping.append([i, ini, end])
            self.num_stacks += stacks
        self.stack_to_file = np.asarray(mapping, dtype=int)

        # output dtype matches input dtype of first page
        with tifffile.TiffFile(str(self.tiff_info[0][0]["filename"])) as tf:
            self._out_dtype = tf.pages[0].dtype

    def close(self) -> None:
        # Close input files
        for row in self.tiff_info:
            for info in row:
                try:
                    info["tif"].close()
                except Exception:
                    pass

    # ---------------- Frame reading ----------------

    def read_stack(self, stack_global: int, channel_to_extract: int = 1) -> Array:
        """
        Read 1-based stack index from concatenated stacks across file groups.
        Apply previously computed x-y shifts, filling empty pixels with NaN.
        """
        if channel_to_extract < 1 or channel_to_extract > self.num_channels:
            raise ValueError(f"channel_to_extract must be in 1..{self.num_channels}")

        # locate file group
        row_idx = np.where((stack_global >= self.stack_to_file[:, 1]) & (stack_global <= self.stack_to_file[:, 2]))[0]
        if row_idx.size == 0:
            raise IndexError("Required stack is not available.")
        file_number0 = int(row_idx[0])  # 0-based row index

        stack_number = stack_global - (int(self.stack_to_file[file_number0, 1]) - 1)  # 1-based within file
        info = self.tiff_info[file_number0][channel_to_extract - 1]
        page = info["tif"].pages[stack_number - 1]
        img = page.asarray()

        # if multi-dimensional, use only first plane
        if img.ndim > 2:
            img = img[:, :, 0]
        img = img.astype(float)

        sy, sx = map(int, self.xy_shift[stack_global - 1])
        return _shift_with_nan(img, sy, sx)

    # ---------------- Alignment steps ----------------

    def calculate_xy_shift(self) -> None:
        self.xy_shift = np.zeros((self.num_stacks, 2), dtype=int)
        r = xy_align(self, list(range(1, self.num_stacks + 1)), self.max_shift_all)
        self.xy_shift = r.T.astype(int)

    def prepare_LK(self) -> None:
        """
        Precompute basis functions, correlation mask and coordinate grids.
        """
        assert self.height % self.n_basis == 0, (
            "Y dimension must be a multiple of n_basis. "
            "Change n_basis to a divisor of image height."
        )

        # Linear B-splines (tent-shaped)
        Bf = np.zeros((self.height, self.n_basis + 1), dtype=float)

        knots = np.linspace(0, self.height, self.n_basis + 1)
        knots = np.concatenate([[knots[0] - (knots[1] - knots[0])], knots, [knots[-1] + (knots[-1] - knots[-2])]])
        delta = 1e-10
        x = np.arange(1, self.height + 1, dtype=float)  # 1-based like MATLAB

        for ii in range(self.n_basis + 1):
            up = ((x - knots[ii]) / (delta + knots[ii + 1] - knots[ii])) * ((x < knots[ii + 1]) & (x >= knots[ii]))
            down = ((knots[ii + 2] - x) / (delta + knots[ii + 2] - knots[ii + 1])) * (
                (x < knots[ii + 2]) & (x >= knots[ii + 1])
            )
            Bf[:, ii] = up + down

        self.Bf = Bf

        # Mask for blockwise correlation initiation
        blocksize = self.height // self.n_basis
        mx, my = np.meshgrid(np.arange(1, self.width + 1), np.arange(1, blocksize + 1))  # 1-based
        center_xy = np.array([self.width // 2 + 1, blocksize // 2 + 1])
        mask = (mx - center_xy[0]) ** 2 + (my - center_xy[1]) ** 2 < (self.max_shift_block ** 2)
        self.mask = mask.astype(float)

        # coordinate grids (0-based for scipy map_coordinates)
        xi, yi = np.meshgrid(np.arange(self.width, dtype=float), np.arange(self.height, dtype=float))
        self.xi = xi
        self.yi = yi

    def _warp(self, img: Array, dpx: Array, dpy: Array, fill_value: float) -> Array:
        """
        Apply vertical basis-defined displacement in x and y to an image (bilinear).
        """
        assert self.Bf is not None and self.xi is not None and self.yi is not None
        disp_x = (self.Bf @ dpx).reshape(-1, 1)  # Hx1
        disp_y = (self.Bf @ dpy).reshape(-1, 1)
        Dx = np.repeat(disp_x, self.width, axis=1)
        Dy = np.repeat(disp_y, self.width, axis=1)

        coords_y = (self.yi + Dy).ravel()
        coords_x = (self.xi + Dx).ravel()
        coords = np.vstack([coords_y, coords_x])

        Id = map_coordinates(img, coords, order=1, mode="constant", cval=np.nan).reshape(self.height, self.width)
        Id[np.isnan(Id)] = fill_value
        return Id

    def LK_align(self, img_I: Array, img_T: Array) -> Tuple[Array, Array, Array]:
        """
        Lucas–Kanade non-rigid alignment.
        Returns (Id, dpx, dpy):
            Id: warped image
            dpx/dpy: basis weights (n_basis+1,)
        """
        if self.Bf is None:
            self.prepare_LK()
        assert self.Bf is not None and self.mask is not None

        img_T = img_T.astype(float).copy()
        img_I = img_I.astype(float).copy()

        # normalize template for correlation
        T_norm = img_T.ravel() - np.mean(img_T)
        T_norm = T_norm / (np.sqrt(np.sum(T_norm ** 2)) + 1e-12)

        fill_T = _nanquantile01(img_T)
        img_T[np.isnan(img_T)] = fill_T

        fill_I = _nanquantile01(img_I)
        img_I[np.isnan(img_I)] = fill_I

        # ---- Blockwise correlation to initialize dpx/dpy
        blocksize = self.height // self.n_basis
        dpx0 = np.zeros((self.n_basis,), dtype=float)
        dpy0 = np.zeros((self.n_basis,), dtype=float)

        for ii in range(self.n_basis):
            block = slice(ii * blocksize, (ii + 1) * blocksize)

            T_block = img_T[block, :]
            I_block = img_I[block, :]

            # subtract column means (nanmean in MATLAB, but NaNs were filled already)
            T_ = T_block - np.mean(T_block, axis=0, keepdims=True)
            I_ = I_block - np.mean(I_block, axis=0, keepdims=True)

            C = self.mask * np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(I_) * np.conj(np.fft.fft2(T_)))))

            maxv = np.max(C)
            inds = np.argwhere(C == maxv)
            if inds.size == 0:
                continue
            ind_y0 = int(np.median(inds[:, 0]))
            ind_x0 = int(np.median(inds[:, 1]))

            center_x0 = self.width // 2
            center_y0 = blocksize // 2
            dpx0[ii] = ind_x0 - center_x0
            dpy0[ii] = ind_y0 - center_y0

        # interpolate to knots (n_basis+1)
        dpx = np.concatenate([[dpx0[0]], (dpx0[:-1] + dpx0[1:]) / 2.0, [dpx0[-1]]]).astype(float)
        dpy = np.concatenate([[dpy0[0]], (dpy0[:-1] + dpy0[1:]) / 2.0, [dpy0[-1]]]).astype(float)

        # ---- Lucas–Kanade optimization
        lambda_ = 1e-4 * (np.median(img_T) ** 2)
        theI = sparse.identity(self.n_basis + 1, format="csr") * lambda_

        allBs = np.hstack([self.Bf ** 2, self.Bf[:, :-1] * self.Bf[:, 1:]])  # H x (2*n_basis+1)

        Id = img_I.copy()
        Id_norm = Id.ravel() - np.mean(Id)
        Id_norm = Id_norm / (np.sqrt(np.sum(Id_norm ** 2)) + 1e-12)
        corr_old = float(Id_norm @ T_norm)

        for _ in range(self.max_iter):
            # gradient (central difference with edge replication)
            dTx = (Id[:, np.r_[0, 2:self.width, self.width - 1]] - Id[:, np.r_[0, 0:self.width - 2, self.width - 1]]) / 2.0
            dTy = (Id[np.r_[0, 2:self.height, self.height - 1], :] - Id[np.r_[0, 0:self.height - 2, self.height - 1], :]) / 2.0

            del_ = img_T - Id

            row_sum_x = np.sum(del_ * dTx, axis=1)  # H
            row_sum_y = np.sum(del_ * dTy, axis=1)

            gx = self.Bf.T @ row_sum_x
            gy = self.Bf.T @ row_sum_y

            Hd_x = allBs.T @ np.sum(dTx ** 2, axis=1)
            Hd_y = allBs.T @ np.sum(dTy ** 2, axis=1)

            Hx = constructH(Hd_x, self.n_basis + 1) + theI
            Hy = constructH(Hd_y, self.n_basis + 1) + theI

            dpx_ = spsolve(Hx, gx)
            dpy_ = spsolve(Hy, gy)

            dpx = dpx + self.damping * dpx_
            dpy = dpy + self.damping * dpy_

            Id = self._warp(img_I, dpx, dpy, fill_I)

            Id_norm = Id.ravel() - np.mean(Id)
            Id_norm = Id_norm / (np.sqrt(np.sum(Id_norm ** 2)) + 1e-12)
            corr_new = float(Id_norm @ T_norm)

            if (corr_new - corr_old) < self.dcorr_goal:
                break
            corr_old = corr_new

        return Id, dpx, dpy

    # ---------------- Export ----------------

    def export_prepare(self) -> None:
        """
        Create output TIFF writers matching the input file group layout.
        """
        assert self._out_dtype is not None
        self._writers = []
        for row in self.tiff_info:
            out_row = []
            for info in row:
                f_in: Path = info["filename"]
                out_path = self.path_out / f_in.name
                if out_path.exists():
                    out_path.unlink()
                out_row.append(
                    tifffile.TiffWriter(str(out_path), bigtiff=self._bigtiff)
                )
            self._writers.append(out_row)

    def export_add(self, Im: Array, stack_num: int, channel_num: int) -> None:
        """
        Write a processed frame to the appropriate output TIFF.
        stack_num and channel_num are 1-based.
        """
        if self._out_dtype is None:
            raise RuntimeError("export_prepare must be called before export_add.")

        # locate output file group
        row_idx = np.where((stack_num >= self.stack_to_file[:, 1]) & (stack_num <= self.stack_to_file[:, 2]))[0]
        if row_idx.size == 0:
            raise IndexError("Required stack is not available.")
        file_number0 = int(row_idx[0])

        # Cast to input dtype (MATLAB trfun)
        out = np.asarray(Im)
        if np.issubdtype(self._out_dtype, np.integer):
            # clip before casting to avoid wrap-around
            info = np.iinfo(self._out_dtype)
            out = np.clip(out, info.min, info.max)
        out = out.astype(self._out_dtype, copy=False)

        writer = self._writers[file_number0][channel_num - 1]
        writer.write(out)

    def export_close(self) -> None:
        for row in self._writers:
            for w in row:
                try:
                    w.close()
                except Exception:
                    pass

    # ---------------- Main loop ----------------

    def process_tiff(self) -> None:
        """
        Run full pipeline:
        1) Rigid x-y alignment
        2) Rolling mean template
        3) Lucas–Kanade warp per frame, apply warp to all channels
        4) Export to output TIFFs
        """
        if self.height % self.n_basis != 0:
            raise ValueError("Image height must be a multiple of n_basis. Set n_basis accordingly.")

        # 1) x-y shifts
        self.calculate_xy_shift()

        # 2) output writers
        self.export_prepare()

        # 3) starting rolling mean
        rolling_template = np.zeros((self.height, self.width), dtype=float)
        frame_count = np.zeros((self.height, self.width), dtype=float)

        n0 = min(self.mean_samples, self.num_stacks)
        for frame_ii in range(1, n0 + 1):
            S = self.read_stack(frame_ii, 1)
            val = np.isfinite(S)
            rolling_template[val] += S[val]
            frame_count[val] += 1

        # 4) frame-by-frame LK + export
        for frame_ii in range(1, self.num_stacks + 1):
            # update rolling template window
            template_range_lo = frame_ii - self.mean_samples
            template_range_hi = frame_ii + self.mean_samples

            if template_range_lo > 0:
                S = self.read_stack(template_range_lo, 1)
                val = np.isfinite(S)
                rolling_template[val] -= S[val]
                frame_count[val] -= 1

            if template_range_hi <= self.num_stacks:
                S = self.read_stack(template_range_hi, 1)
                val = np.isfinite(S)
                rolling_template[val] += S[val]
                frame_count[val] += 1

            # avoid division by zero
            denom = frame_count.copy()
            denom[denom == 0] = np.nan
            template = rolling_template / denom
            template[np.isnan(template)] = _nanquantile01(template)

            # align channel 1 and get warp params
            img1 = self.read_stack(frame_ii, 1)
            Im, dpx, dpy = self.LK_align(img1, template)
            self.export_add(Im, frame_ii, 1)

            # other channels: reuse dpx/dpy
            for ch in range(2, self.num_channels + 1):
                img = self.read_stack(frame_ii, ch)
                fill = _nanquantile01(img)
                img[np.isnan(img)] = fill
                Im_ch = self._warp(img, dpx, dpy, fill)
                self.export_add(Im_ch, frame_ii, ch)

        # 5) close files
        self.export_close()
        self.close()
