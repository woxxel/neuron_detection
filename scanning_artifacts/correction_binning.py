import numpy as np


def temporal_bin_movie(
    movie,
    bin_size,
    method="mean",
    drop_remainder=True,
):
    """
    Non-overlapping temporal binning of a movie.

    Parameters
    ----------
    movie : ndarray, shape (T, Y, X)
        Input movie.

    bin_size : int
        Number of consecutive frames per temporal bin.

    method : {"mean", "median"}
        Aggregation within each bin.

    drop_remainder : bool
        If True, discard trailing frames that do not fill a complete bin.
        If False, keep one final shorter bin.

    Returns
    -------
    binned : ndarray
        Binned movie, shape approximately (T / bin_size, Y, X).

    frame_centers : ndarray
        Time coordinate of each binned frame expressed in original-frame
        units. Useful later for interpolating shifts back to all frames.

    bin_edges : ndarray
        Start/end original-frame indices for each bin.
        Shape (n_bins, 2), with end exclusive.
    """

    movie = np.asarray(movie)

    if movie.ndim != 3:
        raise ValueError("movie must have shape (T, Y, X)")

    if bin_size < 1:
        raise ValueError("bin_size must be >= 1")

    T, Y, X = movie.shape

    if drop_remainder:
        n_bins = T // bin_size
    else:
        n_bins = int(np.ceil(T / bin_size))

    binned = []
    centers = []
    edges = []

    for k in range(n_bins):

        start = k * bin_size
        stop = min(start + bin_size, T)

        block = movie[start:stop]

        if method == "mean":
            frame = np.mean(
                block,
                axis=0,
                dtype=np.float64,
            )

        elif method == "median":
            frame = np.median(
                block,
                axis=0,
            )

        else:
            raise ValueError("method must be 'mean' or 'median'")

        binned.append(frame)

        # Center in original frame coordinates.
        centers.append(0.5 * (start + stop - 1))

        edges.append((start, stop))

    binned = np.stack(binned, axis=0)

    # Usually float32 is plenty for Ca imaging.
    binned = binned.astype(
        np.float32,
        copy=False,
    )

    return (
        binned,
        np.asarray(centers),
        np.asarray(edges),
    )
