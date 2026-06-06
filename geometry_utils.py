"""Geometry helpers for the trajectory shape-distance metrics.

These utilities support the RANSAC-based shape-distance metrics in
``shape_distance_metrics.py``. They operate on 2D trajectories represented as
``(N, 2)`` arrays (or lists) of ordered ``(x, y)`` points.
"""

import numpy as np
from scipy.interpolate import interp1d


def apply_transformation(R, A):
    R_homo = np.hstack([R, np.ones((R.shape[0], 1))])
    R_transformed = (R_homo @ A.T)[:, :2]
    return R_transformed


def resample_by_arc_length(points, density=1.0, total_points=None):
    """Resamples a list of points evenly by arc length.

    Args:
        points: List of (x, y) tuples representing the path
        density: Points per unit of arc length (default: 1.0). Ignored if total_points is not None.
        total_points: If not None, the number of points to resample to

    Returns:
        List of (x, y) tuples with evenly spaced points by arc length
    """
    if len(points) <= 1:
        return points

    ## Convert to numpy array for easier calculations
    points_array = np.array(points)

    ## Calculate cumulative arc lengths
    diffs = np.diff(points_array, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
    total_length = cumulative_lengths[-1]

    ## If total length is zero (all points are the same), return total_points copies of the first point
    if total_length == 0:
        if total_points is not None:
            return np.array([points[0]] * total_points)
        else:
            return points

    ## Calculate number of points proportional to arc length
    if total_points is None:
        num_points = max(2, int(total_length * density))
    else:
        num_points = total_points

    ## Create evenly spaced arc length values
    arc_lengths = np.linspace(0, total_length, num_points)

    ## Use scipy's interp1d for interpolation
    x_interp = interp1d(cumulative_lengths, points_array[:, 0], kind='linear', bounds_error=False, fill_value=(points_array[0, 0], points_array[-1, 0]))
    y_interp = interp1d(cumulative_lengths, points_array[:, 1], kind='linear', bounds_error=False, fill_value=(points_array[0, 1], points_array[-1, 1]))

    ## Interpolate at the evenly spaced arc lengths
    resampled_x = x_interp(arc_lengths)
    resampled_y = y_interp(arc_lengths)

    ## Combine back into list of tuples
    resampled_points = list(zip(resampled_x, resampled_y))

    return np.array(resampled_points)


def arc_length(points):
    points_np = np.array(points)
    return np.sum(np.linalg.norm(np.diff(points_np, axis=0), axis=1))
