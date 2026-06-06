"""
Specialized affine estimation with skew fixed to zero and no parameter bounds.

Uses Levenberg-Marquardt with analytical Jacobian for fast convergence.
Simple centroid-based initialization ensures robust convergence in RANSAC loops.
"""

from scipy.optimize import least_squares
import numpy as np
from numba import jit


@jit(nopython=True)
def _compute_affine_coefficients_no_skew(rotation, sx, sy):
    """Compute affine coefficients when skew_x is fixed at zero."""
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)

    a11 = sx * cos_r
    a12 = -sy * sin_r
    a21 = sx * sin_r
    a22 = sy * cos_r

    return a11, a12, a21, a22


@jit(nopython=True)
def transform_points_fast_no_skew(src_points, tx, ty, rotation, sx, sy):
    """Transform points without building the full affine matrix."""
    a11, a12, a21, a22 = _compute_affine_coefficients_no_skew(rotation, sx, sy)

    n = src_points.shape[0]
    transformed = np.empty((n, 2))

    for i in range(n):
        x, y = src_points[i]
        transformed[i, 0] = a11 * x + a12 * y + tx
        transformed[i, 1] = a21 * x + a22 * y + ty

    return transformed


def compute_affine_no_skew(tx, ty, rotation, sx, sy):
    """Return the final 2x3 affine matrix with zero skew."""
    a11, a12, a21, a22 = _compute_affine_coefficients_no_skew(rotation, sx, sy)

    return np.array([
        [a11, a12, tx],
        [a21, a22, ty]
    ])


def compute_analytical_jacobian_no_skew(src_points, tx, ty, rotation, sx, sy, free_indices, param_names):
    """
    Compute the analytical Jacobian for the zero-skew parameterization.

    Eliminates the skew derivative column to reduce per-iteration work.
    """
    n = src_points.shape[0]
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)

    J = np.zeros((2 * n, len(free_indices)))

    for param_idx, param_name_idx in enumerate(free_indices):
        param_name = param_names[param_name_idx]

        if param_name == 'tx':
            J[::2, param_idx] = 1.0
        elif param_name == 'ty':
            J[1::2, param_idx] = 1.0
        elif param_name == 'rotation':
            for i in range(n):
                x, y = src_points[i]
                J[2 * i, param_idx] = -sx * sin_r * x - sy * cos_r * y
                J[2 * i + 1, param_idx] = sx * cos_r * x - sy * sin_r * y
        elif param_name == 'sx':
            for i in range(n):
                x = src_points[i, 0]
                J[2 * i, param_idx] = cos_r * x
                J[2 * i + 1, param_idx] = sin_r * x
        elif param_name == 'sy':
            for i in range(n):
                y = src_points[i, 1]
                J[2 * i, param_idx] = -sin_r * y
                J[2 * i + 1, param_idx] = cos_r * y

    return J


def estimate_affine_constrained_no_skew(src_points, dst_points, fix_params=None, initial_params=None, max_nfev=30):
    """
    Estimate an affine transform with skew fixed at zero and no parameter bounds.

    Uses Levenberg-Marquardt with analytical Jacobian for fast convergence.
    Simple centroid-based initialization ensures robust convergence in RANSAC loops.
    """
    if fix_params is None:
        fix_params = {}

    src_points = np.ascontiguousarray(src_points, dtype=np.float64)
    dst_points = np.ascontiguousarray(dst_points, dtype=np.float64)

    param_names = ['tx', 'ty', 'rotation', 'sx', 'sy']

    if initial_params is None:
        ## Simple centroid-based initialization
        src_center = np.mean(src_points, axis=0)
        dst_center = np.mean(dst_points, axis=0)
        initial_params = {
            'tx': dst_center[0] - src_center[0],
            'ty': dst_center[1] - src_center[1],
            'rotation': 0.0,
            'sx': 1.0,
            'sy': 1.0
        }
        initial_params.update(fix_params)

    free_params = [name for name in param_names if name not in fix_params]
    if not free_params:
        raise ValueError("At least one parameter must remain free for optimization.")

    x0 = np.array([initial_params[name] for name in free_params], dtype=np.float64)

    params_array = np.array([initial_params[name] for name in param_names], dtype=np.float64)
    free_indices = [i for i, name in enumerate(param_names) if name not in fix_params]

    def residual_vector(x):
        for i, idx in enumerate(free_indices):
            params_array[idx] = x[i]

        transformed = transform_points_fast_no_skew(
            src_points,
            params_array[0], params_array[1], params_array[2],
            params_array[3], params_array[4]
        )

        diff = transformed - dst_points
        return diff.ravel()

    def jacobian_func(x):
        for i, idx in enumerate(free_indices):
            params_array[idx] = x[i]

        return compute_analytical_jacobian_no_skew(
            src_points,
            params_array[0], params_array[1], params_array[2],
            params_array[3], params_array[4],
            free_indices, param_names
        )

    result = least_squares(
        residual_vector,
        x0,
        jac=jacobian_func,
        method='lm',
        max_nfev=max_nfev
    )

    final_params = initial_params.copy()
    final_params.update(fix_params)
    for i, name in enumerate(free_params):
        final_params[name] = result.x[i]

    final_matrix = compute_affine_no_skew(**final_params)

    return final_params, final_matrix


