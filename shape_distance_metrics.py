"""Transformation-invariant shape-distance metrics for 2D trajectories.

This module provides six RANSAC-based shape-distance metrics that measure how
well a *reference* trajectory ``R`` can be aligned to a *target* trajectory
``T`` under a chosen group of geometric transformations. The smaller the
residual distance after the best alignment, the more similar the two shapes are
under that transformation group.

All six metrics are dispatched through :func:`transformation_ransac` via the
``transformation_type`` argument:

==============================  ===============================================
``transformation_type``         Transformation group
==============================  ===============================================
``"rigid"``                     rotation + translation
``"rigid_with_reflections"``    rigid + reflection
``"similarity"``                rotation + uniform scale + translation
``"similarity_with_reflections"`` similarity + reflection
``"affine_without_skew"``       affine without shear
``"affine"``                    full affine
==============================  ===============================================

Each call returns ``(A, history, R_transformed)`` where ``A`` is the best 3x3
transformation matrix, ``history`` is the per-iteration optimization log, and
``R_transformed`` is the aligned reference. The scalar shape distance is
``history[-1]["cost"]`` (the mean point-to-point distance after alignment).

Trajectories are ``(N, 2)`` arrays of ordered ``(x, y)`` points. Closed paths
are detected automatically and use a coarse-to-fine search over cyclic shifts.
"""

import cv2
import time
import numpy as np

from geometry_utils import resample_by_arc_length, apply_transformation, arc_length
from constrained_affine_no_skew import estimate_affine_constrained_no_skew


def is_path_closed(points, threshold=5.0):
    """Check if a path is closed by comparing the first and last points."""
    start_point = np.array(points[0])
    end_point = np.array(points[-1])
    distance = np.linalg.norm(start_point - end_point)
    return distance < threshold


def transformation_ransac(T, R, transformation_type, max_outer_iter=100, max_inner_iter=500, convergence_threshold=1e-1, total_points=100, ransac_threshold=1e-1):
    ## If one of the paths has zero arc length, skip RANSAC
    if np.isclose(arc_length(R), 0) or np.isclose(arc_length(T), 0):
        print("Warning: One of the paths has zero arc length. Skipping RANSAC.")
        return np.eye(3), [{"cost": 999, "A": np.eye(3)}], None

    ## If the paths are closed, use the closed trajectory RANSAC optimization
    if is_path_closed(R) and is_path_closed(T):
        n_refinement_candidates = 2

        ## Closed affine-without-skew trajectories use a single refinement candidate
        ## and a looser threshold.
        if transformation_type == "affine_without_skew":
            n_refinement_candidates = 1
            ransac_threshold = 0.4

        A, history, R_transformed = ransac_optimization_closed_inner(T, R, transformation_type, convergence_threshold=convergence_threshold, max_outer_iter=max_outer_iter, max_inner_iter=max_inner_iter, total_points=total_points, ransac_threshold=ransac_threshold, n_refinement_candidates=n_refinement_candidates)

    else:
        A, history, R_transformed = ransac_optimization_inner(T, R, transformation_type, max_outer_iter=max_outer_iter, max_inner_iter=max_inner_iter, convergence_threshold=convergence_threshold, total_points=total_points, ransac_threshold=ransac_threshold)

    return A, history, R_transformed


def ransac_optimization_closed_inner(T, R, transformation_type, max_outer_iter=100, max_inner_iter=500, convergence_threshold=1e-6, total_points=100, ransac_threshold=1e-3, num_coarse_segments=10, n_refinement_candidates=2, verbose=False):
    """
    RANSAC optimization for closed trajectories with coarse-to-fine search.
    """
    start_time = time.time()
    # print("--- Closed trajectories with coarse-to-fine search. ---")
    best_overall_cost = float('inf')
    best_A = None
    best_history = None
    best_R_transformed = None
    best_shift = None

    # Compute coarse_step based on num_coarse_segments
    coarse_step = max(1, len(R) // num_coarse_segments)
    if verbose:
        print(f"Using coarse_step = {coarse_step} (len(R)={len(R)}, num_coarse_segments={num_coarse_segments})")

    # Stage 1: Coarse search
    if verbose:
        print("=== Stage 1: Coarse Search ===")
    coarse_candidates = []

    for shift in range(0, len(R), coarse_step):
        if verbose:
            print(f"Coarse search - trying shift: {shift}/{len(R)}")
        R_shifted = np.roll(R, shift, axis=0)

        A, history, R_transformed = ransac_optimization_inner(
            T, R_shifted, transformation_type,
            max_outer_iter=max_outer_iter,
            max_inner_iter=max_inner_iter,
            convergence_threshold=convergence_threshold,
            total_points=total_points,
            ransac_threshold=ransac_threshold
        )

        final_cost = history[-1]["cost"]
        coarse_candidates.append({
            'shift': shift,
            'cost': final_cost,
            'A': A,
            'history': history,
            'R_transformed': R_transformed
        })

        if final_cost < best_overall_cost:
            best_overall_cost = final_cost
            best_A = A
            best_history = history
            best_R_transformed = R_transformed
            best_shift = shift
            if verbose:
                print(f"  New best cost: {best_overall_cost:.6f} at shift {shift}")

        # Early stopping: if we found excellent solution in coarse search
        if final_cost < ransac_threshold:  # Very good solution
            if verbose:
                print(f"Early stopping in coarse search! Cost {final_cost:.6f} < threshold {ransac_threshold:.6f}")
            return best_A, best_history, best_R_transformed

    # Sort candidates by cost and select top candidates for refinement
    coarse_candidates.sort(key=lambda x: x['cost'])
    top_candidates = coarse_candidates[:n_refinement_candidates]

    if verbose:
        print(f"\n=== Stage 2: Fine Search around {len(top_candidates)} best candidates ===")
        print(f"Top candidate shifts: {[c['shift'] for c in top_candidates]}")
        print(f"Top candidate costs: {[c['cost'] for c in top_candidates]}")

    # Stage 2: Fine search around best candidates
    for candidate in top_candidates:
        center_shift = candidate['shift']

        # Search in neighborhood around this candidate
        start_shift = max(0, center_shift - coarse_step)
        end_shift = min(len(R), center_shift + coarse_step + 1)

        if verbose:
            print(f"\nRefining around shift {center_shift} (range: {start_shift}-{end_shift})")

        for shift in range(start_shift, end_shift):
            # Skip if we already evaluated this shift in coarse search
            if shift % coarse_step == 0 and shift >= 0:
                continue
            if verbose:
                print(f"Fine search - trying shift: {shift}/{len(R)}")
            R_shifted = np.roll(R, shift, axis=0)

            A, history, R_transformed = ransac_optimization_inner(
                T, R_shifted, transformation_type,
                max_outer_iter=max_outer_iter,
                max_inner_iter=max_inner_iter,
                convergence_threshold=convergence_threshold,
                total_points=total_points,
                ransac_threshold=ransac_threshold
            )

            final_cost = history[-1]["cost"]

            if final_cost < best_overall_cost:
                best_overall_cost = final_cost
                best_A = A
                best_history = history
                best_R_transformed = R_transformed
                best_shift = shift
                if verbose:
                    print(f"  New best cost: {best_overall_cost:.6f} at shift {shift}")

            # Early stopping: found solution below threshold
            if final_cost < ransac_threshold:
                if verbose:
                    print(f"Early stopping! Cost {final_cost:.6f} < threshold {ransac_threshold:.6f}")
                return best_A, best_history, best_R_transformed

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\n=== Final Result ===")
        print(f"Best cost: {best_overall_cost:.6f} at shift {best_shift}")
    # print(f"--- Total time: {elapsed_time:.2f} seconds. Best cost: {best_overall_cost:.6f} at shift {best_shift} ---")

    return best_A, best_history, best_R_transformed


def ransac_optimization_inner(T, R, transformation_type, max_outer_iter=100, max_inner_iter=500, convergence_threshold=1e-6, total_points=100, ransac_threshold=1e-3):
    T = resample_by_arc_length(T, total_points=total_points)
    R_transformed = None

    histories = []
    R_transformed_list = []
    used_combinations = set()
    for i in range(max_outer_iter):
        # randomly sample a subset of 3 points, ensuring no duplicate combinations
        for attempt in range(1000):
            indices = np.random.choice(T.shape[0], size=3, replace=False)
            indices_tuple = tuple(sorted(indices))  ## Sort to make order-independent
            if indices_tuple not in used_combinations:
                ## Check if points are collinear
                p1, p2, p3 = T[indices]
                v1 = p2 - p1
                v2 = p3 - p1
                cross_product = v1[0] * v2[1] - v1[1] * v2[0]
                if abs(cross_product) < 1e-6:  # Points are collinear
                    used_combinations.add(indices_tuple)
                    continue
                used_combinations.add(indices_tuple)
                break
        R_transformed = resample_by_arc_length(R, total_points=total_points)

        converged = False
        history = []
        prev_distance = float('inf')
        prev_convergence_diff = float('inf')
        stagnation_count = 0
        stagnation_threshold = 2  # Stop if no improvement for stagnation_threshold iterations

        for j in range(max_inner_iter):  # inner iterations to refine
            if transformation_type == "rigid":
                try:
                    A_2x3 = kabsch_umeyama(R_transformed[indices], T[indices], allow_scale=False, allow_reflection=False)
                except:
                    A_2x3 = None

            elif transformation_type == "rigid_with_reflections":
                try:
                    A_2x3 = kabsch_umeyama(R_transformed[indices], T[indices], allow_scale=False, allow_reflection=True)
                except:
                    A_2x3 = None

            elif transformation_type == "similarity":
                A_2x3, _ = cv2.estimateAffinePartial2D(R_transformed[indices], T[indices], method=cv2.RANSAC)

            elif transformation_type == "similarity_with_reflections":
                try:
                    A_2x3 = kabsch_umeyama(R_transformed[indices], T[indices], allow_scale=True, allow_reflection=True)
                except:
                    A_2x3 = None

            elif transformation_type == "affine_without_skew":
                try:
                    params, A = estimate_affine_constrained_no_skew(
                        R_transformed[indices], T[indices],
                        # fix_params={'skew_x': 0.0},
                        # initial_params={'tx': 0, 'ty': 0, 'rotation': 0,
                        #                 'sx': 1, 'sy': 1, 'skew_x': 0}
                    )
                    A_2x3 = A[:2, :]
                except:
                    A_2x3 = None

            elif transformation_type == "affine":
                src_pts = np.array(R_transformed[indices], dtype=np.float32)
                dst_pts = np.array(T[indices], dtype=np.float32)
                A_2x3 = cv2.getAffineTransform(src_pts, dst_pts)

            else:
                raise ValueError(f"Unknown transformation type: {transformation_type}")

            ## if still None, skip this iteration
            if A_2x3 is None or np.isnan(A_2x3).any():
                continue

            A = np.vstack([A_2x3, [0, 0, 1]])

            # Apply the accumulated transformation to original R
            R_transformed = apply_transformation(R, A)

            ## Resample R again
            R_transformed = resample_by_arc_length(R_transformed, total_points=total_points)

            ## compute distance
            distances = np.linalg.norm(R_transformed - T, axis=1)
            average_distance = np.mean(distances)

            history.append({
                "cost": average_distance,
                "A": A,
            })

            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                # print("Warning: Singular matrix encountered, skipping inversion.")
                continue

            convergence_diff = abs(average_distance - prev_distance)
            if convergence_diff < convergence_threshold:
                # print(f"Converged after {j} iterations: {average_distance}")
                converged = True
                break

            if convergence_diff > prev_convergence_diff:
                stagnation_count += 1
                if stagnation_count > stagnation_threshold:
                    # print(f"    Stagnation threshold {stagnation_threshold} reached after {j} iterations: {average_distance}")
                    break

            prev_distance = average_distance
            prev_convergence_diff = convergence_diff
            R_transformed = apply_transformation(R_transformed, A_inv)

        if converged:
            histories.append(history)
            R_transformed_list.append(R_transformed)
            if average_distance < ransac_threshold:
                # print(f"Overall converged after {i} iterations: {average_distance}")
                break

    ## find the minimum cost history
    if not histories:
        # print("No histories found")
        return np.eye(3), [{"cost": 999, "A": np.eye(3)}], None

    min_cost_index = np.argmin([history[-1]["cost"] for history in histories])
    min_cost_history = histories[min_cost_index]
    return min_cost_history[-1]["A"], min_cost_history, R_transformed_list[min_cost_index]


def kabsch_umeyama(R, T, allow_scale=True, allow_reflection=False):
    """
    Estimate rigid or similarity transform (rotation, uniform scale, translation, possibly reflection)
    By default, estimates similarity transform (with uniform scaling).
    """
    ## Center the points
    centroid_R = np.mean(R, axis=0)
    centroid_T = np.mean(T, axis=0)
    R_centered = R - centroid_R
    T_centered = T - centroid_T
    n = R.shape[0]

    ## Compute optimal rotation using SVD
    H = R_centered.T @ T_centered  # 2x2 matrix
    if allow_scale:
        H = H / n

    U, D, Vt = np.linalg.svd(H)
    S = np.eye(R.shape[1])

    ## Reflection
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    if d < 0 and allow_reflection == False:
            S[1, 1] = -1

    R_opt = Vt.T @ S @ U.T
    # R_opt = U @ S @ Vt

    ## Compute scale if allowed
    s = 1.0
    if allow_scale:
        ## Variance of R
        var_R = np.sum(R_centered ** 2) / n
        s = np.trace(np.diag(D) @ S) / var_R

    ## Compute translation: t = centroid_T - R_opt @ centroid_R
    t_opt = centroid_T - s * R_opt @ centroid_R

    A_matrix = np.hstack([s * R_opt, t_opt.reshape(2, 1)])

    return A_matrix
