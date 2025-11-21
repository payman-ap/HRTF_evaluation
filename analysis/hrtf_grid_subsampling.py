"""
HRTF subsampling tools

This script builds the candidate measurement grid from a fixed vertical loudspeaker arc (given elevations)
rotated in azimuth and provides several subsampling methods:
 - greedy_angular_exclusion(alpha): keep points that are separated by at least alpha degrees
 - farthest_first(K): choose K points maximizing minimum pairwise angular distance (greedy)
 - sh_based_selection(L): choose elevations (for all azimuths) to target spherical-harmonic order L

It also provides plotting utilities (3D scatter of full candidate set and selected subset) and
conditioning diagnostics for spherical-harmonic fitting.

Usage: run the script or import functions in your own code. Example at the bottom shows how to
call the functions and display results.

Dependencies: numpy, scipy, matplotlib

Author: ChatGPT (GPT-5 Thinking mini)
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import radians

# ------------------------- parameters from your setup -------------------------
ELEVATIONS = np.array([ 90.        , 83.77752   , 68.09975   , 59.49227   , 52.68235   , 46.80371,
                        41.5091    , 36.61715   , 32.01837   , 27.64017   , 23.43094   , 19.35192,
                        15.37263   , 11.46806    , 7.616845   , 3.8        , 0.        , -3.8,
                        -7.61684   , -11.4681   , -15.3726   , -19.3519   , -23.4309   , -27.6402,
                        -32.0184   , -36.6171   , -41.5091   , -46.8037   , -52.6824   , -59.4923,
                        -68.0997   , -80.      ])
RADIUS = 1350.0  # mm
AZ_STEP = 5.0    # degrees -> 72 azimuths

# ------------------------- basic conversions --------------------------------

def sph_from_elev_az(elev_deg, az_deg):
    """Convert elevation (deg, 0=horizontal, +90=top) and azimuth (deg) to colatitude theta (rad) and phi (rad).
    Returns (theta, phi) where theta = colatitude = 90deg - elevation.
    """
    theta = np.deg2rad(90.0 - np.asarray(elev_deg))
    phi = np.deg2rad(np.asarray(az_deg))
    return theta, phi


def unit_cartesian_from_elev_az(elev_deg, az_deg):
    """Return unit Cartesian vector (x,y,z) for given elevation and azimuth in degrees.
    Elevation: 0 = horizontal, +90 = top.
    """
    theta, phi = sph_from_elev_az(elev_deg, az_deg)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.vstack((x, y, z)).T


def angular_distance_matrix(u):
    """Given Nx3 unit vectors u, return NxN angular distance matrix (degrees).
    Uses dot product -> acos(clamp).
    """
    D = np.clip(np.dot(u, u.T), -1.0, 1.0)
    ang = np.rad2deg(np.arccos(D))
    return ang

# ------------------------- candidate grid builder ----------------------------

def build_candidate_grid(elevations=ELEVATIONS, az_step_deg=AZ_STEP, radius=RADIUS):
    azs = np.arange(0.0, 360.0, az_step_deg)
    elevs = elevations
    candidates = []  # list of dicts with fields elev, az, cart (unit), pos (mm)
    for az in azs:
        for e in elevs:
            unit = unit_cartesian_from_elev_az(e, az)[0]
            pos = unit * radius
            candidates.append({'elev': float(e), 'az': float(az), 'unit': unit, 'pos': pos})
    return candidates

# ------------------------- selection algorithms ------------------------------

def greedy_angular_exclusion(candidates, alpha_deg=12.0, preserve_poles=True):
    """Greedy selection: keep points that are at least alpha_deg apart (angular).
    candidates: list as returned by build_candidate_grid
    preserve_poles: if True, always keep exactly the points at +90 and -90 elevations if present
    Returns list of selected candidate indices (into candidates list)
    """
    N = len(candidates)
    units = np.vstack([c['unit'] for c in candidates])
    # Identify poles
    pole_indices = []
    if preserve_poles:
        for i, c in enumerate(candidates):
            if abs(c['elev'] - 90.0) < 1e-6 or abs(c['elev'] + 90.0) < 1e-6:
                pole_indices.append(i)
    # Greedy: iterate candidates in a deterministic order (az then elev) but mark poles first
    order = list(range(N))
    selected = []
    if pole_indices:
        for p in pole_indices:
            selected.append(p)
    for i in order:
        if i in selected:
            continue
        # compute angular distances to selected
        if selected:
            dots = np.dot(units[i], units[selected].T)
            dots = np.clip(dots, -1.0, 1.0)
            angs = np.rad2deg(np.arccos(dots))
            if np.min(angs) < alpha_deg - 1e-9:
                continue
        selected.append(i)
    return selected


def farthest_first(candidates, K=500, seed_index=0):
    """Farthest-first traversal: pick K points maximizing minimum distance to already selected.
    Returns indices of selected candidates.
    """
    units = np.vstack([c['unit'] for c in candidates])
    N = units.shape[0]
    if K >= N:
        return list(range(N))
    selected = [seed_index]
    dists = np.rad2deg(np.arccos(np.clip(units @ units[seed_index], -1.0, 1.0)))
    for _ in range(1, K):
        # distance to nearest selected for each point
        mind = np.min(np.vstack([np.rad2deg(np.arccos(np.clip(units @ units[s], -1.0, 1.0))) for s in selected]), axis=0)
        # pick argmax among unselected
        for idx in np.argsort(-mind):
            if idx not in selected:
                selected.append(int(idx))
                break
    return selected


def sh_design_matrix(candidates, L):
    """Build spherical harmonic design matrix for candidates up to order L (inclusive).
    Returns Y (N x M) where M = (L+1)^2, in real form (real SH basis ordering) and the list of (l,m) pairs.
    Note: uses complex sph_harm and converts to real harmonics: real(Y_lm) for m>0 combined.
    """
    units = np.vstack([c['unit'] for c in candidates])
    # recover elev/az from unit vectors
    x, y, z = units[:,0], units[:,1], units[:,2]
    theta = np.arccos(z)  # colatitude
    phi = np.arctan2(y, x) % (2*np.pi)
    N = len(candidates)
    M = (L+1)**2
    Y = np.zeros((N, M), dtype=float)
    lm_list = []
    col = 0
    for l in range(L+1):
        for m in range(-l, l+1):
            # use complex spherical harmonics
            Y_lm_c = sph_harm(m, l, phi, theta)  # note: sph_harm(m,l,phi,theta)
            if m < 0:
                # real-valued combination: sqrt(2)*(-1)^m Im(Y_l_|m|)
                Y[:, col] = np.sqrt(2) * (-1)**m * np.imag(sph_harm(abs(m), l, phi, theta))
            elif m == 0:
                Y[:, col] = np.real(Y_lm_c)
            else:  # m>0
                Y[:, col] = np.sqrt(2) * (-1)**m * np.real(Y_lm_c)
            lm_list.append((l, m))
            col += 1
    return Y, lm_list


def sh_based_selection(elevations, candidates_all, L_target=16, keep_all_azimuths=True):
    """Select a subset that aims to support spherical harmonic order L_target.
    Strategy used here:
    - We want roughly N >= (L+1)^2 samples.
    - If we keep all azimuths and only drop elevations, compute how many elevations are needed:
         Ntheta = ceil(Nmin / Nphi)
      where Nphi = number of azimuths in candidates_all.
    - Choose that many elevations from the available ones by testing which subset gives better SH conditioning.
    Returns indices of candidates_all selected.
    """
    azs = sorted(set([c['az'] for c in candidates_all]))
    Nphi = len(azs)
    Nmin = (L_target + 1)**2
    Ntheta = int(np.ceil(Nmin / Nphi))
    print(f"SH target L={L_target}: need N_min={Nmin} samples. With Nphi={Nphi} azimuths -> Ntheta ~ {Ntheta} elevations.")
    # Candidate elevation unique set (preserve order)
    elevs = np.array(sorted(set([c['elev'] for c in candidates_all]), reverse=True))
    # We'll search among combinations of elevations (choose Ntheta) — exhaustive combinatorics may be large.
    # Use a heuristic: pick elevations closest to Gauss-Legendre nodes for L_target (Ntheta nodes)
    from numpy.polynomial.legendre import leggauss
    # Gauss-Legendre nodes are roots in [-1,1] for order Ntheta; convert to colatitude
    if Ntheta <= 1:
        chosen_elevs = np.array([0.0])
    else:
        nodes, weights = leggauss(Ntheta)
        # nodes in [-1,1] correspond to cos(theta) values. Convert to theta: theta = arccos(nodes)
        theta_nodes = np.arccos(nodes)  # in radians
        elev_nodes = 90.0 - np.rad2deg(theta_nodes)
        # For each node, pick the available elevation closest to it (and keep symmetry)
        chosen_elevs = []
        for en in elev_nodes:
            # find closest elevation in elevs
            idx = np.argmin(np.abs(elevs - en))
            chosen_elevs.append(elevs[idx])
        chosen_elevs = np.unique(np.round(chosen_elevs, 6))
    print("Chosen elevation rows (deg):", chosen_elevs)
    # now keep all azimuths for those elevations
    chosen_indices = [i for i, c in enumerate(candidates_all) if c['elev'] in chosen_elevs]
    return chosen_indices, chosen_elevs

# ------------------------- plotting utilities --------------------------------

def plot_candidates_and_selection(candidates, selected_indices=None, title=None, show=True):
    units = np.vstack([c['unit'] for c in candidates])
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(units[:,0], units[:,1], units[:,2], s=6, alpha=0.25, label='candidates')
    if selected_indices is not None and len(selected_indices)>0:
        sel = np.array(selected_indices, dtype=int)
        ax.scatter(units[sel,0], units[sel,1], units[sel,2], s=24, alpha=1.0, label='selected')
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    ax.legend()
    if show:
        plt.show()
    return fig, ax

# ------------------------- diagnostics ---------------------------------------

def selection_pairwise_stats(candidates, selected_indices):
    units = np.vstack([c['unit'] for c in candidates])
    sel = np.array(selected_indices, dtype=int)
    if len(sel) < 2:
        return {}
    um = units[sel]
    D = angular_distance_matrix(um)
    # take upper triangle
    iu = np.triu_indices_from(D, k=1)
    vals = D[iu]
    return {
        'n_selected': len(sel),
        'min_deg': float(np.min(vals)),
        'mean_deg': float(np.mean(vals)),
        'median_deg': float(np.median(vals)),
        'max_deg': float(np.max(vals)),
        'hist': vals
    }


def sh_conditioning(candidates, selected_indices, L):
    sel_cands = [candidates[i] for i in selected_indices]
    Y, lm = sh_design_matrix(sel_cands, L)
    # compute SVD
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    cond = s[0] / s[-1] if s[-1] > 0 else np.inf
    return {'cond': cond, 'singular_values': s}

# ------------------------- example usage ------------------------------------
if __name__ == '__main__':
    # build full grid
    cands = build_candidate_grid()
    print(f"Built candidate grid with {len(cands)} points (should be 72*32 = 2304)")

    # 1) Greedy angular exclusion example
    alpha = 12.0  # degrees
    sel_greedy = greedy_angular_exclusion(cands, alpha_deg=alpha, preserve_poles=True)
    stats_g = selection_pairwise_stats(cands, sel_greedy)
    print(f"Greedy alpha={alpha} deg selected {stats_g['n_selected']} points; min angular sep = {stats_g['min_deg']:.2f} deg")
    plot_candidates_and_selection(cands, sel_greedy, title=f'Greedy exclusion alpha={alpha}°')

    # 2) Farthest-first example: select ~600 points
    K = 600
    sel_ff = farthest_first(cands, K=K, seed_index=0)
    stats_ff = selection_pairwise_stats(cands, sel_ff)
    print(f"Farthest-first K={K} selected {stats_ff['n_selected']} points; min angular sep = {stats_ff['min_deg']:.2f} deg")
    plot_candidates_and_selection(cands, sel_ff, title=f'Farthest-first K={K}')

    # 3) SH-based elevation selection for L=16
    L = 16
    sel_sh_indices, chosen_elevs = sh_based_selection(ELEVATIONS, cands, L_target=L)
    stats_sh = selection_pairwise_stats(cands, sel_sh_indices)
    print(f"SH-based selection keeps {len(selected_elevs := sel_sh_indices)} points (elevations: {chosen_elevs})")
    cond_info = sh_conditioning(cands, sel_sh_indices, L)
    print(f"Condition number of SH design matrix up to L={L}: {cond_info['cond']:.2e}")
    plot_candidates_and_selection(cands, sel_sh_indices, title=f'SH-based elevation pick L={L}')

    # end
    print('Done.')
