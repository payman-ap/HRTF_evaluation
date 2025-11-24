import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------- parameters from my setup -------------------------
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
    theta = np.deg2rad(90.0 - np.asarray(elev_deg))
    phi = np.deg2rad(np.asarray(az_deg))
    return theta, phi

def unit_cartesian_from_elev_az(elev_deg, az_deg):
    theta, phi = sph_from_elev_az(elev_deg, az_deg)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.vstack((x, y, z)).T

def angular_distance_matrix(u):
    D = np.clip(np.dot(u, u.T), -1.0, 1.0)
    return np.rad2deg(np.arccos(D))

# ------------------------- candidate builder --------------------------------

def build_candidate_grid(elevations, az_step_deg=5.0, radius=1350.0):
    azs = np.arange(0.0, 360.0, az_step_deg)
    candidates = []
    for az in azs:
        for e in elevations:
            unit = unit_cartesian_from_elev_az(e, az)[0]
            pos = unit * radius
            candidates.append({'elev': float(e), 'az': float(az), 'unit': unit, 'pos': pos})
    return candidates

def directions_to_candidates(directions):
    """Convert Nx3 array [az, elev, radius] to candidate dicts with 'unit'."""
    candidates = []
    for row in directions:
        az, elev, r = row
        unit = unit_cartesian_from_elev_az(elev, az)[0]
        pos = unit * r
        candidates.append({'az': float(az), 'elev': float(elev), 'unit': unit, 'pos': pos})
    return candidates

# ------------------------- selection algorithms ------------------------------

def greedy_angular_exclusion(candidates, alpha_deg=12.0, preserve_poles=True):
    N = len(candidates)
    units = np.vstack([c['unit'] for c in candidates])
    pole_indices = []
    if preserve_poles:
        for i, c in enumerate(candidates):
            if abs(c['elev'] - 90.0) < 1e-6 or abs(c['elev'] + 90.0) < 1e-6:
                pole_indices.append(i)
    order = list(range(N))
    selected = []
    if pole_indices:
        for p in pole_indices:
            selected.append(p)
    for i in order:
        if i in selected:
            continue
        if selected:
            dots = np.dot(units[i], units[selected].T)
            dots = np.clip(dots, -1.0, 1.0)
            angs = np.rad2deg(np.arccos(dots))
            if np.min(angs) < alpha_deg - 1e-9:
                continue
        selected.append(i)
    return selected

def farthest_first(candidates, K=500, seed_index=0):
    units = np.vstack([c['unit'] for c in candidates])
    N = units.shape[0]
    if K >= N:
        return list(range(N))
    selected = [seed_index]
    for _ in range(1, K):
        mind = np.min(np.vstack([np.rad2deg(np.arccos(np.clip(units @ units[s], -1.0, 1.0))) for s in selected]), axis=0)
        for idx in np.argsort(-mind):
            if idx not in selected:
                selected.append(int(idx))
                break
    return selected

def sh_design_matrix(candidates, L):
    units = np.vstack([c['unit'] for c in candidates])
    x, y, z = units[:,0], units[:,1], units[:,2]
    theta = np.arccos(z)
    phi = np.arctan2(y, x) % (2*np.pi)
    N = len(candidates)
    M = (L+1)**2
    Y = np.zeros((N, M), dtype=float)
    lm_list = []
    col = 0
    for l in range(L+1):
        for m in range(-l, l+1):
            Y_lm_c = sph_harm(m, l, phi, theta)
            if m < 0:
                Y[:, col] = np.sqrt(2) * (-1)**m * np.imag(sph_harm(abs(m), l, phi, theta))
            elif m == 0:
                Y[:, col] = np.real(Y_lm_c)
            else:
                Y[:, col] = np.sqrt(2) * (-1)**m * np.real(Y_lm_c)
            lm_list.append((l, m))
            col += 1
    return Y, lm_list

def sh_based_selection(elevations, candidates_all, L_target=16):
    azs = sorted(set([c['az'] for c in candidates_all]))
    Nphi = len(azs)
    Nmin = (L_target + 1)**2
    Ntheta = int(np.ceil(Nmin / Nphi))
    elevs = np.array(sorted(set([c['elev'] for c in candidates_all]), reverse=True))
    from numpy.polynomial.legendre import leggauss
    if Ntheta <= 1:
        chosen_elevs = np.array([0.0])
    else:
        nodes, _ = leggauss(Ntheta)
        theta_nodes = np.arccos(nodes)
        elev_nodes = 90.0 - np.rad2deg(theta_nodes)
        chosen_elevs = []
        for en in elev_nodes:
            idx = np.argmin(np.abs(elevs - en))
            chosen_elevs.append(elevs[idx])
        chosen_elevs = np.unique(np.round(chosen_elevs,6))
    chosen_indices = [i for i, c in enumerate(candidates_all) if c['elev'] in chosen_elevs]
    return chosen_indices, chosen_elevs

# ------------------------- plotting ------------------------------------------

def plot_candidates_and_selection(candidates, selected_indices=None, title=None, show=True):
    # Handle both dicts or raw Nx3 array
    if isinstance(candidates, np.ndarray) and candidates.shape[1] == 3:
        # convert to candidate dicts internally
        candidates = directions_to_candidates(candidates)
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
    Y, _ = sh_design_matrix(sel_cands, L)
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    cond = s[0]/s[-1] if s[-1]>0 else np.inf
    return {'cond': cond, 'singular_values': s}



if __name__=="__main__":
    # 1. Build full candidate grid
    candidates = build_candidate_grid(ELEVATIONS, az_step_deg=5.0)

    # 2. Greedy selection
    selected_greedy = greedy_angular_exclusion(candidates, alpha_deg=12.0)
    plot_candidates_and_selection(candidates, selected_greedy, title="Greedy Angular Exclusion")

    # 3. Farthest-first
    selected_ff = farthest_first(candidates, K=500)
    plot_candidates_and_selection(candidates, selected_ff, title="Farthest-First")

    # 4. SH-based selection
    selected_sh, elev_chosen = sh_based_selection(ELEVATIONS, candidates, L_target=16)
    plot_candidates_and_selection(candidates, selected_sh, title="SH-based Elevation Selection")

    # 5. Optional: diagnostics
    stats = selection_pairwise_stats(candidates, selected_greedy)
    cond = sh_conditioning(candidates, selected_greedy, L=16)
    print(stats, cond['cond'])





