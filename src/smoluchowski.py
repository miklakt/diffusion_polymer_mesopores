import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

def is_inside_ellipse(x, y, a, b, x0=0.0, y0=0.0, side='full'):
    """
    Determine whether a point (or array of points) lies inside a full or
    half-ellipse centered at (x0, y0).

    The ellipse is defined such that its major axis is aligned along the
    y-axis, with focal points separated by a distance `a`. The semi-minor
    axis is `b`. This construction allows placement of oblate ellipsoidal
    source regions consistent with the analytical solution.

    Parameters
    ----------
    x, y : float or np.ndarray
        Coordinates of the evaluation points.
    a : float
        Distance between the two focal points of the ellipse (along y-axis).
    b : float
        Semi-minor axis length.
    x0, y0 : float, optional
        Center of the ellipse (default is (0,0)).
    side : {"full", "left", "right"}, optional
        Whether to select the full ellipse, or restrict to the left or right half.

    Returns
    -------
    inside : bool or np.ndarray of bool
        True if the point lies inside the specified ellipse region.
    """
    f1 = np.array([x0, y0 - a / 2])
    f2 = np.array([x0, y0 + a / 2])
    p = np.stack([x, y], axis=-1)

    d1 = np.linalg.norm(p - f1, axis=-1)
    d2 = np.linalg.norm(p - f2, axis=-1)

    c = np.sqrt(b**2 + (a / 2)**2)
    inside = (d1 + d2) <= 2 * c

    if side == 'left':
        return inside & (x <= x0)
    elif side == 'right':
        return inside & (x >= x0)
    else:  # 'full'
        return inside

#%%
def R_steady_state(conductivity, bc_source):
    """
    Compute the steady-state resistance of a cylindrical pore system
    by solving the discretized Smoluchowski (generalized Laplace) equation.

    The discretization follows a finite-volume scheme on a 2D cylindrical
    grid in (r,z), with harmonic averaging of conductivity at cell faces
    to preserve flux continuity. Dirichlet boundary conditions are applied
    at the source and sink, and impermeable regions are enforced by setting
    conductivity = 0.

    Parameters
    ----------
    conductivity : 2D ndarray, shape (Nz, Nr)
        Spatially varying conductivity (diffusivity) field.
        Elements with zero value are treated as impermeable walls.
    bc_source : 2D ndarray, shape (Nz, Nr)
        Boolean mask specifying source boundary condition nodes.

    Returns
    -------
    R_total : float
        Effective resistance of the pore system, computed as Δψ / J_total.
    psi : 2D ndarray
        Solution field ψ(r,z), transposed to match (z,r)-convention.
    A : scipy.sparse.csr_matrix
        Sparse system matrix of the discretized operator.

    Notes
    -----
    - Discretization uses Δr = Δz = 1 (uniform grid spacing).
    - The sparse linear system A ψ = b is solved using a direct sparse solver.
    - Fluxes are computed at a control contour (z_c, r_c) enclosing the pore.
    """

    def mean(a,b):
        """Harmonic mean of two conductivities (face-averaged)."""
        if a == 0 or b== 0:
            return 0.0
        return (a**-1+b**-1)**-1*2

    Nr, Nz = np.shape(conductivity)
    dr, dz = 1,1 # uniform grid spacing
    R = np.arange(0, Nr) # radial coordinate array

    # Storage for sparse matrix assembly
    data, rows, cols = [], [], []
    b = np.zeros(Nr * Nz)

    # Transpose so that indexing matches (r,z) convention
    # conductivity = np.array(conductivity.T, dtype=np.float64)
    # bc_source = np.array(bc_source.T, dtype = np.float64)

    # Dirichlet boundary conditions
    psi_source = 1.0
    psi_sink = 0.0

    def idx(i,j):
        """Map 2D indices (i,j) → flattened 1D index."""
        return i*Nz + j

    def is_valid(ii, jj):
        """Check if neighbor index (ii,jj) is valid and not inside a wall."""
        return 0 <= ii < Nr and 0 <= jj < Nz and conductivity[ii, jj] != 0

    def add_stencil_term(i, j, k, coeff_center, stencil):
        """Add stencil contributions for node (i,j) into sparse matrix storage."""
        rows.append(k)
        cols.append(k)
        data.append(coeff_center)
        for col_idx, val in stencil:
            rows.append(k)
            cols.append(col_idx)
            data.append(val)

    # Assemble sparse matrix row-by-row
    for i in range(Nr):
        for j in range(Nz):
            k = idx(i, j)
            r_i = R[i]

            # Source boundary condition: ψ = 1
            if bc_source[i, j]:
                rows.append(k); cols.append(k); data.append(1.0); b[k] = 1.0
                continue
            
            # Sink boundary condition at top (z = Nz - 1): ψ = 0
            if j == Nz - 1:
                if conductivity[i, j] != 0:
                    coeff_center = -3.0 * conductivity[i, j] / dz**2
                    coeff_neigh = conductivity[i, j] / dz**2
                    rows.extend([k, k]); cols.extend([k, idx(i, j - 1)]); data.extend([coeff_center, coeff_neigh])
                    b[k] += 2.0 * conductivity[i, j] * 0.0 / dz**2
                else: #impermeable
                    rows.append(k); cols.append(k); data.append(1.0)
                continue
            
            # impermeable element
            if conductivity[i, j] == 0:
                rows.append(k); cols.append(k); data.append(1.0)
                continue

            coeff_center = 0
            stencil = []

            def add_term(ii, jj, coeff):
                """Add neighbor contribution and subtract from diagonal elements of the matrix"""
                nonlocal coeff_center
                stencil.append((idx(ii, jj), coeff))
                coeff_center -= coeff

            # Special treatment at r = 0
            if i == 0:
                if is_valid(i + 1, j):
                    kp = idx(i + 1, j)
                    rows.extend([k, k]); cols.extend([k, kp]); data.extend([-1 / dr, 1 / dr])
                if j < Nz - 1 and is_valid(i, j + 1):
                    sz_p = mean(conductivity[i, j], conductivity[i, j + 1])
                    add_term(i, j + 1, sz_p / dz**2)
                if j > 0 and is_valid(i, j - 1):
                    sz_m = mean(conductivity[i, j], conductivity[i, j - 1])
                    add_term(i, j - 1, sz_m / dz**2)
                add_stencil_term(i, j, k, coeff_center, stencil)
            
            # outer radial boundary (i = Nr - 1)
            elif i == Nr - 1:
                if is_valid(i - 1, j):
                    km = idx(i - 1, j)
                    rows.extend([k, k]); cols.extend([k, km]); data.extend([-1 / dr, 1 / dr])
                else:
                    rows.append(k); cols.append(k); data.append(1.0)
            # interior elements of the domain
            else:
                if is_valid(i + 1, j):
                    sr_p = mean(conductivity[i + 1, j], conductivity[i, j])
                    add_term(i + 1, j, sr_p / dr**2 + sr_p / (2 * dr * r_i))
                if is_valid(i - 1, j):
                    sr_m = mean(conductivity[i - 1, j], conductivity[i, j])
                    add_term(i - 1, j, sr_m / dr**2 - sr_m / (2 * dr * r_i))
                if is_valid(i, j + 1):
                    sz_p = mean(conductivity[i, j + 1], conductivity[i, j])
                    add_term(i, j + 1, sz_p / dz**2)
                if is_valid(i, j - 1):
                    sz_m = mean(conductivity[i, j - 1], conductivity[i, j])
                    add_term(i, j - 1, sz_m / dz**2)
                add_stencil_term(i, j, k, coeff_center, stencil)

    # Convert triplets → sparse matrix
    data = np.array(data,dtype=np.float64)
    rows = np.array(rows,dtype=np.float64)
    cols = np.array(cols,dtype=np.float64)
    A = sp.coo_matrix((data, (rows, cols)), shape=(Nr * Nz, Nr * Nz), dtype = np.float64).tocsr()

    # Solve linear system A ψ = b
    psi_vec = spla.spsolve(A, b)    
    psi = psi_vec.reshape((Nr, Nz))

    # ----------------------------------------------------
    # Postprocessing: compute fluxes through control surface
    # ----------------------------------------------------
    Jz_total = 0.0

    # Choose contour at some axial (z_c) and radial (r_c) cutoff
    z_c = int(Nz*0.3+50)
    r_c = int(Nr*0.3+50)
    if z_c + 1 >= Nz or r_c + 1 >= Nr:
        raise ValueError("z_c or r_c exceeds domain bounds.")

    # Axial flux contribution (through plane z = z_c)
    for i in range(r_c):
        dpsi_dz = (psi[i, z_c + 1] - psi[i, z_c]) / dz
        cond_face = mean(conductivity[i, z_c], conductivity[i, z_c + 1])
        Jz = -cond_face * dpsi_dz
        Jz_total += np.pi * (2 * R[i]) * Jz * dr

    # Radial flux contribution (through plane r = r_c)
    for j in range(z_c, Nz):
        dpsi_dr = (psi[r_c + 1, j] - psi[r_c, j]) / dr
        cond_face = mean(conductivity[r_c + 1, j], conductivity[r_c, j])
        Jr = cond_face * dpsi_dr
        Jz_total += 2 * np.pi * (R[r_c]) * Jr * dz

    # Resistance from Ohm's law: Δψ / flux
    delta_psi = psi_source - psi_sink
    R_total = delta_psi / Jz_total if Jz_total != 0 else np.inf

    print(
        'Flux:', Jz_total,
        'Resistance:', R_total)
    
    return R_total, psi, A



#%%

def prepare_domain_with_padding(
        conductivity, 
        walls, 
        pore_radius, 
        pore_length, 
        z_boundary):
    """
    Prepare conductivity and boundary-condition fields for the cylindrical
    pore system by applying symmetry reduction and padding the domain.

    The procedure:
      1. Restricts the domain to the z- side (since the problem is symmetric).
      2. Pads the z-direction so the source ellipse lies within the domain.
      3. Pads the r-direction so the radial extent of the source ellipse fits.
      4. Sets conductivity to zero inside impermeable wall regions.
      5. Constructs the binary mask for source boundary nodes based on an
         oblate ellipse criterion.

    Parameters
    ----------
    conductivity : 2D ndarray, shape (Nr, Nz)
        Initial conductivity field (r, z). Values = 0 inside walls.
    walls : 2D ndarray of bool, shape (Nr, Nz)
        Boolean mask of impermeable regions (True = wall).
    pore_radius : float
        Effective pore radius (after colloid exclusion).
    pore_length : float
        Effective pore length (after colloid exclusion).
    z_boundary : int
        Desired axial distance to the boundary plane (extent in z).

    Returns
    -------
    conductivity : 2D ndarray, shape (Nr, Nz)
        Modified conductivity field with padding and wall constraints applied.
    bc_source : 2D ndarray of bool, shape (Nr, Nz)
        Binary mask indicating source boundary condition nodes.

    Notes
    -----
    - Symmetry about the z=0 plane is exploited: only the z- side is kept.
    - Bulk conductivity is inferred from the first non-wall interior value.
    - The source region is defined outside a half-ellipse with
      semi-axes (a = pore_radius, b = z_boundary).
    """
    # --- Transpose input to work internally in (z, r) orientation ---
    conductivity = conductivity.T
    walls = walls.T

    # Keep only the z- half (problem is symmetric in z)
    conductivity = conductivity[:np.shape(conductivity)[0] // 2]
    walls = walls[:np.shape(walls)[0] // 2]

    # Estimate bulk conductivity (from a representative interior point)
    bulk_conductivity = conductivity[1, 1]

    # Pad in z to reach the requested z_boundary
    pad_z = z_boundary - pore_length // 2 + 1
    if pad_z > 0:
        conductivity = np.pad(conductivity,
                              ((pad_z, 0), (0, 0)),
                              "constant",
                              constant_values=bulk_conductivity)
        walls = np.pad(walls,
                       ((pad_z, 0), (0, 0)),
                       "edge")

    # Pad in r so the source ellipse fits inside the domain
    major_axis = int(np.sqrt(z_boundary**2 + pore_radius**2 / 2))
    pad_r = major_axis - np.shape(conductivity)[1] + 1
    if pad_r > 0:
        conductivity = np.pad(conductivity,
                              ((0, 0), (0, pad_r)),
                              "constant",
                              constant_values=bulk_conductivity)
        walls = np.pad(walls,
                       ((0, 0), (0, pad_r)),
                       "edge")

    # Set conductivity = 0 inside wall regions
    conductivity[walls == True] = 0.0

    # Build meshgrid of coordinates
    z, r = np.shape(conductivity)
    R = np.arange(0, r)
    Z = np.arange(0, z)
    RR, ZZ = np.meshgrid(R, Z)

    # Define ellipse center at (x0, y0) in (z,r)-coordinates
    x0 = z_boundary + 1
    y0 = 0

    # Source boundary = outside the ellipse, left side only
    bc_source = ~is_inside_ellipse(
        ZZ, RR,
        a=pore_radius,
        b=z_boundary,
        x0=x0, y0=y0,
        side="left"
    )

    # Only keep z < z_boundary and exclude walls
    bc_source[z_boundary:] = False
    bc_source[walls == True] = False

    # --- Transpose back to (r, z) orientation ---
    return conductivity.T, bc_source.T


def solve_resistance(conductivity, walls,
                     pore_radius, pore_length, 
                     D_0 = 1,
                     z_boundary=1000,
                     colloid_diameter = 0,
                     clip_max_conductivity:float = None
                     ):
    """
    Solve the stationary diffusion (Laplace/Smoluchowski) problem for a
    cylindrical pore system and compute effective resistance.

    Parameters
    ----------
    conductivity : 2D ndarray
        Spatial map of conductivity (z, r). Zero inside walls.
    walls : 2D ndarray of bool
        Boolean mask of impermeable regions (True = wall).
    pore_radius : float
        Effective pore radius.
    pore_length : float
        Effective pore length.
    s : int
        Axial pore shift/offset (used for pore resistance extraction).
    xlayers, ylayers : int
        Target cropped domain size in r and z directions.
    D_0 : float
        Reference bulk diffusivity (for external resistance correction).
    z_boundary : int, optional
        Location of the far-field boundary in z (default 1000).
    conductivity_min, conductivity_max : float, optional
        Clipping thresholds for conductivity field (default 1e-8, 1e8).
    clip : bool, optional
        If True, clip conductivity to finite range.
        If False, set under/overflows to 0/∞.

    Returns
    -------
    R : float
        Total effective resistance of the pore system.
    psi : 2D ndarray
        Symmetrized potential field ψ(r,z) after solving.
    J_r : 2D ndarray
        Radial flux field.
    J_z : 2D ndarray
        Axial flux field.
    """
    nr, nz = np.shape(conductivity)
    # --- Preprocess fields and build source mask ---
    conductivity, bc_source = prepare_domain_with_padding(
        conductivity, walls, pore_radius-colloid_diameter//2, pore_length+colloid_diameter, z_boundary
    )

    # Check conductivity ranges
    if clip_max_conductivity is not None:
        if np.nanmax(conductivity)>=clip_max_conductivity:print("Very high conductivity")
        conductivity[conductivity >= clip_max_conductivity] = clip_max_conductivity

    # --- Solve linear system ---
    R, psi, A = R_steady_state(conductivity, bc_source)

    # Symmetrize solution to z+
    psi_mirror = np.flip(psi, axis=1)
    psi = 0.5 + psi / 2
    psi_mirror = 0.5 - psi_mirror / 2
    psi = np.concatenate([psi, psi_mirror], axis=1)

    # # --- Compute fluxes ---
    conductivity = np.concatenate([conductivity, np.flip(conductivity, axis=1)], axis=1)
    grad_r, grad_z = np.gradient(psi)  # grad_z = dψ/dz, grad_r = dψ/dr
    structure = np.array([[1,1,1]])

    J_z = grad_z * conductivity
    J_r = grad_r * conductivity

    # --- Crop fields back to desired size ---
    nr_, nz_ = psi.shape
    crop_z = (nz_ - nz) // 2
    # crop_r = nr_ - nr

    psi = psi[:nr, crop_z:-crop_z]
    J_z = J_z[:nr, crop_z:-crop_z]
    J_r = J_r[:nr, crop_z:-crop_z]


    # No fluxes to the wall faces
    walls_z = binary_dilation(walls, structure=structure)
    walls_r = binary_dilation(walls, structure=structure.T)
    J_r[walls_r==True] = 0
    J_z[walls_z==True] = 0

    # --- Correct total resistance ---
    R_left = (np.pi - 2 * np.arctan(z_boundary / (pore_radius-colloid_diameter//2))) \
             / (4 * np.pi * (pore_radius-colloid_diameter//2)) / D_0
    R = (R + R_left) * 2  # symmetry

    return {"R":R, "psi":psi, "J_r":J_r, "J_z": J_z}


#%%
if __name__=="__main__":
    from read_namics_output import create_walls
    pore_radius = 26
    pore_length = 52
    colloid_diameter = 0
    walls = create_walls(66, 266, pore_radius, pore_length, colloid_diameter)
    conductivity = np.ones_like(walls, dtype = float)
    conductivity[walls == True] = 0
    free_energy = np.zeros_like(walls, dtype = float)

    result = solve_resistance(conductivity, walls, pore_radius, pore_length, z_boundary=400)
    result["psi"][walls == True] = np.nan
    plt.imshow(result["psi"], origin = "lower", interpolation  = "none")
    plt.title(r"$\psi (r,z)$")

    def empty_pore_permeability(D, r, s):
        return 2*D*r/(1 + 2*s/(r*np.pi))
    perm_0_analytic = empty_pore_permeability(1, pore_radius-colloid_diameter//2, pore_length+colloid_diameter)

    # Sanity check
    print("---Permeability of an empty pore---")
    print(f"Numerical result: {1/result['R']}")
    print(f"Analytical result: {perm_0_analytic}")

# %%
