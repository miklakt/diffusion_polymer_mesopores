#%%
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from functools import lru_cache
import matplotlib.pyplot as plt
from typing import Tuple, List

# ----------------------------------------------------------------------
# Utility: downscale 2D arrays conservatively
# ----------------------------------------------------------------------
def bin2d(a:np.ndarray, K0:int, K1:int) -> np.ndarray:
    """
    Downscale a 2D numpy array by averaging over non-overlapping blocks.

    Parameters
    ----------
    a : np.ndarray
        Input 2D array.
    K0 : int
        Downscale factor along axis 0.
    K1 : int
        Downscale factor along axis 1.

    Returns
    -------
    np.ndarray
        Downscaled array, preserving the average value (not the sum).
    """
    m_bins = a.shape[0]//K0
    n_bins = a.shape[1]//K1
    return a.reshape(m_bins, K0, n_bins, K1).mean(3).mean(1)

# ----------------------------------------------------------------------
# Circle intersection and interpositions
# ----------------------------------------------------------------------
def circles_points_of_intersection(d:float, R1:float, R2:float) -> List:
    """
    Compute intersection points of two circles centered at (0,0) and (0,d).

    Parameters
    ----------
    d : float
        Distance between circle centers.
    R1 : float
        Radius of the first circle (center at origin).
    R2 : float
        Radius of the second circle (center at (0,d)).

    Returns
    -------
    list
        [x, y] of intersection point with y ≥ 0. Empty list if no intersection.
    """

    # circle1 is at the origin with radius r1
    # d distance to the second circle with radius r2

    # No intersection
    if ((R1+R2)<d) or np.abs(R1-R2)>d:
        return []

    # Tangential intersection (one point)
    if ((R1+R2)==d) or ((R1-R2)==d):
        return [R1, 0]

    # Two intersections (choose y ≥ 0)
    x = (R1**2 - R2**2 + d**2)/(2*d)
    y = np.sqrt(max(0,R1**2 - x**2))
    return [x, y]


def circles_interposition(d:float, R1:float, R2:float) -> str:
    """
    Determine relative positioning (interposition) of two circles.

    Possible outcomes: 'coincide', 'no_overlap', 'subset', 'superset', 'intersect'.

    Parameters
    ----------
    d : float
        Distance between centers.
    R1 : float
        Radius of the first circle.
    R2 : float
        Radius of the second circle.

    Returns
    -------
    str
        Relationship between the circles.
    """ 
    if d==0 and R1==R2:
        return "coincide"
    if ((R1+R2)<d):
        return "no_overlap"
    if (R1-R2)>d:
        return "subset"
    if (R2-R1)>d:
        return "superset"
    else:
        return "intersect"
    
def circle_intersection_area(d : float, R1 : float, R2 : float) -> float:
    """
    Calculate the area of intersection between two circles.

    Parameters
    ----------
    d : float
        Distance between centers.
    R1 : float
        Radius of first circle.
    R2 : float
        Radius of second circle.

    Returns
    -------
    float
        Intersection area.
    """

    # No overlap
    if d >= R1 + R2:
        return 0

    # One circle inside the other
    if d <= abs(R1 - R2):
        return np.pi * min(R1, R2) ** 2

    alpha = 2 * np.arccos((R1**2 + d**2 - R2**2) / (2 * R1 * d))
    beta = 2 * np.arccos((R2**2 + d**2 - R1**2) / (2 * R2 * d))

    return (
        0.5 * (alpha - np.sin(alpha)) * R1**2
        + 0.5 * (beta - np.sin(beta)) * R2**2
    )

def circle_annulus_intersection_area(d: float, R: float, r: float, dr: float) -> float:
    """
    Intersection area between a circle of radius R and an annulus [r, r+dr].

    Parameters
    ----------
    d : float
        Distance between centers.
    R : float
        Radius of the outer circle.
    r : float
        Inner radius of annulus.
    dr : float
        Annulus thickness.

    Returns
    -------
    float
        Intersection area.
    """
    return circle_intersection_area(d, R, r + dr) - circle_intersection_area(d, R, r)

#%%

def circle_annulus_intersections_segment_length(d, R, r, dr):
    """
    Arc segment length of circle R overlapping with annulus [r, r+dr].

    Returns (theta1, theta2, segment_length). Returns zeros if no intersection.
    """

    no_intersect_result = 0, 0, 0
    if R == 0: return no_intersect_result

    pos1 = circles_interposition(d, R, r)
    pos2 = circles_interposition(d, R, r+dr)

    theta1 = None
    theta2 = None
    
    if pos1 == "no_overlap":
        if pos2 == "no_overlap": return no_intersect_result
        elif pos2 == "intersect": theta1 = 0
        elif pos2 == "subset": raise Exception("impossible intersection")
        elif pos2 == "superset": theta1, theta2 = 0, np.pi
        elif pos2 == "coincide": raise Exception("impossible intersection")
        else: raise Exception("wrong interposition string")

    elif pos1 == "intersect":
        if pos2 == "no_overlap": raise Exception("impossible intersection")
        elif pos2 == "intersect": pass
        elif pos2 == "subset": raise Exception("impossible intersection")
        elif pos2 == "superset": theta2 = np.pi
        elif pos2 == "coincide": raise Exception("impossible intersection")
        else: raise Exception("wrong interposition string")

    elif pos1 == "subset":
        if pos2 == "no_overlap": raise Exception("impossible intersection")
        elif pos2 == "intersect": theta1=0
        elif pos2 == "subset": return no_intersect_result
        elif pos2 == "superset": theta1, theta2 = 0, np.pi
        elif pos2 == "coincide": return no_intersect_result
        else: raise Exception("wrong interposition string")

    elif pos1 == "superset":
        if pos2 == "no_overlap": raise Exception("impossible intersection")
        elif pos2 == "intersect": raise Exception("impossible intersection")
        elif pos2 == "subset": raise Exception("impossible intersection")
        elif pos2 == "superset": return no_intersect_result
        elif pos2 == "coincide": raise Exception("impossible intersection")
        else: raise Exception("wrong interposition string")

    elif pos1 == "coincide":
        if pos2 == "no_overlap": raise Exception("impossible intersection")
        elif pos2 == "intersect": raise Exception("impossible intersection")
        elif pos2 == "subset": raise Exception("impossible intersection")
        elif pos2 == "superset": theta1, theta2 = 0, np.pi
        elif pos2 == "coincide": theta1, theta2 = 0, np.pi
        else: raise Exception("wrong interposition string")

    else:
        raise Exception("wrong interposition string")

    if theta1 is None:
        x,y = circles_points_of_intersection(d, R, r)
        theta1 = np.arccos(x/R)

    if theta2 is None:
        x,y = circles_points_of_intersection(d, R, r+dr)
        theta2 = np.arccos(x/R)

    segment_length = R*(theta2-theta1)*2

    return theta1, theta2, segment_length

def circle_segment_length_two_lines(R, d1, d2):
    """
    Arc length of a circle between two vertical lines x=d1 and x=d2.
    """
    if R==0: return 0
    if d1 == d2: return 0
    if d1 > d2: d1, d2 = d2, d1
    if d1 > R: return 0
    if d2 <= -R: return 0

    d1 = max(d1, -R)
    d2 = min(d2, R)

    R2 = R**2
    y1 = np.sqrt(R2-d1**2)
    y2 = np.sqrt(R2-d2**2)

    theta1 = np.arctan2(d1,y1)
    theta2 = np.arctan2(d2,y2)
    segment_length = R*(theta2-theta1)*2
    return segment_length

class Sphere2D:
    """
    Represent a sphere offset from the r-axis in cylindrical coordinates.
    Provides methods to compute surface/volume projections.
    """
    d = 0 #distance from the center to the origin
    R = 0 #sphere radius
    V = 0 #sphere volume
    S = 0 #sphere surface

    def __init__(self, radius, d, eps = None):
        """
        Parameters
        ----------
        radius : float
            Sphere radius.
        d : float
            Radial offset from origin (distance to pore axis).
        eps : float, optional
            Integration step. Defaults to 500 / radius grid.
        """
        self.R = radius
        self.d = d
        self.V = 4/3*np.pi*self.R**3
        self.S = 4*np.pi*self.R**2
        if eps is None:
            self.eps = 1/int(500/radius)


    def crossection_r(self, z):
        """Radius of cross-section circle at axial position z."""
        if np.abs(z)>self.R: return 0
        return np.sqrt(self.R**2-z**2)

    def arc_length_in_sphere_rz(self, r, z):
        """Arc length of intersection between circle r and sphere cross-section at z."""
        if r==0: return 0
        r_crossection = self.crossection_r(z)
        d = self.d
        if ((r_crossection-r)>=d): return 2*np.pi*r

        intersection_points = circles_points_of_intersection(d, r, r_crossection)
        if intersection_points:
            x, y = intersection_points
            if y==0: return 0
            arc_angle = 2*np.arccos(x/r)
            arc_length = arc_angle*r
            return arc_length
        else:
            return 0

    def r_crossection_annulus_intersection_length(self, r, z, dr):
        """Arc length of intersection with annulus [r, r+dr] at height z."""
        if r==0: return 0
        r_crossection = self.crossection_r(z)
        d = self.d
        theta1, theta2, intersection_length  = circle_annulus_intersections_segment_length(d, r_crossection, r, dr)
        return intersection_length

    @lru_cache
    def integrate_volume(self, r0, z0, r1, z1):
        """
        Numerically integrate the volume projection of the sphere in cylindrical coordinates.

        The method evaluates how much of the sphere's volume overlaps with a bounding box
        defined in (r,z) space: [r0, r1] × [z0, z1].

        Approach
        --------
        In cylindrical coordinates, the volume element is:
            dV = r · dθ · dr · dz

        - For each grid point (r, z), we determine the arc length of the circle of radius r
        lying inside the sphere cross-section at height z.
        (This arc length is effectively r · Δθ.)

        - Multiplying the arc length by the cell area (eps²) approximates the volume
        contribution of that grid cell.

        - Summing contributions over the grid yields the total volume within the box.

        Parameters
        ----------
        r0, r1 : float
            Radial bounds of the integration domain.
        z0, z1 : float
            Axial bounds of the integration domain.

        Returns
        -------
        integral : float
            Total integrated volume inside the box.
        arc_length : np.ndarray
            2D array of arc lengths for each (r,z) grid point,
            representing the projection kernel before normalization.
        """
        eps = self.eps  # grid spacing for discretization

        # Discretize radial and axial coordinates
        r_ticks = np.arange(r0, r1, eps)
        z_ticks = np.arange(z0, z1, eps)

        # Array to store arc lengths for each (r,z)
        arc_length = np.zeros((len(r_ticks), len(z_ticks)))

        # Loop over grid points
        for i, r in enumerate(r_ticks):
            for j, z in enumerate(z_ticks):
                # Arc length of overlap between circle at (r,z) and sphere cross-section
                arc = self.arc_length_in_sphere_rz(r, z)
                arc_length[i, j] = arc

        # Approximate integral: sum of arc lengths × cell area
        integral = np.sum(arc_length) * eps**2

        return integral, arc_length
    

    @lru_cache
    def integrate_surface(self, r0, z0, r1, z1):
        eps = self.eps
        r_ticks = np.arange(r0, r1, eps)
        z_ticks = np.arange(z0, z1, eps)
        dz = [circle_segment_length_two_lines(self.R, z_, z_+eps)/2 for z_ in z_ticks]
        area_element = np.zeros((len(r_ticks), len(z_ticks)))
        for i, r in enumerate(r_ticks):
            for j, z in enumerate(z_ticks):
                segment = self.r_crossection_annulus_intersection_length(r, z, eps)
                area_element[i, j] = segment*dz[j]
        integral = np.sum(area_element)
        area_element = area_element/eps**2
        return integral, area_element

    def get_bounding_box(self):
        d=self.d
        R = self.R
        r0=max(0, d-R)
        r1=R+d
        z0=-R
        z1=R
        return [r0, r1, z0, z1]

    def generate_volume_kernel(self, bins = 1, tune = True):
        eps = self.eps
        r0, r1, z0, z1 = self.get_bounding_box()
        integral, kernel = self.integrate_volume(r0, z0, r1, z1)
        kernel = bin2d(kernel, int(1/eps/bins), int(1/eps/bins))
        if tune:
            F = self.V/integral
            if F > 1.01 or F < 0.995:
                raise Exception("Not enough precision")
            else:
                kernel = kernel*F
        return kernel

    def generate_surface_kernel(self, bins = 1, tune = True):
        eps = self.eps
        r0, r1, z0, z1 = self.get_bounding_box()
        integral, kernel = self.integrate_surface(r0, z0, r1, z1)
        kernel = bin2d(kernel, int(1/eps/bins), int(1/eps/bins))
        if tune:
            F = self.S/integral
            if F > 1.01 or F < 0.995:
                raise Exception("Not enough precision")
            else:
                kernel = kernel*F
        return kernel


# Try to import joblib.Memory
try:
    from joblib import Memory
    memory = Memory("__func_cache__", verbose=0)
    cache_decorator = memory.cache
except ImportError:
    from functools import lru_cache
    cache_decorator = lru_cache(maxsize=None)

@cache_decorator
def generate_sphere_volume_surface_matrices(sphere_radius, center_coord: float | list, bins=1):
    """
    Generate volume and surface kernels for a sphere at arbitrary offset.

    Parameters
    ----------
    sphere_radius : float
        Sphere radius.
    center_coord : float | list
        Offset of the sphere center. If float -> radial offset only. If list/tuple (z_shift, r_shift).
    bins : int
        Downscaling factor.

    Returns
    -------
    tuple
        (volume_kernel, surface_kernel, extent [r0,r1,z0,z1])
    """
    try:
        z_shift, r_shift = center_coord
    except TypeError:
        z_shift = 0
        r_shift = center_coord
    sphere = Sphere2D(sphere_radius, r_shift)
    volume = sphere.generate_volume_kernel(bins)
    surface = sphere.generate_surface_kernel(bins)
    extent = sphere.get_bounding_box()
    extent[0], extent[1], extent[2], extent[3] = extent[2], extent[3], extent[0], extent[1]
    extent[0] += z_shift
    extent[1] += z_shift
    return volume, surface, extent

#%%
if __name__ == "__main__":
    # Example usage for a sphere of radius 10, shifted radially by 57 units
    radius = 3
    r_shift = 1
    z_shift = 0
    volume, surface, extent = generate_sphere_volume_surface_matrices(radius, (z_shift, r_shift), bins = 1)

    # Plot volume kernel
    fig, ax = plt.subplots()
    im = ax.imshow(volume.T, extent=extent, origin="lower", cmap="Blues_r")
    cbar = plt.colorbar(im, shrink=0.6)
    cbar.set_label("Volume")
    ax.set_xlabel("$r$")
    ax.set_ylabel("$z$")
    fig.show()

    # Plot surface kernel
    fig, ax = plt.subplots()
    im = ax.imshow(surface.T, extent=extent, origin="lower", cmap="Blues_r")
    cbar = plt.colorbar(im, shrink=0.6)
    cbar.set_label("Surface")
    ax.set_xlabel("$r$")
    ax.set_ylabel("$z$")
    fig.show()

    # # Combined RGB visualization (surface in G, volume in B, difference in R)
    # shape = *np.shape(surface), 3
    # img = np.zeros(shape)
    # img[:, :, 1] = surface / np.max(surface)
    # img[:, :, 2] = volume / np.max(volume)
    # img[:, :, 0] = np.abs(volume / np.max(volume) - surface / np.max(surface))
    # fig, ax = plt.subplots()
    # im = ax.imshow(img.swapaxes(0, 1), origin="lower")
    # cbar.set_label("Surface and volume")
    # ax.set_xlabel("$r$")
    # ax.set_ylabel("$z$")
    # fig.show()
    # input()
# %%
