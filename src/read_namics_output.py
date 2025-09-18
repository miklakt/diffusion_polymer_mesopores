import numpy as np
def extract_volume_fractions_per_grafting_range(df, value_cols=None
):
    # cylindrical axes
    r_vals = np.sort(df["x"].unique())
    z_vals = np.sort(df["y"].unique())
    
    # which value columns to transform 2d arrays
    if value_cols is None:
        value_cols = [c for c in df.columns if c not in ["x", "y"]]
        
    
    # pivot each column into a grid
    grids = {}
    for col in value_cols:
        pivoted = df.pivot(index="y", columns="x", values=col)
        # ensure correct ordering
        Z = pivoted.loc[z_vals, r_vals].to_numpy()
        grids[col] = Z

    import re
    pattern = re.compile(r"mol_pol(\d+)_phi")
    mol_keys = sorted(
        (int(pattern.match(k).group(1)), k)
        for k in grids if pattern.match(k)
    )

    phis = []
    for _, k in mol_keys:
        phi = grids[k]
        phis.append(phi)
    
    phis = np.array(phis)

    z_vals = z_vals - (np.max(z_vals)-np.min(z_vals))/2 #make pore canter at z=0
    
    return r_vals, z_vals, phis

def generate_circle_kernel(d):
    radius = d/2
    a = np.zeros((d, d), dtype =bool)
    radius2 = radius**2
    for i in range(d):
        for j in range(d):
            distance2 = (radius-i-0.5)**2 + (radius-j-0.5)**2
            if distance2<radius2:
                a[i,j] = True
    return a

def create_walls(xlayers, ylayers, pore_radius, pore_length, colloid_diameter = None):
    from scipy import ndimage
    W_arr = np.zeros((ylayers,xlayers), dtype = bool)
    W_arr[int(ylayers/2-pore_length/2):int(ylayers/2+pore_length/2+1), pore_radius:] = True
    if colloid_diameter is not None:
        W_arr = ndimage.binary_dilation(W_arr, structure = generate_circle_kernel(d))
    return W_arr

# %%
if __name__ == "__main__":
    import pandas as pd
    from make_plot import plot_with_slider
    raw = pd.read_table("SCF/output/empty_pore_input.pro", index_col=False)
    R, Z, Phis = extract_volume_fractions_per_grafting_range(raw)
    pore_radius = 26
    pore_length = 52
    walls = create_walls(len(R), len(Z), pore_radius, pore_length)
    fig, ax = plot_with_slider(Z, R, Phis.transpose(0,2,1), fix_clim=True, walls=walls.T)
    fig.show()
# %%

