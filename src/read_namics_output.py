from pathlib import Path
import pandas as pd
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
        phis.append(phi.T)
    
    phis = np.array(phis)

    z_vals = z_vals - (np.max(z_vals)-np.min(z_vals))/2 #make pore canter at z=0

    result = {
        "R":r_vals,
        "Z":z_vals,
        "phi_s":phis 
    }

    if "mon_C_phi" in grids:
        result.update({"colloid_mask":grids["mon_C_phi"].T})
    
    return result

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

def build_scf_results(
    basename: str,
    colloid_positions,
    chi_PS: float,
    chi_PC: float,
    colloid_diameter: float,
    *,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Build SCF_results for a given simulation basename.

    Parameters
    ----------
    basename : str
        Path + file prefix without extension. Example: "SCF/output/colloid_traversing_pore".
        This function will read:
          - f"{basename}.kal"
          - f"{basename}_*.pro"
    colloid_positions : sequence of int/float
        Positions used to label rows (aligned to the order in the .kal / *_N.pro files).
    chi_PS, chi_PC : float
        Chi parameters to attach as scalar columns.
    colloid_diameter : float
        Colloid diameter to attach as a scalar column.
    verbose : bool
        If True, prints a small log.

    Returns
    -------
    pd.DataFrame
        SCF_results dataframe with columns:
          - colloid_position
          - free_energy (ground-state corrected)
          - (optionally) phi and any fields from extract_volume_fractions_per_grafting_range
          - chi_PS, chi_PC, colloid_diameter
    """
    base = Path(basename)
    kal_path = base.with_suffix(".kal")
    parent = base.parent
    prefix = base.name  # e.g., "colloid_traversing_pore"

    if not kal_path.exists():
        raise FileNotFoundError(f"Could not find {kal_path}")

    # --- Read scalar results (.kal) ---
    df_kal = pd.read_table(kal_path, index_col=False)

    # Normalize free energy column naming
    if "sys_variable_free_energy" in df_kal.columns:
        df_kal = df_kal.rename(columns={"sys_variable_free_energy": "free_energy"})
    elif "free_energy" not in df_kal.columns:
        raise KeyError("Could not find 'sys_variable_free_energy' or 'free_energy' in .kal file")

    # Attach colloid positions (truncate to the number of rows we currently have)
    if len(colloid_positions) < len(df_kal):
        if verbose:
            print(f"[build_scf_results] Warning: {len(df_kal)} rows in .kal, "
                  f"but only {len(colloid_positions)} positions; truncating .kal to match positions.")
        df_kal = df_kal.iloc[:len(colloid_positions)].copy()

    df_kal = df_kal.copy()
    df_kal["colloid_position"] = list(colloid_positions)[:len(df_kal)]

    # Keep minimal scalar info for plotting + any useful identifiers if present
    keep_cols = ["colloid_position", "free_energy"]
    for extra in ("R", "Z"):
        if extra in df_kal.columns:
            keep_cols.append(extra)
    df_scalar = df_kal[keep_cols].copy()

    # --- Ground state correction on free energy ---
    # Use the row where |colloid_position| is maximal as reference (ground state)
    ref_idx = df_scalar["colloid_position"].abs().idxmax()
    ground_state_free_energy = df_scalar.loc[ref_idx, "free_energy"]
    df_scalar["free_energy"] = df_scalar["free_energy"] - ground_state_free_energy

    # --- Optional: read φ profiles (*.pro) and merge ---
    # Expect files like "<prefix>_1.pro", "<prefix>_2.pro", ... in the same directory
    pro_paths = sorted(parent.glob(f"{prefix}_*.pro"),
                       key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 10**9)

    profiles_records = []
    if len(pro_paths) == 0 and verbose:
        print(f"[build_scf_results] No profile files found matching {parent / (prefix + '_*.pro')} (this is OK).")

    for path in pro_paths:
        try:
            idx = int(path.stem.split("_")[-1]) - 1  # map to 0-based
        except ValueError:
            if verbose:
                print(f"[build_scf_results] Skipping unrecognized file name: {path.name}")
            continue

        if idx < 0 or idx >= len(colloid_positions):
            if verbose:
                print(f"[build_scf_results] Skipping {path.name}: index {idx} out of range.")
            continue

        if verbose:
            print(f"[build_scf_results] {path.name} -> colloid_position = {colloid_positions[idx]}")

        raw = pd.read_table(path, index_col=False)

        # You must provide this function in your environment.
        # It should return a dict-like structure with keys including "phi_s" (list/array per species)
        extracted = extract_volume_fractions_per_grafting_range(raw)

        # Collapse species-specific phi_s into total phi
        phi_total = np.sum(extracted["phi_s"], axis=0)

        rec = dict(extracted)  # shallow copy
        rec["colloid_position"] = colloid_positions[idx]
        rec["phi"] = phi_total
        # remove phi_s to avoid carrying large per-species arrays if not needed
        if "phi_s" in rec:
            del rec["phi_s"]

        profiles_records.append(rec)

    df_profiles = pd.DataFrame(profiles_records) if profiles_records else pd.DataFrame(columns=["colloid_position"])

    # --- Merge scalar and profile data on colloid_position ---
    if not df_profiles.empty:
        SCF_results = pd.merge(df_scalar, df_profiles, on="colloid_position", how="left")
    else:
        SCF_results = df_scalar

    # --- Attach constants ---
    SCF_results["chi_PS"] = chi_PS
    SCF_results["chi_PC"] = chi_PC
    SCF_results["colloid_diameter"] = colloid_diameter

    return SCF_results



def build_scf_empty_pore_results(
    basename: str,
    chi_PS: float,
    *,
    verbose: bool = False,
    keep_phi_s: bool = True
) -> dict:
    """
    Build 'empty pore' SCF results from a single .pro file.

    Parameters
    ----------
    basename : str
        Path + prefix without extension (e.g. "SCF/output/empty_pore").
        The function will read f"{basename}.pro".
    chi_PS : float
        χ_PS value to attach.
    return_df : bool
        If True, return a single-row DataFrame; else return a dict.
    verbose : bool
        Print brief status messages.

    Returns
    -------
    dict
        Contains aggregated 'phi' plus any other extracted fields,
        and a 'chi_PS' column/key.
    """
    pro_path = Path(basename).with_suffix(".pro")
    if not pro_path.exists():
        raise FileNotFoundError(f"Could not find profile file: {pro_path}")

    raw = pd.read_table(pro_path, index_col=False)

    # User-provided function expected to return a dict-like object
    extracted = extract_volume_fractions_per_grafting_range(raw)
    if "phi_s" not in extracted:
        raise KeyError(
            "extract_volume_fractions_per_grafting_range did not return 'phi_s'. "
            "Cannot compute total phi."
        )

    # Safely compute total phi without mutating the original dict
    phi_s = extracted["phi_s"]
    try:
        phi_total = np.sum(np.asarray(phi_s, dtype=float), axis=0)
    except Exception as e:
        raise ValueError(
            f"Failed to sum 'phi_s' into total 'phi'. Ensure it's a list/array of equal-length arrays. Error: {e}"
        )

    # Build result without side-effects
    result = {k: v for k, v in extracted.items() if k != "phi_s"}
    result["phi"] = phi_total
    result["chi_PS"] = chi_PS

    if keep_phi_s:
        result["phi_s"] = phi_s

    if verbose:
        print(f"[build_scf_empty_pore_results] Loaded {pro_path.name}, "
              f"phi length = {len(phi_total)}")

    return result


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

