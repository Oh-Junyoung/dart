import os
import numpy as np
import matplotlib.pyplot as plt
import common_utils as utils

def process_component(title, t_ref, y_ref, t_sim, y_sim, unit, out_name):
    shift = utils.find_best_shift(t_sim, y_sim, t_ref, y_ref)
    utils.plot_comparison(title, t_ref, y_ref, "Ref", t_sim, y_sim, "Sim", 
                          f"Value ({unit})", out_name, shift)
    return shift

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.abspath(os.path.join(base_dir, "../../ref_data"))
    sim_dir = os.path.abspath(os.path.join(base_dir, "../../sim_data"))
    result_dir = base_dir
    # if not os.path.exists(result_dir): os.makedirs(result_dir)

    print("\n--- J2 Comparison ---")
    START_TRIM = 0.5
    
    # 1. Position X, Y
    REF_DT = 1.0/24.0
    t_rx, rx = utils.load_csv_data(os.path.join(ref_dir, "J2_Pos_X_mm.csv"), REF_DT)
    t_ry, ry = utils.load_csv_data(os.path.join(ref_dir, "J2_Pos_Y_mm.csv"), REF_DT)
    t_sx, sx = utils.load_csv_data(os.path.join(sim_dir, "J2_Pos_X_mm.csv"))
    t_sy, sy = utils.load_csv_data(os.path.join(sim_dir, "J2_Pos_Y_mm.csv"))
    
    # Trim Sim
    if t_sx is not None: t_sx, sx = utils.trim_data(t_sx, sx, START_TRIM)
    if t_sy is not None: t_sy, sy = utils.trim_data(t_sy, sy, START_TRIM)
    
    # Normalize Sim Time to start at 0
    if t_sx is not None and len(t_sx) > 0: t_sx -= t_sx[0]
    if t_sy is not None and len(t_sy) > 0: t_sy -= t_sy[0]

    if all(v is not None for v in [t_rx, rx, t_sx, sx]):
        process_component("J2 Pos X", t_rx, rx, t_sx, sx, "mm", os.path.join(result_dir, "J2_Pos_X_Comparison.png"))
    if all(v is not None for v in [t_ry, ry, t_sy, sy]):
        process_component("J2 Pos Y", t_ry, ry, t_sy, sy, "mm", os.path.join(result_dir, "J2_Pos_Y_Comparison.png"))
        
    # 2. Velocity X, Y
    t_rvx, rvx = utils.load_csv_data(os.path.join(ref_dir, "J2_Vel_x.csv"))
    t_rvy, rvy = utils.load_csv_data(os.path.join(ref_dir, "J2_Vel_y.csv"))
    t_svx, svx = utils.load_csv_data(os.path.join(sim_dir, "J2_Vel_x.csv"))
    t_svy, svy = utils.load_csv_data(os.path.join(sim_dir, "J2_Vel_y.csv"))
    
    if t_svx is not None: t_svx, svx = utils.trim_data(t_svx, svx, START_TRIM)
    if t_svy is not None: t_svy, svy = utils.trim_data(t_svy, svy, START_TRIM)
    
    # Normalize Sim Time
    if t_svx is not None and len(t_svx) > 0: t_svx -= t_svx[0]
    if t_svy is not None and len(t_svy) > 0: t_svy -= t_svy[0]
    
    if all(v is not None for v in [t_rvx, rvx, t_svx, svx]):
        process_component("J2 Vel X", t_rvx, rvx, t_svx, svx, "mm/s", os.path.join(result_dir, "J2_Vel_X_Comparison.png"))
    if all(v is not None for v in [t_rvy, rvy, t_svy, svy]):
        process_component("J2 Vel Y", t_rvy, rvy, t_svy, svy, "mm/s", os.path.join(result_dir, "J2_Vel_Y_Comparison.png"))

    # 3. Reaction Force (Magnitude Only)
    t_rFx, rFx = utils.load_csv_data(os.path.join(ref_dir, "J2_Reaction_x.csv"))
    t_rFy, rFy = utils.load_csv_data(os.path.join(ref_dir, "J2_Reaction_y.csv"))
    t_sFx, sFx = utils.load_csv_data(os.path.join(sim_dir, "J2_Reaction_x.csv"))
    t_sFy, sFy = utils.load_csv_data(os.path.join(sim_dir, "J2_Reaction_y.csv"))
    
    if t_sFx is not None: t_sFx, sFx = utils.trim_data(t_sFx, sFx, START_TRIM)
    if t_sFy is not None: t_sFy, sFy = utils.trim_data(t_sFy, sFy, START_TRIM)

    # Normalize Sim Time
    if t_sFx is not None and len(t_sFx) > 0: t_sFx -= t_sFx[0]
    if t_sFy is not None and len(t_sFy) > 0: t_sFy -= t_sFy[0]
    
    if all(v is not None for v in [t_rFx, rFx, t_rFy, rFy, t_sFx, sFx, t_sFy, sFy]):
        # Ignore X/Y components as they may be Local Frame in Ref
        # process_component("J2 Reaction Force X", ...) 
        # process_component("J2 Reaction Force Y", ...)
        
        # Compute Magnitude (Use X's time)
        rMag = np.sqrt(rFx**2 + rFy**2)
        sMag = np.sqrt(sFx**2 + sFy**2)
        process_component("J2 Reaction Force (Magnitude)", t_rFx, rMag, t_sFx, sMag, "N", os.path.join(result_dir, "J2_Reaction_Comparison.png"))

if __name__ == "__main__":
    main()
