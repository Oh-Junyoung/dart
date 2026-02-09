import os
import numpy as np
import common_utils as utils

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.abspath(os.path.join(base_dir, "../../ref_data"))
    sim_dir = os.path.abspath(os.path.join(base_dir, "../../sim_data"))
    result_dir = base_dir
    # if not os.path.exists(result_dir): os.makedirs(result_dir)

    print("\n--- J4 Comparison ---")
    
    print("\n--- J4 Comparison ---")
    
    START_TRIM = 0.5
    START_TRIM_INDICES = int(START_TRIM * 120) # Approx or use time masking
    # Better to use utils.trim_data

    # Load J4 Data
    t_rFx, rFx = utils.load_csv_data(os.path.join(ref_dir, "J4_Reaction_x.csv"))
    t_rFy, rFy = utils.load_csv_data(os.path.join(ref_dir, "J4_Reaction_y.csv"))
    t_sFx, sFx = utils.load_csv_data(os.path.join(sim_dir, "J4_Reaction_x.csv"))
    t_sFy, sFy = utils.load_csv_data(os.path.join(sim_dir, "J4_Reaction_y.csv"))
    
    # Load J3 Data (For Alignment)
    t_r3x, r3x = utils.load_csv_data(os.path.join(ref_dir, "J3_Reaction_x.csv"))
    t_r3y, r3y = utils.load_csv_data(os.path.join(ref_dir, "J3_Reaction_y.csv"))
    t_s3x, s3x = utils.load_csv_data(os.path.join(sim_dir, "J3_Reaction_x.csv"))
    t_s3y, s3y = utils.load_csv_data(os.path.join(sim_dir, "J3_Reaction_y.csv"))

    # Trim Sim (J4)
    if t_sFx is not None: t_sFx, sFx = utils.trim_data(t_sFx, sFx, START_TRIM)
    if t_sFy is not None: t_sFy, sFy = utils.trim_data(t_sFy, sFy, START_TRIM)
    # Re-zero Sim (J4)
    if t_sFx is not None and len(t_sFx) > 0: t_sFx -= t_sFx[0]
    if t_sFy is not None and len(t_sFy) > 0: t_sFy -= t_sFy[0]

    # Trim Sim (J3)
    if t_s3x is not None: t_s3x, s3x = utils.trim_data(t_s3x, s3x, START_TRIM)
    if t_s3y is not None: t_s3y, s3y = utils.trim_data(t_s3y, s3y, START_TRIM)
    # Re-zero Sim (J3)
    if t_s3x is not None and len(t_s3x) > 0: t_s3x -= t_s3x[0]
    if t_s3y is not None and len(t_s3y) > 0: t_s3y -= t_s3y[0]

    if all(v is not None for v in [t_rFx, rFx, t_rFy, rFy, t_sFx, sFx, t_sFy, sFy,
                                   t_r3x, r3x, t_r3y, r3y, t_s3x, s3x, t_s3y, s3y]):
        
        # Calculate Magnitude for J3 (Reference for shift)
        r3Mag = np.sqrt(r3x**2 + r3y**2)
        s3Mag = np.sqrt(s3x**2 + s3y**2)
        
        print("Calculating Master Shift from J3 Reaction Force...")
        # Use J3 Mag to find shift
        shift = utils.find_best_shift(t_s3x, s3Mag, t_r3x, r3Mag)
        print(f"Master Shift (from J3): {shift:.4f}s")

        # Compute Magnitude J4
        rMag = np.sqrt(rFx**2 + rFy**2)
        sMag = np.sqrt(sFx**2 + sFy**2)
        
        # Plot J4 using Master Shift
        utils.plot_comparison("J4 Reaction Force (Magnitude)", t_rFx, rMag, "Ref", t_sFx, sMag, "Sim", "Force (N)",
                              os.path.join(result_dir, "J4_Reaction_Comparison.png"), shift)

if __name__ == "__main__":
    main()
