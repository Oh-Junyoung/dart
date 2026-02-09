import os
import numpy as np
import common_utils as utils

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(os.path.join(base_dir, "../../"))
    ref_path = os.path.join(data_root, "ref_data", "J1_AngularVelocity.csv")
    sim_path = os.path.join(data_root, "sim_data", "J1_Speed_RPM.csv")
    result_dir = base_dir # Save in current 'old' dir
    # result_dir = os.path.join(base_dir, "result")
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    # Load
    t_ref, y_ref_deg = utils.load_csv_data(ref_path) # deg/s
    t_sim, y_sim_rpm = utils.load_csv_data(sim_path) # RPM
    
    if t_ref is None or t_sim is None: return

    # Trim Startup Transients
    START_TRIM = 0.5
    t_sim, y_sim_rpm = utils.trim_data(t_sim, y_sim_rpm, START_TRIM)
    # Re-zero Sim Time
    if len(t_sim) > 0: t_sim -= t_sim[0]

    # Convert Ref deg/s to RPM for comparison
    # 1 RPM = 6 deg/s
    y_ref_rpm = y_ref_deg / 6.0
    
    print("--- J1 Angular Velocity Comparison ---")
    shift = utils.find_best_shift(t_sim, y_sim_rpm, t_ref, y_ref_rpm)
    print(f"Optimal Shift: {shift:.4f}s")
    
    out_path = os.path.join(result_dir, "J1_Comparison.png")
    utils.plot_comparison("J1 Angular Velocity", 
                          t_ref, y_ref_rpm, "Ref (Converted to RPM)", 
                          t_sim, y_sim_rpm, "Sim (RPM)", 
                          "Speed (RPM)", out_path, shift)

    # --- Torque Comparison ---
    ref_torque_path = os.path.join(data_root, "ref_data", "Motor_Torque_.csv")
    sim_torque_path = os.path.join(data_root, "sim_data", "Motor_Torque.csv")
    
    t_ref_tq, y_ref_tq_nmm = utils.load_csv_data(ref_torque_path)
    t_sim_tq, y_sim_tq_nm = utils.load_csv_data(sim_torque_path)
    
    if t_ref_tq is not None and t_sim_tq is not None:
        # Trim Sim Torque
        t_sim_tq, y_sim_tq_nm = utils.trim_data(t_sim_tq, y_sim_tq_nm, START_TRIM)
        # Re-zero Sim Time
        if len(t_sim_tq) > 0: t_sim_tq -= t_sim_tq[0]
        
        # Convert Ref Nmm -> Nm
        # Check Ref Magnitude:
        print(f"Ref Torque Data (First 5): {y_ref_tq_nmm[:5]}")
        # Validating Ref Unit: If mean is ~300, and Sim is ~100. 
        # If Ref is Nmm, 0.3 Nm. If Ref is Nm, 300 Nm.
        # User says Sim (previously 100k) was kN (100k Nm?? No, 100k Nmm).
        # We now save Sim as Nm -> Expect ~100.
        # If Ref is 300 (Nmm), then Sim(100) vs Ref(0.3).
        # If Ref is 300 (Nm), then Sim(100) vs Ref(300). Closer.
        # I will assume Ref header 'newton-mm' might be right (small torque?) OR wrong.
        # Let's trust header provided by user as 'Nmm' for conversion first. 
        # But if the plot is way off, we know.
        y_ref_tq_nm = y_ref_tq_nmm / 1000.0  # Nmm -> Nm
        
        print("--- J1 Torque Comparison ---")
        print(f"Sim Torque Mean (Nm): {np.mean(y_sim_tq_nm):.2f}")
        print(f"Ref Torque Mean (Nm): {np.mean(y_ref_tq_nm):.2f}")
        
        out_path_tq = os.path.join(result_dir, "J1_Torque_Comparison.png")
        utils.plot_comparison("J1 Motor Torque",
                              t_ref_tq, y_ref_tq_nm, "Ref (Converted to Nm)",
                              t_sim_tq, y_sim_tq_nm, "Sim (Nm)",
                              "Torque (Nm)", out_path_tq, shift)

if __name__ == "__main__":
    main()
