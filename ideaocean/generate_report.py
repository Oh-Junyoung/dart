import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add current directory to path to import common_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import common_utils as utils

def get_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    sim_dir = os.path.join(data_dir, "sim_data")
    ref_dir = os.path.join(data_dir, "ref_data")
    result_dir = os.path.join(data_dir, "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return sim_dir, ref_dir, result_dir

def plot_single_ax(ax, t_sim, y_sim, label_sim, t_ref=None, y_ref=None, label_ref=None, 
                  ylabel="", title="", color_sim='r', color_ref='k', shift=0.0):
    
    if t_ref is not None and y_ref is not None:
        ax.plot(t_ref, y_ref, linestyle='--', color=color_ref, label=f"{label_ref}", alpha=0.7)
    
    if t_sim is not None and y_sim is not None:
        # t_sim is already absolute time (e.g. 0.5 ~ 3.0)
        # shift is forced to 0.0 in main logic usually
        t_sim_aligned = t_sim + shift
        ax.plot(t_sim_aligned, y_sim, linestyle='-', color=color_sim, label=label_sim)
        
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)
    if label_ref or label_sim:
        ax.legend(loc='upper right', fontsize='small')

def plot_with_error(fig, outer_gs_pos, t_sim, y_sim, label_sim, t_ref=None, y_ref=None, label_ref=None, 
                    ylabel="", title="", color_sim='r', color_ref='k', shift=0.0):
    
    if t_ref is None or y_ref is None:
        ax = plt.subplot(outer_gs_pos)
        plot_single_ax(ax, t_sim, y_sim, label_sim, None, None, None, ylabel, title, color_sim, color_ref, shift)
        return ax
    
    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs_pos, height_ratios=[3, 1], hspace=0.1)
    
    ax_main = plt.subplot(inner_gs[0])
    ax_err = plt.subplot(inner_gs[1], sharex=ax_main)
    
    # Main Plot
    plot_single_ax(ax_main, t_sim, y_sim, label_sim, t_ref, y_ref, label_ref, ylabel, title, color_sim, color_ref, shift)
    plt.setp(ax_main.get_xticklabels(), visible=False)
    
    # Error Plot
    t_sim_aligned = t_sim + shift
    
    # Interpolate Ref at t_sim_aligned points
    # Ref domain: 0.0 ~ 3.0
    # Sim domain: 0.5 ~ 3.0
    # Interp works correctly in overlapping region
    y_ref_interp = np.interp(t_sim_aligned, t_ref, y_ref, left=np.nan, right=np.nan)
    error = y_sim - y_ref_interp
    
    valid = ~np.isnan(error)
    if np.any(valid):
        # Subset identifying valid comparison points
        err_v = error[valid]
        ref_v = y_ref_interp[valid]

        # RMSE & MAE
        mse = np.mean(err_v**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(err_v))

        # NRMSE (Normalized by Range)
        ref_range = np.max(ref_v) - np.min(ref_v) if len(ref_v) > 0 else 0
        nrmse = (rmse / ref_range) * 100.0 if ref_range > 1e-9 else 0.0

        # R-squared
        ss_res = np.sum(err_v**2)
        ss_tot = np.sum((ref_v - np.mean(ref_v))**2) if len(ref_v) > 0 else 0
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-9 else (1.0 if ss_res < 1e-9 else 0.0)
        
        ax_err.plot(t_sim_aligned, error, 'b-', linewidth=1)
        ax_err.axhline(0, color='k', linestyle=':', alpha=0.5)
        ax_err.set_ylabel("Err")
        ax_err.grid(True, alpha=0.5)
        
        text = f"NRMSE: {nrmse:.1f}%\nMAE: {mae:.2f}\nR²: {r2:.4f}"
        ax_err.text(0.95, 0.05, text, transform=ax_err.transAxes, ha='right', va='bottom', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    return ax_main

def process_j1(sim_dir, ref_dir, result_dir):
    print("Processing J1 Summary...")
    START_TRIM = 0.5
    
    # 1. Speed
    t_ref_spd, y_ref_spd_deg = utils.load_csv_data(os.path.join(ref_dir, "J1_AngularVelocity.csv"))
    t_sim_spd, y_sim_spd = utils.load_csv_data(os.path.join(sim_dir, "J1_Speed_RPM.csv"))
    
    if y_ref_spd_deg is not None:
        y_ref_spd = y_ref_spd_deg / 6.0 
    else:
        y_ref_spd = None
    
    t_sim_spd, y_sim_spd = utils.trim_data(t_sim_spd, y_sim_spd, START_TRIM)
    # REMOVED: t -= t[0]
    
    shift = 0.0

    t_ref_tq, y_ref_tq_nmm = utils.load_csv_data(os.path.join(ref_dir, "Motor_Torque_.csv"))
    t_sim_tq, y_sim_tq = utils.load_csv_data(os.path.join(sim_dir, "Motor_Torque.csv"))
    
    if y_ref_tq_nmm is not None:
         y_ref_tq = y_ref_tq_nmm / 1000.0
    else:
         y_ref_tq = None
    
    t_sim_tq, y_sim_tq = utils.trim_data(t_sim_tq, y_sim_tq, START_TRIM)
    # REMOVED: t -= t[0]

    t_sim_p, y_sim_p = utils.load_csv_data(os.path.join(sim_dir, "J1_Power_W.csv"))
    t_sim_p, y_sim_p = utils.trim_data(t_sim_p, y_sim_p, START_TRIM)
    # REMOVED: t -= t[0]

    fig = plt.figure(figsize=(10, 14))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.8])
    fig.suptitle('J1 (Crank) Analysis Summary', fontsize=16)

    plot_with_error(fig, gs[0], t_sim_spd, y_sim_spd, "Sim", t_ref_spd, y_ref_spd, "Ref", 
                    "Speed (RPM)", "Motor Speed", shift=shift)

    plot_with_error(fig, gs[1], t_sim_tq, y_sim_tq, "Sim", t_ref_tq, y_ref_tq, "Ref", 
                    "Torque (Nm)", "Motor Torque", shift=shift)

    ax2 = plt.subplot(gs[2])
    plot_single_ax(ax2, t_sim_p, y_sim_p, "Sim (Abs)", None, None, None, 
                   "Power (W)", "Motor Power Consumption", color_sim='g', shift=shift)
    ax2.set_xlabel("Time (s)")

    plt.tight_layout()
    out_path = os.path.join(result_dir, "J1_Summary.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")
    return shift

def process_j2(sim_dir, ref_dir, result_dir):
    print("Processing J2 Summary...")
    START_TRIM = 0.5
    REF_DT_POS = 1.0/24.0
    
    def get_data(name, load_dt=None):
        tr, yr = utils.load_csv_data(os.path.join(ref_dir, name), load_dt if load_dt else 1/120.0)
        ts, ys = utils.load_csv_data(os.path.join(sim_dir, name))
        if ts is not None:
            ts, ys = utils.trim_data(ts, ys, START_TRIM)
            # REMOVED: t -= t[0]
        return tr, yr, ts, ys

    trx, yrx, tsx, ysx = get_data("J2_Pos_X_mm.csv", REF_DT_POS)
    t_ry, yry, tsy, ysy = get_data("J2_Pos_Y_mm.csv", REF_DT_POS)
    trvx, yrvx, tsvx, ysvx = get_data("J2_Vel_x.csv")
    trvy, yrvy, tsvy, ysvy = get_data("J2_Vel_y.csv")
    trfx, yrfx, tsfx, ysfx = get_data("J2_Reaction_x.csv")
    trfy, yrfy, tsfy, ysfy = get_data("J2_Reaction_y.csv")
    
    ts_fmag, ys_fmag = None, None
    tr_fmag, yr_fmag = None, None
    if ysfx is not None:
        ys_fmag = np.sqrt(ysfx**2 + ysfy**2)
        ts_fmag = tsfx
    if yrfx is not None and yrfy is not None:
        yr_fmag = np.sqrt(yrfx**2 + yrfy**2)
        tr_fmag = trfx 

    shift = 0.0

    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(3, 2)
    fig.suptitle('J2 (Coupler) Analysis Summary', fontsize=16)

    plot_with_error(fig, gs[0,0], tsx, ysx, "Sim X", trx, yrx, "Ref X", "Pos X (mm)", "Position X", shift=shift)
    plot_with_error(fig, gs[0,1], tsy, ysy, "Sim Y", t_ry, yry, "Ref Y", "Pos Y (mm)", "Position Y", shift=shift)
    plot_with_error(fig, gs[1,0], tsvx, ysvx, "Sim Vx", trvx, yrvx, "Ref Vx", "Vel X (mm/s)", "Velocity X", shift=shift)
    plot_with_error(fig, gs[1,1], tsvy, ysvy, "Sim Vy", trvy, yrvy, "Ref Vy", "Vel Y (mm/s)", "Velocity Y", shift=shift)
    plot_with_error(fig, gs[2,:], ts_fmag, ys_fmag, "Sim |F|", tr_fmag, yr_fmag, "Ref |F|", "Force (N)", "Reaction Force Magnitude", shift=shift)

    plt.tight_layout()
    out_path = os.path.join(result_dir, "J2_Summary.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")
    return shift

def process_j3(sim_dir, ref_dir, result_dir):
    print("Processing J3 Summary...")
    START_TRIM = 0.5
    REF_DT_POS = 1.0/24.0
    
    def get_data(name, load_dt=None):
        tr, yr = utils.load_csv_data(os.path.join(ref_dir, name), load_dt if load_dt else 1/120.0)
        ts, ys = utils.load_csv_data(os.path.join(sim_dir, name))
        if ts is not None:
            ts, ys = utils.trim_data(ts, ys, START_TRIM)
            # REMOVED: t -= t[0]
        return tr, yr, ts, ys

    trx, yrx, tsx, ysx = get_data("J3_Pos_X_mm.csv", REF_DT_POS)
    t_ry, yry, tsy, ysy = get_data("J3_Pos_Y_mm.csv", REF_DT_POS)
    trvx, yrvx, tsvx, ysvx = get_data("J3_Vel_x.csv")
    trvy, yrvy, tsvy, ysvy = get_data("J3_Vel_y.csv")
    trfx, yrfx, tsfx, ysfx = get_data("J3_Reaction_x.csv")
    trfy, yrfy, tsfy, ysfy = get_data("J3_Reaction_y.csv")
    tsi, ysi, ts_mu, ys_mu = get_data("J3_Trans_Angle_deg.csv")

    ts_fmag, ys_fmag = None, None
    tr_fmag, yr_fmag = None, None
    if ysfx is not None:
        ys_fmag = np.sqrt(ysfx**2 + ysfy**2)
        ts_fmag = tsfx
    if yrfx is not None:
        yr_fmag = np.sqrt(yrfx**2 + yrfy**2)
        tr_fmag = trfx

    shift = 0.0

    fig = plt.figure(figsize=(14, 20))
    gs = gridspec.GridSpec(4, 2)
    fig.suptitle('J3 (Rocker) Analysis Summary', fontsize=16)

    plot_with_error(fig, gs[0,0], tsx, ysx, "Sim X", trx, yrx, "Ref X", "Pos X (mm)", "Position X", shift=shift)
    plot_with_error(fig, gs[0,1], tsy, ysy, "Sim Y", t_ry, yry, "Ref Y", "Pos Y (mm)", "Position Y", shift=shift)
    plot_with_error(fig, gs[1,0], tsvx, ysvx, "Sim Vx", trvx, yrvx, "Ref Vx", "Vel X (mm/s)", "Velocity X", shift=shift)
    plot_with_error(fig, gs[1,1], tsvy, ysvy, "Sim Vy", trvy, yrvy, "Ref Vy", "Vel Y (mm/s)", "Velocity Y", shift=shift)
    plot_with_error(fig, gs[2,:], ts_fmag, ys_fmag, "Sim |F|", tr_fmag, yr_fmag, "Ref |F|", "Force (N)", "Reaction Force Magnitude", shift=shift)

    ax_mu = plt.subplot(gs[3,:])
    plot_single_ax(ax_mu, ts_mu, ys_mu, "Sim Mu", None, None, None, "Angle (deg)", "Transmission Angle", color_sim='purple', shift=shift)
    ax_mu.axhline(40, color='r', linestyle='--', linewidth=1)
    ax_mu.axhline(140, color='r', linestyle='--', linewidth=1)
    ax_mu.axhspan(40, 140, color='g', alpha=0.1, label='Good Zone')
    ax_mu.set_xlabel("Time (s)")

    plt.tight_layout()
    out_path = os.path.join(result_dir, "J3_Summary.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def process_j4(sim_dir, ref_dir, result_dir, master_shift=0.0):
    print("Processing J4 Summary...")
    START_TRIM = 0.5
    
    def get_data(name):
        tr, yr = utils.load_csv_data(os.path.join(ref_dir, name))
        ts, ys = utils.load_csv_data(os.path.join(sim_dir, name))
        if ts is not None:
            ts, ys = utils.trim_data(ts, ys, START_TRIM)
            # REMOVED: t -= t[0]
        return tr, yr, ts, ys

    trfx, yrfx, tsfx, ysfx = get_data("J4_Reaction_x.csv")
    trfy, yrfy, tsfy, ysfy = get_data("J4_Reaction_y.csv")
    tsi, ysi, ts_shaking, ys_shaking = get_data("J4_Shaking_Force_N.csv")

    ts_fmag, ys_fmag = None, None
    tr_fmag, yr_fmag = None, None
    if ysfx is not None:
        ys_fmag = np.sqrt(ysfx**2 + ysfy**2)
        ts_fmag = tsfx
    if yrfx is not None:
        yr_fmag = np.sqrt(yrfx**2 + yrfy**2)
        tr_fmag = trfx

    shift = 0.0

    fig = plt.figure(figsize=(10, 14))
    gs = gridspec.GridSpec(2, 1)
    fig.suptitle('J4 (Ground Pivot) Analysis Summary', fontsize=16)

    plot_with_error(fig, gs[0], ts_fmag, ys_fmag, "Sim |F|", tr_fmag, yr_fmag, "Ref |F|", 
                    "Force (N)", "Reaction Force Magnitude", shift=shift)

    ax_shaking = plt.subplot(gs[1])
    plot_single_ax(ax_shaking, ts_shaking, ys_shaking, "Sim Shaking Force", None, None, None, 
                   "Force (N)", "Shaking Force Magnitude (|F_total|)", color_sim='brown', shift=shift)
    ax_shaking.set_xlabel("Time (s)")

    plt.tight_layout()
    out_path = os.path.join(result_dir, "J4_Summary.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    sim_dir, ref_dir, result_dir = get_paths()
    print(f"Generating consolidated reports in: {result_dir}")
    
    shift_j1 = process_j1(sim_dir, ref_dir, result_dir)
    print(f"Master Shift (J1): {shift_j1}")
    
    process_j2(sim_dir, ref_dir, result_dir)
    process_j3(sim_dir, ref_dir, result_dir)
    process_j4(sim_dir, ref_dir, result_dir)
    
    print("Done.")

if __name__ == "__main__":
    main()
