import numpy as np
import matplotlib
matplotlib.use('Agg') # Force headless backend
import matplotlib.pyplot as plt
import csv
import os

def load_csv_data(filepath, dt_default=1/120.0):
    """
    Load CSV data robustly handling encodings.
    Returns (time_array, value_array).
    """
    print(f"Loading: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: File not found - {filepath}")
        return None, None

    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
    data = None
    
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                reader = csv.reader(f)
                temp_t, temp_v = [], []
                for row in reader:
                    if not row: continue
                    # Filter empty strings (e.g. from trailing commas)
                    row = [c for c in row if c.strip()]
                    
                    try:
                        if len(row) >= 2:
                            t = float(row[0])
                            v = float(row[1])
                            temp_t.append(t)
                            temp_v.append(v)
                        elif len(row) == 1:
                            v = float(row[0])
                            temp_v.append(v)
                    except ValueError:
                        continue 

                if len(temp_v) > 0:
                    if len(temp_t) == 0:
                        temp_t = [i * dt_default for i in range(len(temp_v))]
                    
                    data = np.column_stack((temp_t, temp_v))
                    data = data[data[:, 0].argsort()] # Sort by time
                    break 
        except UnicodeDecodeError:
            continue
            
    if data is None or len(data) == 0:
        print(f"Failed to load data from {filepath} (or file empty)")
        return None, None
        
    return data[:, 0], data[:, 1]

def find_best_shift(t_base, y_base, t_target, y_target, search_range=0.5):
    """
    Find time shift 's' such that y_target(t+s) matches y_base(t).
    """
    shifts = np.linspace(-search_range, search_range, 201) 
    best_rmse = float('inf')
    best_shift = 0.0
    
    for s in shifts:
        # Shift target time: t_shifted = t_target + s
        # Interp target at t_base
        y_target_interp = np.interp(t_base, t_target + s, y_target)
        rmse = np.sqrt(np.mean((y_base - y_target_interp)**2))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = s
    return best_shift

def calculate_metrics(y_true, y_pred):
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    return mae, rmse, err

def plot_comparison(title, t_ref, y_ref, label_ref, t_sim, y_sim, label_sim, 
                   y_label, output_path, shift_val=0.0):
    
    # 1. Align time for error calc
    t_ref_shifted = t_ref + shift_val
    y_ref_interp = np.interp(t_sim, t_ref_shifted, y_ref)
    mae, rmse, err = calculate_metrics(y_sim, y_ref_interp)
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(f"{title}\n(Shift Applied: {shift_val:.4f}s)", fontsize=14)
    
    # Comparison
    axs[0].plot(t_ref_shifted, y_ref, 'k--', label=f'{label_ref} (Shifted)', alpha=0.7)
    axs[0].plot(t_sim, y_sim, 'r-', label=label_sim)
    axs[0].set_ylabel(y_label)
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title('Comparison')

    # Error
    axs[1].plot(t_sim, err, 'b-')
    axs[1].axhline(0, color='k', linestyle=':', alpha=0.5)
    axs[1].set_ylabel(f'Error ({y_label.split("(")[-1][:-1]})') # Extract unit
    axs[1].set_xlabel('Time (s)')
    axs[1].set_title(f'Error (Sim - Ref)\nMAE: {mae:.4f}, RMSE: {rmse:.4f}')
    axs[1].grid(True)
    
    # Enforce positive Time Axis
    axs[1].set_xlim(left=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    print(f"{title} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    plt.close()

def trim_data(t, y, t_min):
    """
    Trim data to ignore start up transients.
    """
    mask = t >= t_min
    return t[mask], y[mask]
