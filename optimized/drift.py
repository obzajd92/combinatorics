import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_parameter_drift(healthy_path, crash_path):
    """
    Loads telemetry report logs from the baseline and crash optimization scenarios,
    extracts the top performing configurations, and quantifies the parameter drift.
    """
    if not os.path.exists(healthy_path) or not os.path.exists(crash_path):
        print("[!] Error: One or both required telemetry CSV log files are missing.")
        return
        
    # Load optimization logs
    df_healthy = pd.read_csv(healthy_path)
    df_crash = pd.read_csv(crash_path)
    
    # Filter for non-ruined, viable iterations
    viable_healthy = df_healthy[df_healthy['Status'] != 'REJECTED_HARD']
    viable_crash = df_crash[df_crash['Status'] != 'REJECTED_HARD']
    
    # Isolate absolute top configurations (minimum adjusted loss)
    opt_healthy = viable_healthy.loc[viable_healthy['Adjusted_Loss'].idxmin()]
    opt_crash = viable_crash.loc[viable_crash['Adjusted_Loss'].idxmin()]
    
    # Calculate drift metrics
    kelly_drift = opt_crash['Kelly_Fraction'] - opt_healthy['Kelly_Fraction']
    vault_drift = opt_crash['Vault_Threshold'] - opt_healthy['Vault_Threshold']
    
    kelly_pct_change = (kelly_drift / opt_healthy['Kelly_Fraction']) * 100
    vault_pct_change = (vault_drift / opt_healthy['Vault_Threshold']) * 100
    
    print("=======================================================")
    print("📈 PARAMETER DRIFT METRIC COMPARISON ANALYSIS")
    print("=======================================================")
    print(f"Baseline Kelly Allocation Fraction : {opt_healthy['Kelly_Fraction']:.4f}")
    print(f"Defensive Crash Allocation Fraction: {opt_crash['Kelly_Fraction']:.4f}")
    print(f"➡️ Kelly Shift Delta               : {kelly_drift:.4f} ({kelly_pct_change:+.2f}%)")
    print("-------------------------------------------------------")
    print(f"Baseline Vault Sweep Threshold     : ${opt_healthy['Vault_Threshold']:.2f}")
    print(f"Defensive Crash Sweep Threshold    : ${opt_crash['Vault_Threshold']:.2f}")
    print(f"➡️ Vault Shift Delta               : ${vault_drift:.2f} ({vault_pct_change:+.2f}%)")
    print("=======================================================\n")
    
    # --- Visualization Comparison Frame ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Kelly distribution migration plot
    ax1.hist(viable_healthy['Kelly_Fraction'], bins=15, alpha=0.5, color='teal', label='Healthy Regime')
    ax1.hist(viable_crash['Kelly_Fraction'], bins=15, alpha=0.5, color='orangered', label='Crash Regime')
    ax1.axvline(opt_healthy['Kelly_Fraction'], color='teal', linestyle='--', linewidth=2, label='Opt Baseline')
    ax1.axvline(opt_crash['Kelly_Fraction'], color='orangered', linestyle='--', linewidth=2, label='Opt Crash')
    ax1.set_title('Kelly Fraction Grid Vector Migration')
    ax1.set_xlabel('Position Sizing Fraction')
    ax1.set_ylabel('Observation Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Vault threshold distribution migration plot
    ax2.hist(viable_healthy['Vault_Threshold'], bins=15, alpha=0.5, color='teal', label='Healthy Regime')
    ax2.hist(viable_crash['Vault_Threshold'], bins=15, alpha=0.5, color='orangered', label='Crash Regime')
    ax2.axvline(opt_healthy['Vault_Threshold'], color='teal', linestyle='--', linewidth=2, label='Opt Baseline')
    ax2.axvline(opt_crash['Vault_Threshold'], color='orangered', linestyle='--', linewidth=2, label='Opt Crash')
    ax2.set_title('Vault Sweep Threshold Migration')
    ax2.set_xlabel('Sweep Limits ($)')
    ax2.set_ylabel('Observation Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run the drift analytical framework
analyze_parameter_drift(
    healthy_path="generated/hyperopt_backtest_report.csv", 
    crash_path="generated/hyperopt_crash_test_report.csv"
)
