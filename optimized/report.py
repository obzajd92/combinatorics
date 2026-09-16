import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# --- Global Parameter & Frictional Settings ---
P_NORMAL, P_JACKPOT, P_LOSS = 0.49, 0.01, 0.50
PAYOUT_NORMAL, PAYOUT_JACKPOT = 1000.0, 10000.0
TICKET_PRICE, FLAT_FEE = 520.0, 10.0
BASE_COST = TICKET_PRICE + FLAT_FEE
TAX_NORMAL, TAX_JACKPOT = 0.15, 0.35
DEFAULT_RISK = 0.02
INITIAL_BANKROLL = 2000.0
NUM_GAMES = 150
NUM_SIMS = 200  

SLIDING_LOCK_PCT = 0.75
MAX_ALLOWED_DRAWDOWN = 0.50  

# Frictional Calculations
EFFECTIVE_JACKPOT = PAYOUT_JACKPOT * (1.0 - DEFAULT_RISK)
MULT_LOSS = -1.0
MULT_NORMAL = ((PAYOUT_NORMAL - BASE_COST) * (1.0 - TAX_NORMAL)) / BASE_COST
MULT_JACKPOT = ((EFFECTIVE_JACKPOT - BASE_COST) * (1.0 - TAX_JACKPOT)) / BASE_COST

# Fixed Seed for Path Reproducibility Across Space Grid
np.random.seed(42)
ROLLS_MATRIX = np.random.rand(NUM_SIMS, NUM_GAMES)

# Global tracker for hyperopt run telemetry reporting
iteration_records = []

def objective(params):
    """
    Bivariate Objective Function with Asymmetric Exponential Drawdown Penalties.
    """
    f = params['kelly_fraction']
    vault_threshold = params['vault_threshold']
    
    mult_matrix = np.where(ROLLS_MATRIX < P_JACKPOT, MULT_JACKPOT, 
                           np.where(ROLLS_MATRIX < (P_JACKPOT + P_NORMAL), MULT_NORMAL, MULT_LOSS))
    
    jackpot_matrix = ROLLS_MATRIX < P_JACKPOT
    terminal_balances = np.zeros(NUM_SIMS)
    max_drawdown_observed = 0.0
    
    for sim in range(NUM_SIMS):
        active_balance = INITIAL_BANKROLL
        vault_balance = 0.0
        max_path_equity = INITIAL_BANKROLL  
        jackpot_hit = False
        max_active_balance = INITIAL_BANKROLL 
        
        for game in range(NUM_GAMES):
            if jackpot_hit:
                trailing_floor = INITIAL_BANKROLL + (SLIDING_LOCK_PCT * (max_active_balance - INITIAL_BANKROLL))
                if active_balance <= trailing_floor:
                    active_balance = trailing_floor
                    break
            
            wager = BASE_COST * f
            if active_balance < wager:
                active_balance = 0.0  
                break
                
            pnl = wager * mult_matrix[sim, game]
            active_balance += pnl
            
            if jackpot_matrix[sim, game]:
                jackpot_hit = True
                
            if active_balance > max_active_balance:
                max_active_balance = active_balance
                
            if active_balance > vault_threshold:
                surplus = active_balance - vault_threshold
                vault_balance += surplus
                active_balance = vault_threshold
            
            current_combined_equity = active_balance + vault_balance
            if current_combined_equity > max_path_equity:
                max_path_equity = current_combined_equity
                
            current_drawdown = (max_path_equity - current_combined_equity) / max_path_equity
            if current_drawdown > max_drawdown_observed:
                max_drawdown_observed = current_drawdown
                
        terminal_balances[sim] = active_balance + vault_balance

    # Downside Risk Hurdles
    path_returns = (terminal_balances - INITIAL_BANKROLL) / INITIAL_BANKROLL
    mean_return = np.mean(path_returns)
    target_return = 0.02 
    
    downside_diffs = np.where(path_returns < target_return, path_returns - target_return, 0.0)
    downside_deviation = np.sqrt(np.mean(downside_diffs ** 2))
    
    if downside_deviation > 1e-6:
        sortino_ratio = (mean_return - target_return) / downside_deviation
    else:
        sortino_ratio = -10.0 if mean_return <= target_return else (mean_return - target_return) / 1e-6
        
    # --- Asymmetric Risk Vectorization ---
    if max_drawdown_observed >= MAX_ALLOWED_DRAWDOWN:
        penalty = 1000.0 * (1.0 + (max_drawdown_observed - MAX_ALLOWED_DRAWDOWN) * 50.0)
        status_flag = "REJECTED_HARD"
    elif max_drawdown_observed > 0.35:
        penalty = 5.0 * np.exp((max_drawdown_observed - 0.35) / (MAX_ALLOWED_DRAWDOWN - 0.35) * 3.0)
        status_flag = "PENALIZED_WARN"
    else:
        penalty = 0.0
        status_flag = "OPTIMAL"
        
    adjusted_loss = -sortino_ratio + penalty
    
    iteration_records.append({
        'Iteration': len(iteration_records) + 1,
        'Kelly_Fraction': round(f, 4),
        'Vault_Threshold': round(vault_threshold, 2),
        'Max_Observed_Drawdown': round(max_drawdown_observed, 4),
        'Raw_Sortino': round(sortino_ratio, 4),
        'Applied_Penalty': round(penalty, 4),
        'Adjusted_Loss': round(adjusted_loss, 4),
        'Status': status_flag
    })
    
    return {'loss': adjusted_loss, 'status': STATUS_OK}

def isolate_highest_performing_warned_profile(report_path):
    """
    Custom parsing function to extract the peak executing configuration 
    that operated inside the high-drawdown PENALIZED_WARN danger zone.
    """
    if not os.path.exists(report_path):
        return "Target log file not found."
        
    df = pd.read_csv(report_path)
    warned_sector = df[df['Status'] == 'PENALIZED_WARN']
    
    if warned_sector.empty:
        return "\n[!] System Alert: No parameter sweeps landed inside the PENALIZED_WARN risk band during this run."
        
    # Highest performance equals the lowest objective loss profile
    best_profile = warned_sector.loc[warned_sector['Adjusted_Loss'].idxmin()]
    
    summary = (
        f"\n=======================================================\n"
        f"🏆 OPTIMIZED SECTOR PROFILE IDENTIFIED: PENALIZED_WARN\n"
        f"=======================================================\n"
        f"• Iteration Row:        #{int(best_profile['Iteration'])}\n"
        f"• Kelly Allocation:    {best_profile['Kelly_Fraction']:.4f}\n"
        f"• Vault Sweep Level:   ${best_profile['Vault_Threshold']:.2f}\n"
        f"• Max Risk Drawdown:   {(best_profile['Max_Observed_Drawdown']*100):.2f}%\n"
        f"• Unadjusted Sortino:  {best_profile['Raw_Sortino']:.4f}\n"
        f"• Exponential Penalty: {best_profile['Applied_Penalty']:.4f}\n"
        f"• Adjusted Asset Loss: {best_profile['Adjusted_Loss']:.4f}\n"
        f"======================================================="
    )
    return summary

# --- Run Hyperopt ---
space = {
    'kelly_fraction': hp.uniform('kelly_fraction', 0.01, 0.40),
    'vault_threshold': hp.uniform('vault_threshold', 2200.0, 6000.0)
}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=250, trials=trials)

# Export Telemetry CSV
os.makedirs("generated", exist_ok=True)
report_csv = "generated/hyperopt_backtest_report.csv"
df_report = pd.DataFrame(iteration_records)
df_report.to_csv(report_csv, index=False)

# Execute the custom parser function
print(isolate_highest_performing_warned_profile(report_csv))

# --- Loss Distribution Visualization Frame ---
optimal_data = df_report[df_report['Status'] != 'REJECTED_HARD']

plt.figure(figsize=(11, 5))
plt.hist(optimal_data['Raw_Sortino'], bins=25, alpha=0.6, color='blue', label='Raw Sortino Performance (Inverse)')
plt.hist(optimal_data['Adjusted_Loss'], bins=25, alpha=0.6, color='crimson', label='Adjusted Optimization Loss (With Penalties)')
plt.title('Performance Telemetry Distribution Analysis', fontsize=12, fontweight='bold')
plt.xlabel('Metric Space Value', fontsize=10)
plt.ylabel('Observation Frequency Count', fontsize=10)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
