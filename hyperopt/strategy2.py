import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
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
NUM_SIMS = 150  # Path iterations per individual trial

SLIDING_LOCK_PCT = 0.75

# Frictional Calculations
EFFECTIVE_JACKPOT = PAYOUT_JACKPOT * (1.0 - DEFAULT_RISK)
MULT_LOSS = -1.0
MULT_NORMAL = ((PAYOUT_NORMAL - BASE_COST) * (1.0 - TAX_NORMAL)) / BASE_COST
MULT_JACKPOT = ((EFFECTIVE_JACKPOT - BASE_COST) * (1.0 - TAX_JACKPOT)) / BASE_COST

# Fixed Seed for Path Reproducibility Across Space Grid
np.random.seed(42)
ROLLS_MATRIX = np.random.rand(NUM_SIMS, NUM_GAMES)
DEFAULT_MATRIX = np.random.rand(NUM_SIMS, NUM_GAMES)

def objective(params):
    """
    Bivariate Objective Function for Hyperopt.
    Simultaneously benchmarks Kelly Fraction and Vault Sweep Thresholds.
    """
    f = params['kelly_fraction']
    vault_threshold = params['vault_threshold']
    
    max_wins = []
    max_losses = []
    
    for sim in range(NUM_SIMS):
        active_balance = INITIAL_BANKROLL
        vault_balance = 0.0
        max_balance = INITIAL_BANKROLL
        jackpot_hit = False
        consecutive_losses = 0
        
        for game in range(NUM_GAMES):
            # 75% Sliding Profit Trailing Floor Check
            if jackpot_hit:
                trailing_floor = INITIAL_BANKROLL + (SLIDING_LOCK_PCT * (max_balance - INITIAL_BANKROLL))
                if active_balance <= trailing_floor:
                    active_balance = trailing_floor
                    break
            
            # Capital Ruin Cap Check
            if active_balance < BASE_COST * f:
                break
                
            # Circuit Breaker Sizing Adjustments
            if consecutive_losses >= 5: leverage = 0.25
            elif consecutive_losses >= 3: leverage = 0.50
            else: leverage = 1.0
            
            wager = active_balance * f * leverage
            
            roll = ROLLS_MATRIX[sim, game]
            if roll < 0.01:
                if DEFAULT_MATRIX[sim, game] < DEFAULT_RISK:
                    active_balance += wager * MULT_LOSS
                    consecutive_losses += 1
                else:
                    active_balance += wager * MULT_JACKPOT
                    jackpot_hit = True
                    consecutive_losses = 0
            elif roll < (0.01 + P_NORMAL):
                active_balance += wager * MULT_NORMAL
                consecutive_losses = 0
            else:
                active_balance += wager * MULT_LOSS
                consecutive_losses += 1
                
            # Dynamic Sweep Execution
            if active_balance > vault_threshold:
                excess = active_balance - vault_threshold
                vault_balance += excess
                active_balance = vault_threshold
                
            total_w = active_balance + vault_balance
            if total_w > max_balance:
                max_balance = total_w
                
        final_return = (active_balance + vault_balance) - INITIAL_BANKROLL
        if final_return > 0:
            max_wins.append(final_return)
        else:
            max_losses.append(final_return)
            
    p_win = len(max_wins) / NUM_SIMS
    p_loss_state = len(max_losses) / NUM_SIMS
    
    avg_max_win = np.mean(max_wins) if max_wins else 0.0
    avg_max_loss = np.mean(max_losses) if max_losses else 0.0
    
    # Calculate Objective Function
    obj_score = (p_win * avg_max_win) - (p_loss_state * abs(avg_max_loss))
    
    return {
        'loss': -obj_score,  # Hyperopt minimizes the negative loss
        'status': STATUS_OK,
        'kelly_fraction_pct': f * 100,
        'vault_threshold_val': vault_threshold,
        'objective_score': obj_score
    }

# =========================================================================
# EXECUTING BIVARIATE SEARCH
# =========================================================================
# Search Space: Kelly Fraction (0.01% - 99.9%) & Sweep Target ($2,100 - $5,000)
bivariate_space = {
    'kelly_fraction': hp.uniform('kelly_fraction', 0.0001, 0.999),
    'vault_threshold': hp.uniform('vault_threshold', 2100.0, 5000.0)
}

trials = Trials()
print("Launching Bivariate Hyperopt Optimization (1,000 Global Runs)...")
best = fmin(
    fn=objective,
    space=bivariate_space,
    algo=tpe.suggest,
    max_evals=1000,
    trials=trials
)

# =========================================================================
# 3D MATPLOTLIB VISUALIZATION WRAPPER
# =========================================================================
x_vals = [t['kelly_fraction_pct'] for t in trials.results if t['status'] == STATUS_OK]
y_vals = [t['vault_threshold_val'] for t in trials.results if t['status'] == STATUS_OK]
z_vals = [t['objective_score'] for t in trials.results if t['status'] == STATUS_OK]

# Convert array structures to numpy formats
x = np.array(x_vals)
y = np.array(y_vals)
z = np.array(z_vals)

# Create structured mesh grid for surface interpolation
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate irregularly spaced hyperopt trials onto a smooth grid
zi = griddata((x, y), z, (xi, yi), method='linear')

# Plotting the 3D Optimization Surface
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

# Render color-mapped surface topography
surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', alpha=0.8)

# Customize Axis Labels
ax.set_title("Bivariate Risk Topology: Sizing Fraction vs. Vault Threshold Optimization", fontsize=12, fontweight='bold')
ax.set_xlabel("Dynamic Kelly Fraction (%)", fontsize=10)
ax.set_ylabel("Vault Sweep Threshold ($)", fontsize=10)
ax.set_zlabel("Objective Performance Score", fontsize=10)

# Add color bar legend indicator
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Strategy Value Profile")

# Identify and isolate the absolute peak coordinate
best_idx = np.argmax(z)
ax.scatter(x[best_idx], y[best_idx], z[best_idx], color='red', s=100, label='Global Optimum Peak', depthshade=False)
ax.legend()

plt.tight_layout()
plt.show()

print("\n" + "="*55)
print("     BIVARIATE GLOBAL OPTIMUM COORDINATES FOUND      ")
print("="*55)
print(f"Optimal Sizing (Kelly Fraction) : {best['kelly_fraction']*100:.4f}%")
print(f"Optimal Protective Sweep Level : ${best['vault_threshold']:.2f}")
print(f"Peak Bivariate Objective Score : {z[best_idx]:.2f}")
print("="*55 + "\n")
