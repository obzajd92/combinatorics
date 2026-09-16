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
NUM_SIMS = 200  # Upgraded simulation iterations for dense data processing

SLIDING_LOCK_PCT = 0.75

# Frictional Calculations
EFFECTIVE_JACKPOT = PAYOUT_JACKPOT * (1.0 - DEFAULT_RISK)
MULT_LOSS = -1.0
MULT_NORMAL = ((PAYOUT_NORMAL - BASE_COST) * (1.0 - TAX_NORMAL)) / BASE_COST
MULT_JACKPOT = ((EFFECTIVE_JACKPOT - BASE_COST) * (1.0 - TAX_JACKPOT)) / BASE_COST

# Fixed Seed for Path Reproducibility Across Space Grid
np.random.seed(42)
ROLLS_MATRIX = np.random.rand(NUM_SIMS, NUM_GAMES)

def objective(params):
    """
    Optimized Vectorized Bivariate Objective Function for Hyperopt.
    Simultaneously benchmarks Kelly Fraction and Vault Sweep Thresholds.
    """
    f = params['kelly_fraction']
    vault_threshold = params['vault_threshold']
    
    # Pre-map game multipliers based on probability matrices
    # Vectorized condition assigning payout multipliers to random states
    mult_matrix = np.where(ROLLS_MATRIX < P_JACKPOT, MULT_JACKPOT, 
                           np.where(ROLLS_MATRIX < (P_JACKPOT + P_NORMAL), MULT_NORMAL, MULT_LOSS))
    
    jackpot_matrix = ROLLS_MATRIX < P_JACKPOT
    
    terminal_balances = np.zeros(NUM_SIMS)
    
    for sim in range(NUM_SIMS):
        active_balance = INITIAL_BANKROLL
        vault_balance = 0.0
        max_balance = INITIAL_BANKROLL
        jackpot_hit = False
        
        for game in range(NUM_GAMES):
            # Dynamic Sliding Profit Trailing Floor Check
            if jackpot_hit:
                trailing_floor = INITIAL_BANKROLL + (SLIDING_LOCK_PCT * (max_balance - INITIAL_BANKROLL))
                if active_balance <= trailing_floor:
                    active_balance = trailing_floor
                    break
            
            # Capital Ruin Cap Check
            wager = BASE_COST * f
            if active_balance < wager:
                break
                
            # Process Game Outcome
            outcome_mult = mult_matrix[sim, game]
            is_jackpot = jackpot_matrix[sim, game]
            
            # Update Balances
            pnl = wager * outcome_mult
            active_balance += pnl
            
            if is_jackpot:
                jackpot_hit = True
                
            # Track peak maximum balance for the trailing floor calculation
            if active_balance > max_balance:
                max_balance = active_balance
                
            # Vault Sweep Mechanic
            if active_balance > vault_threshold:
                surplus = active_balance - vault_threshold
                vault_balance += surplus
                active_balance = vault_threshold
                
        terminal_balances[sim] = active_balance + vault_balance

    # Objective: Minimize negative expected utility (Maximize expected ending bankroll)
    mean_outcome = np.mean(terminal_balances)
    
    return {'loss': -mean_outcome, 'status': STATUS_OK, 'mean_balance': mean_outcome}

# --- Hyperopt Space Setup & Run ---
space = {
    'kelly_fraction': hp.uniform('kelly_fraction', 0.05, 0.50),
    'vault_threshold': hp.uniform('vault_threshold', 2500.0, 7500.0)
}

trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=300,
    trials=trials
)

print(f"\n--- Optimization Complete ---")
print(f"Best Kelly Fraction: {best['kelly_fraction']:.4f}")
print(f"Best Vault Threshold: ${best['vault_threshold']:.2f}")

# --- Data Structuring & 3D Visualization Surface ---
results = [{
    'x': t['result']['status'] == 'ok' and t['misc']['vals']['kelly_fraction'][0],
    'y': t['misc']['vals']['vault_threshold'][0],
    'z': -t['result']['loss']
} for t in trials.trials]

df = pd.DataFrame(results)

xi = np.linspace(df['x'].min(), df['x'].max(), 100)
yi = np.linspace(df['y'].min(), df['y'].max(), 100)
X, Y = np.meshgrid(xi, yi)
Z = griddata((df['x'], df['y']), df['z'], (X, Y), method='cubic')

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

ax.set_title('Risk Topology Surface Optimization', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Kelly Fraction Allocation', fontsize=11, labelpad=10)
ax.set_ylabel('Vault Sweep Threshold ($)', fontsize=11, labelpad=10)
ax.set_zlabel('Expected Terminal Balance ($)', fontsize=11, labelpad=10)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Expected Return Profile')

plt.show()
