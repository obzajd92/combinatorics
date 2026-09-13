import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# --- Global Parameter & Friction Matrix Configuration ---
P_NORMAL, P_JACKPOT, P_LOSS = 0.49, 0.01, 0.50
PAYOUT_NORMAL, PAYOUT_JACKPOT = 1000.0, 10000.0
TICKET_PRICE, FLAT_FEE = 520.0, 10.0
BASE_COST = TICKET_PRICE + FLAT_FEE
TAX_NORMAL, TAX_JACKPOT = 0.15, 0.35
DEFAULT_RISK = 0.02
INITIAL_BANKROLL = 2000.0
NUM_GAMES = 150
NUM_SIMS = 200  # Number of simulation paths per trial evaluation

SLIDING_LOCK_PCT = 0.75
VAULT_THRESHOLD = 3000.0

# Pre-adjust game multipliers for variable frictions
EFFECTIVE_JACKPOT = PAYOUT_JACKPOT * (1.0 - DEFAULT_RISK)
MULT_LOSS = -1.0
MULT_NORMAL = ((PAYOUT_NORMAL - BASE_COST) * (1.0 - TAX_NORMAL)) / BASE_COST
MULT_JACKPOT = ((EFFECTIVE_JACKPOT - BASE_COST) * (1.0 - TAX_JACKPOT)) / BASE_COST

# Set random seed for path alignment consistency across trials
np.random.seed(42)
ROLLS_MATRIX = np.random.rand(NUM_SIMS, NUM_GAMES)
DEFAULT_MATRIX = np.random.rand(NUM_SIMS, NUM_GAMES)

def objective(params):
    """
    Objective function for Hyperopt.
    Calculates: probability(max_win) * max_win - probability(max_loss) * |max_loss|
    Since Hyperopt minimizes, we return the negative of this score as the loss.
    """
    f = params['kelly_fraction']
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
            
            # Capital Ruin Safeguard Cap
            if active_balance < BASE_COST * f:
                break
                
            # Leverage Modifier Circuit Breakers
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
                
            # Rebalancing Vault Sweep
            if active_balance > VAULT_THRESHOLD:
                excess = active_balance - VAULT_THRESHOLD
                vault_balance += excess
                active_balance = VAULT_THRESHOLD
                
            total_w = active_balance + vault_balance
            if total_w > max_balance:
                max_balance = total_w
                
        final_return = (active_balance + vault_balance) - INITIAL_BANKROLL
        if final_return > 0:
            max_wins.append(final_return)
        else:
            max_losses.append(final_return)
            
    # Compute statistical probabilities
    p_win = len(max_wins) / NUM_SIMS
    p_loss_state = len(max_losses) / NUM_SIMS
    
    avg_max_win = np.mean(max_wins) if max_wins else 0.0
    avg_max_loss = np.mean(max_losses) if max_losses else 0.0
    
    # Target Equation: P(Win)*Max_Win - P(Loss)*|Max_Loss|
    obj_score = (p_win * avg_max_win) - (p_loss_state * abs(avg_max_loss))
    
    # Store dynamic parameters alongside execution outputs
    return {
        'loss': -obj_score,  # Negated because Hyperopt minimizes
        'status': STATUS_OK,
        'kelly_fraction_pct': f * 100,
        'max_win_achieved': avg_max_win,
        'max_loss_suffered': avg_max_loss,
        'objective_score': obj_score
    }

# =========================================================================
# RUN HYPEROPT OPTIMIZATION SWEEP
# =========================================================================
# 1. Define continuous search space bounding parameter domain from 0.01% to 99.9%
space = {
    'kelly_fraction': hp.uniform('kelly_fraction', 0.0001, 0.999)
}

# 2. Instantiate optimization logging trial vector
trials = Trials()

print("Launching Hyperopt TPE Optimization Search (1,000 Evaluations)...")
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=1000,
    trials=trials
)

# =========================================================================
# PROCESS & EXPORT STRUCTURAL RESULTS
# =========================================================================
sweep_ledger = []
for trial in trials.results:
    if trial['status'] == STATUS_OK:
        sweep_ledger.append({
            "Kelly Fraction (%)": round(trial['kelly_fraction_pct'], 4),
            "Max Win Achieved": round(trial['max_win_achieved'], 2),
            "Max Loss Suffered": round(trial['max_loss_suffered'], 2),
            "Objective Score": round(trial['objective_score'], 2)
        })

df_results = pd.DataFrame(sweep_ledger)

# Order values by parameter size descending (Price/Fraction Sizing Priority)
df_results = df_results.sort_values(by="Kelly Fraction (%)", ascending=False).reset_index(drop=True)

# Limit to top 1,000 unique records and write to file
df_results.head(1000).to_csv("hyperopt_kelly_results.csv", index=False)
print(f"Optimization Sweep Complete. Top 1,000 rows exported to 'hyperopt_kelly_results.csv'.")
print(f"Mathematically Optimal Sizing Found: {best['kelly_fraction'] * 100:.4f}%")
