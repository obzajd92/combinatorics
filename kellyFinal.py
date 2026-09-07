import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# 1. System Parameters & Configuration Boundaries
COIN_PROBS = [0.35, 0.55, 0.10]  # P(Heads), P(Tails), P(In-between)
START_BALANCE = 10000
MAX_STEPS = 500
NUM_SIMULATIONS = 150  # Path runs per distribution for empirical calculation

# Adaptive Exit Boundaries
BASE_TAKE_PROFIT = 25000    
MIN_TAKE_PROFIT = 15000        
INITIAL_TRAILING_STOP_PERCENT = 0.80  

# Option Theta / Time-Decay parameters
DECAY_START_STEP = 150         
DECAY_RATE_PER_STEP = 0.001     
MAX_STOP_TIGHTNESS = 0.95       

# Sizing Matrix
USER_INITIAL_SIZING_INPUT = 0.04  
BASE_LOOKBACK = 20                

# --- ASYMMETRIC VARIANCE THRESHOLDS MATRIX ---
VOL_THRESHOLDS = {
    'Normal': 120,        # Sensitive filter
    'Exponential': 480,   # Permissive filter
    'Weibull': 280        # Targeted downside filter
}

# --- REGIME RISK SCALARS ---
LOW_VOL_SCALAR = 1.10     # Aggressive compounding (+10% risk payload)
HIGH_VOL_SCALAR = 0.40    # Defensive risk contraction (-60% risk payload)

# Attrition drag baseline
TAIL_DRAG_PER_STEP = 15    

def get_coin_flip():
    """Functional generator for the 3-faced coin outcome."""
    return np.random.choice(['H', 'T', 'I'], p=COIN_PROBS)

def normal_step(outcome):
    """Asymmetric Normal base multipliers."""
    if outcome == 'H':
        return np.random.normal(loc=1.2, scale=0.3)   
    elif outcome == 'T':
        return np.random.normal(loc=-1.8, scale=0.9)  
    return np.random.normal(loc=0.1, scale=0.1)

def exponential_step(outcome):
    """Asymmetric Exponential base multipliers."""
    if outcome == 'H':
        return np.random.exponential(scale=3.5)     
    elif outcome == 'T':
        return -np.random.exponential(scale=1.1)    
    return np.random.exponential(scale=0.1)

def weibull_step(outcome):
    """Asymmetric Weibull base multipliers."""
    if outcome == 'H':
        return 1.4 * np.random.weibull(a=1.2)       
    elif outcome == 'T':
        return -2.2 * np.random.weibull(a=0.7)      
    return 0.15 * np.random.weibull(a=1.0)

STRATEGIES = {
    'Normal': normal_step,
    'Exponential': exponential_step,
    'Weibull': weibull_step
}

# --- LAYERED FEE ENGINE WITH REBATE INCENTIVES ---
def calculate_fees_with_rebate(current_balance, peak_balance, raw_payoff):
    """Progressive tiered fee structure coupled with a 30% discount at lifetime highs."""
    if current_balance > 8000:
        fixed_fee, slippage_rate = 5.0, 0.005
    elif current_balance > 5000:
        fixed_fee, slippage_rate = 7.5, 0.010
    else:
        fixed_fee, slippage_rate = 10.0, 0.025
        
    # --- FEE REBATE INCENTIVE LAYER ---
    # Unlocks a 30% cost discount if trading at or above previous equity peak milestones
    rebate_scalar = 0.70 if current_balance >= peak_balance and current_balance > 10000 else 1.0
    
    total_fee = (fixed_fee + (slippage_rate * abs(raw_payoff))) * rebate_scalar
    return total_fee

def simulate_walk(start_balance, steps, name, step_func, initial_sizing):
    """Simulates a single path running adaptive scalars, regime lookbacks, and fee rebates."""
    balance = start_balance
    peak_balance = start_balance
    
    balance_history = [balance]
    floor_history = [balance * INITIAL_TRAILING_STOP_PERCENT]
    target_history = [BASE_TAKE_PROFIT]
    recent_returns = []
    
    high_vol_streak = 0
    strategy_threshold = VOL_THRESHOLDS[name]
    
    for step_idx in range(steps):
        # --- VOLATILITY REGIME DETECTION ---
        is_high_vol = False
        if len(recent_returns) >= 10:
            rolling_vol = np.std(recent_returns[-10:])
            if rolling_vol > strategy_threshold:
                active_lookback = 6   
                high_vol_streak += 1
                is_high_vol = True
            else:
                active_lookback = 32  
                high_vol_streak = max(0, high_vol_streak - 1)
        else:
            active_lookback = BASE_LOOKBACK
            
        # --- DYNAMIC TARGET CONSTRAINTS ---
        target_deflation = high_vol_streak * 160
        current_take_profit = max(MIN_TAKE_PROFIT, BASE_TAKE_PROFIT - target_deflation)
        
        # --- OPTION-THETA TIME-DECAY ---
        if step_idx > DECAY_START_STEP:
            elapsed_past_threshold = step_idx - DECAY_START_STEP
            decay_penalty = elapsed_past_threshold * DECAY_RATE_PER_STEP
            current_stop_pct = min(MAX_STOP_TIGHTNESS, INITIAL_TRAILING_STOP_PERCENT + decay_penalty)
        else:
            current_stop_pct = INITIAL_TRAILING_STOP_PERCENT
        
        # --- ROLLING fractional KELLY ENGINE ---
        if len(recent_returns) >= 5:
            window_slice = recent_returns[-active_lookback:]
            win_rate = sum(1 for r in window_slice if r > 0) / len(window_slice)
            kelly_factor = max(0.01, min(initial_sizing * 1.5, win_rate - (1.0 - win_rate)))
        else:
            kelly_factor = initial_sizing
            
        # --- ADAPTIVE KELLY SCALAR MULTIPLIER ---
        # Scale active risk exposure parameter down during high-vol regimes, expand when quiet
        regime_scalar = HIGH_VOL_SCALAR if is_high_vol else LOW_VOL_SCALAR
        current_stake = balance * kelly_factor * regime_scalar
        
        base_multiplier = step_func(get_coin_flip())
        raw_payoff = base_multiplier * current_stake
        
        # Apply fees via Rebate Engine
        fees = calculate_fees_with_rebate(balance, peak_balance, raw_payoff)
        net_payoff = raw_payoff - TAIL_DRAG_PER_STEP - fees
        
        balance += net_payoff
        recent_returns.append(net_payoff)
        
        # Update Equity Peak and Decaying Trailing Stop Line
        if balance > peak_balance:
            peak_balance = balance
        current_floor = peak_balance * current_stop_pct
        
        # Boundary conditional exits
        if balance <= current_floor:
            balance_history.append(current_floor)
            floor_history.append(current_floor)
            target_history.append(current_take_profit)
            break
        if balance >= current_take_profit:
            balance_history.append(current_take_profit)
            floor_history.append(current_floor)
            target_history.append(current_take_profit)
            break
            
        balance_history.append(balance)
        floor_history.append(current_floor)
        target_history.append(current_take_profit)
        
    return np.array(balance_history), np.array(floor_history), np.array(target_history)

def calculate_ruin_probability(initial_balance, steps, name, step_func, initial_sizing, num_sims=NUM_SIMULATIONS):
    """Calculates empirical risk of trailing floor breach over a given capital curve."""
    ruined_count = 0
    for _ in range(num_sims):
        walk, _, targets = simulate_walk(initial_balance, steps, name, step_func, initial_sizing)
        if walk[-1] < targets[-1] and len(walk) <= steps:
            ruined_count += 1
    return ruined_count / num_sims

def run_game_iteration(retry_num):
    """Executes a full tourney iteration across all three strategies with visual tracking maps."""
    print(f"\n--- Running Scalar & Rebate Adaptive Tournament (Run #{retry_num}) ---")
    
    # Generate visualization data frames
    walks, floors, targets = {}, {}, {}
    for name, func in STRATEGIES.items():
        walks[name], floors[name], targets[name] = simulate_walk(START_BALANCE, MAX_STEPS, name, func, USER_INITIAL_SIZING_INPUT)
        
    # Analyze Risk Area Curve (AUC) over entry boundaries
    floor_base = START_BALANCE * INITIAL_TRAILING_STOP_PERCENT
    balances_to_test = np.linspace(floor_base, START_BALANCE, 15)
    ruin_curves = {name: [] for name in STRATEGIES}
    
    for b in balances_to_test:
        for name, func in STRATEGIES.items():
            prob = calculate_ruin_probability(b, MAX_STEPS, name, func, USER_INITIAL_SIZING_INPUT)
            ruin_curves[name].append(prob)
            
    # Compute integral surface area footprint (Lower Area = Winner)
    aucs = {}
    domain_range = START_BALANCE - floor_base
    for name, curves in ruin_curves.items():
        auc_val = simpson(curves, x=balances_to_test) / domain_range
        aucs[name] = auc_val
        print(f"[{name}] Scalar-Adaptive AUC: {auc_val:.4f} | End Walk Balance: ${walks[name][-1]:,.2f}")
        
    winner = min(aucs, key=aucs.get)
    print(f"-> WINNER FOR RUN #{retry_num}: {winner} Strategy Model")
    
    # Render Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'Normal': 'blue', 'Exponential': 'orange', 'Weibull': 'green'}
    
    for name, path in walks.items():
        ax1.plot(path, label=f"{name} (Final: ${path[-1]:,.0f})", color=colors[name])
        ax1.plot(floors[name], color=colors[name], linestyle=':', alpha=0.4)
        ax1.plot(targets[name], color=colors[name], linestyle='--', alpha=0.5, label=f'{name} Dynamic TP')
        
    ax1.set_title(f"Random Walks: Adaptive Scalars + Fee Rebates (Run #{retry_num})")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Balance ($)")
    ax1.legend(loc='upper left', bbox_to_anchor=(1,1))
    ax1.grid(True, alpha=0.3)
    
    for name, curves in ruin_curves.items():
        ax2.plot(balances_to_test, curves, label=f"{name} (AUC: {aucs[name]:.2f})", color=colors[name])
        ax2.fill_between(balances_to_test, curves, color=colors[name], alpha=0.1)
    ax2.set_title("Risk Surface Footprint (AUC Comparison)")
    ax2.set_xlabel("Starting Balance Floor Variant ($)")
    ax2.set_ylabel("Probability of Floor Breach")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
results_summary.append((i, round_winner))print("\n==========================================")print("FINAL TOURNAMENT SCOREBOARD")print("==========================================")for num_retries, winner in results_summary:print(f"Retry Run #{num_retries} -> Tournament Winner: {winner} Distribution")
    
    plt.tight_layout()
    plt.show()
    
    return winner

# 2. Main Tournament Automation Interface (3 Automated Retries)
if __name__ == "__main__":
    total_retries = 3
    results_summary = []
    
    for i in range(1, total_retries + 1):
        round_winner = run_game_iteration(retry_num=i)
