import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# 1. Base Configuration & Boundaries
COIN_PROBS = [0.35, 0.55, 0.10]  # P(Heads), P(Tails), P(In-between)
START_BALANCE = 10000
MAX_STEPS = 500
NUM_SIMULATIONS = 200  

# Bounded Conditions
TAKE_PROFIT = 25000    
TRAILING_STOP_PERCENT = 0.80  # Relative Stop: 80% of achieved peak capital

# Friction parameters
TAIL_DRAG_PER_STEP = 15    
FIXED_FEE_PER_STEP = 5.0   
SLIPPAGE_RATE = 0.005      

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

def calculate_fees(raw_payoff):
    """Functional calculation of total transaction friction."""
    return FIXED_FEE_PER_STEP + (SLIPPAGE_RATE * abs(raw_payoff))

def simulate_walk(start_balance, steps, step_func, initial_sizing):
    """Simulates a path with dynamic adaptive Kelly sizing tracking a trailing stop floor."""
    balance = start_balance
    peak_balance = start_balance
    
    balance_history = [balance]
    floor_history = [balance * TRAILING_STOP_PERCENT]
    
    # Track trailing performance variables for adaptive sizing adjustments
    recent_returns = []
    
    for _ in range(steps):
        # --- ADAPTIVE KELLY ENGINE ---
        # Starts at user input; adapts dynamically based on running asset-base performance
        if len(recent_returns) > 5:
            win_rate = sum(1 for r in recent_returns[-15:] if r > 0) / len(recent_returns[-15:])
            # Classic simplified Kelly adjustment fraction: f = p - q
            kelly_factor = max(0.01, min(initial_sizing * 1.5, win_rate - (1.0 - win_rate)))
        else:
            kelly_factor = initial_sizing
            
        current_stake = balance * kelly_factor
        base_multiplier = step_func(get_coin_flip())
        raw_payoff = base_multiplier * current_stake
        
        # Deduct transaction fees and core drag
        fees = calculate_fees(raw_payoff)
        net_payoff = raw_payoff - TAIL_DRAG_PER_STEP - fees
        
        balance += net_payoff
        recent_returns.append(net_payoff)
        
        # Track Peak and Trailing Floor
        if balance > peak_balance:
            peak_balance = balance
        current_floor = peak_balance * TRAILING_STOP_PERCENT
        
        if balance <= current_floor:
            balance_history.append(current_floor)
            floor_history.append(current_floor)
            break
        if balance >= TAKE_PROFIT:
            balance_history.append(TAKE_PROFIT)
            floor_history.append(current_floor)
            break
            
        balance_history.append(balance)
        floor_history.append(current_floor)
        
    return np.array(balance_history), np.array(floor_history)

def calculate_ruin_probability(initial_balance, steps, step_func, initial_sizing, num_sims=NUM_SIMULATIONS):
    """Calculates empirical probability of triggering the dynamic floor under specific sizing constraints."""
    ruined_count = 0
    for _ in range(num_sims):
        walk, _ = simulate_walk(initial_balance, steps, step_func, initial_sizing)
        if walk[-1] < TAKE_PROFIT and len(walk) <= steps:
            ruined_count += 1
    return ruined_count / num_sims

def run_game_iteration(retry_num, initial_sizing):
    """Executes a full simulation evaluation cycle with adaptive fractional input constraints."""
    print(f"\n--- Running Adaptive Kelly Simulation (Run: #{retry_num} | Initial Sizing Input: {initial_sizing:.2%}) ---")
    
    # Generate sample visualization steps
    walks = {}
    floors = {}
    for name, func in STRATEGIES.items():
        walks[name], floors[name] = simulate_walk(START_BALANCE, MAX_STEPS, func, initial_sizing)
        
    # Track Area metrics over a range of test initial boundaries
    floor_base = START_BALANCE * TRAILING_STOP_PERCENT
    balances_to_test = np.linspace(floor_base, START_BALANCE, 15)
    ruin_curves = {name: [] for name in STRATEGIES}
    
    for b in balances_to_test:
        for name, func in STRATEGIES.items():
            prob = calculate_ruin_probability(b, MAX_STEPS, func, initial_sizing)
            ruin_curves[name].append(prob)
            
    # Calculate integration curves
    aucs = {}
    domain_range = START_BALANCE - floor_base
    for name, curves in ruin_curves.items():
        auc_val = simpson(curves, x=balances_to_test) / domain_range
        aucs[name] = auc_val
        print(f"[{name}] Bounded Ruin AUC: {auc_val:.4f} | End Sample Balance: ${walks[name][-1]:,.2f}")
        
    winner = min(aucs, key=aucs.get)
    print(f"-> WINNER FOR RUN #{retry_num}: {winner} Distribution")
    
    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'Normal': 'blue', 'Exponential': 'orange', 'Weibull': 'green'}
    
    for name, path in walks.items():
        ax1.plot(path, label=f"{name} Path (Final: ${path[-1]:,.0f})", color=colors[name])
        ax1.plot(floors[name], color=colors[name], linestyle=':', alpha=0.4)
    ax1.axhline(TAKE_PROFIT, color='purple', linestyle='--', alpha=0.8, label=f'Take-Profit (${TAKE_PROFIT:,})')
    ax1.set_title(f"Adaptive Kelly Random Walks (Run #{retry_num})")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Balance ($)")
    ax1.legend(loc='upper left', bbox_to_anchor=(1,1))
    ax1.grid(True, alpha=0.3)
    
    for name, curves in ruin_curves.items():
        ax2.plot(balances_to_test, curves, label=f"{name} (AUC: {aucs[name]:.2f})", color=colors[name])
        ax2.fill_between(balances_to_test, curves, color=colors[name], alpha=0.1)
    ax2.set_title(f"Risk Surface Area Profile (Run #{retry_num})")
    ax2.set_xlabel("Starting Balance Floor Variant ($)")
    ax2.set_ylabel("Probability of Hitting Floor")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return winner

# 2. Main Tournament Control Interface
if __name__ == "__main__":
    total_retries = 3
    results_summary = []
    
    # --- USER SIZING INPUT ---
    # Change this fractional value to alter initial allocation size per step.
    USER_INITIAL_SIZING_INPUT = 0.04  # 4% initial bankroll allocation fraction
    
    for i in range(1, total_retries + 1):
        round_winner = run_game_iteration(retry_num=i, initial_sizing=USER_INITIAL_SIZING_INPUT)
        results_summary.append((i, round_winner))
        
    print("\n==========================================")
    print("FINAL TOURNAMENT RESULTS SUMMARY")
    print("==========================================")
    for num_retries, winner in results_summary:
        print(f"Retry Run #{num_retries} -> Winner: {winner} Distribution")
