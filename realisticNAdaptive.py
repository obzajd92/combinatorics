import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# 1. Configuration & Bounded Thresholds
COIN_PROBS = [0.35, 0.55, 0.10]  # P(Heads), P(Tails), P(In-between)
START_BALANCE = 10000
MAX_STEPS = 500
NUM_SIMULATIONS = 200  

# Boundaries
TAKE_PROFIT = 25000    
TRAILING_STOP_PERCENT = 0.80  # Relative Floor: Stop-loss trails at 80% of peak balance

# Adaptive position sizing fraction
KELLY_FRACTION = 0.05  

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

def simulate_walk(start_balance, steps, step_func):
    """Simulates a path bounded by Take-Profit and a Dynamic Trailing Stop-Loss Floor."""
    balance = start_balance
    peak_balance = start_balance
    
    # Track paths for balance and the trailing floor itself
    balance_history = [balance]
    floor_history = [balance * TRAILING_STOP_PERCENT]
    
    for _ in range(steps):
        outcome = get_coin_flip()
        
        # Adaptive stake calculation based on current asset base
        current_stake = balance * KELLY_FRACTION
        raw_payoff = step_func(outcome) * current_stake
        fees = calculate_fees(raw_payoff)
        
        # Apply payoff adjusted for drag metrics
        balance += (raw_payoff - TAIL_DRAG_PER_STEP - fees)
        
        # Update peak and establish relative stop floor
        if balance > peak_balance:
            peak_balance = balance
        current_floor = peak_balance * TRAILING_STOP_PERCENT
        
        # Boundary conditional logic breaks out early
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

def calculate_ruin_probability(initial_balance, steps, step_func, num_sims=NUM_SIMULATIONS):
    """Calculates empirical probability of hitting the relative trailing floor."""
    ruined_count = 0
    for _ in range(num_sims):
        walk, _ = simulate_walk(initial_balance, steps, step_func)
        # Check if early termination was caused by hitting the trailing floor
        if walk[-1] < TAKE_PROFIT and len(walk) <= steps:
            ruined_count += 1
    return ruined_count / num_sims

def run_game_iteration(retry_num):
    """Executes full iteration cycle evaluating survival AUC over trailing floors."""
    print(f"\n--- Running Trailing Floor Simulation (Run: #{retry_num}) ---")
    
    # Generate visual test example paths
    walks = {}
    floors = {}
    for name, func in STRATEGIES.items():
        walks[name], floors[name] = simulate_walk(START_BALANCE, MAX_STEPS, func)
        
    # Calculate Risk Surface Area Curves using various entry benchmarks
    floor_base = START_BALANCE * TRAILING_STOP_PERCENT
    balances_to_test = np.linspace(floor_base, START_BALANCE, 15)
    ruin_curves = {name: [] for name in STRATEGIES}
    
    for b in balances_to_test:
        for name, func in STRATEGIES.items():
            prob = calculate_ruin_probability(b, MAX_STEPS, func)
            ruin_curves[name].append(prob)
            
    # Compute Simpson's integral area
    aucs = {}
    domain_range = START_BALANCE - floor_base
    for name, curves in ruin_curves.items():
        auc_val = simpson(curves, x=balances_to_test) / domain_range
        aucs[name] = auc_val
        print(f"[{name}] Trailing Ruin AUC: {auc_val:.4f} | End Balance: ${walks[name][-1]:,.2f}")
        
    winner = min(aucs, key=aucs.get)
    print(f"-> WINNER FOR RUN #{retry_num}: {winner} Distribution")
    
    # Rendering
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left Axis: Dynamic Trailing Paths
    colors = {'Normal': 'blue', 'Exponential': 'orange', 'Weibull': 'green'}
    for name, path in walks.items():
        ax1.plot(path, label=f"{name} Path (Final: ${path[-1]:,.0f})", color=colors[name])
        ax1.plot(floors[name], label=f"{name} Floor", color=colors[name], linestyle=':', alpha=0.5)
        
    ax1.axhline(TAKE_PROFIT, color='purple', linestyle='--', alpha=0.8, label=f'Take-Profit (${TAKE_PROFIT:,})')
    ax1.set_title(f"Dynamic Trailing Stop Paths (Run #{retry_num})")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Balance ($)")
    ax1.legend(loc='upper left', bbox_to_anchor=(1,1))
    ax1.grid(True, alpha=0.3)
    
    # Right Axis: Integration Profiles
    for name, curves in ruin_curves.items():
        ax2.plot(balances_to_test, curves, label=f"{name} (AUC: {aucs[name]:.2f})", color=colors[name])
        ax2.fill_between(balances_to_test, curves, color=colors[name], alpha=0.1)
        
    ax2.set_title(f"Risk Curve over Entry Variance (Run #{retry_num})")
    ax2.set_xlabel("Starting Balance Floor Variant ($)")
    ax2.set_ylabel("Probability of Hitting Trailing Floor")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return winner

# 2. Outer Engine
if __name__ == "__main__":
    total_retries = 3
    results_summary = []
    
    for i in range(1, total_retries + 1):
        round_winner = run_game_iteration(retry_num=i)
        results_summary.append((i, round_winner))
        
    print("\n==========================================")
    print("FINAL TOURNAMENT RESULTS SUMMARY")
    print("==========================================")
    for num_retries, winner in results_summary:
        print(f"Retry Run #{num_retries} -> Winner: {winner} Distribution")
