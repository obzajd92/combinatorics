import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# 1. Configuration & Bounded Thresholds
COIN_PROBS = [0.35, 0.55, 0.10]  # P(Heads), P(Tails), P(In-between)
START_BALANCE = 10000
MAX_STEPS = 500
NUM_SIMULATIONS = 200  

# Boundaries
STOP_LOSS = 2000       
TAKE_PROFIT = 25000    

# --- THE TAIL DRAG ENGINE ---
# A compounding per-step friction cost (representing market slippage, fees, or continuous bleed)
TAIL_DRAG_PER_STEP = 35  

def get_coin_flip():
    """Functional generator for the 3-faced coin outcome."""
    return np.random.choice(['H', 'T', 'I'], p=COIN_PROBS)

def normal_step(outcome):
    """Asymmetric Normal: Moderate reliable gains vs penalizing losses."""
    if outcome == 'H':
        return np.random.normal(loc=120, scale=30)   
    elif outcome == 'T':
        return np.random.normal(loc=-180, scale=90)  
    return np.random.normal(loc=10, scale=10)

def exponential_step(outcome):
    """Asymmetric Exponential: High-magnitude right-tail spikes vs steady downside bleed."""
    if outcome == 'H':
        return np.random.exponential(scale=350)     
    elif outcome == 'T':
        return -np.random.exponential(scale=110)    
    return np.random.exponential(scale=10)

def weibull_step(outcome):
    """Asymmetric Weibull: Heavy sub-exponential black swan crashes."""
    if outcome == 'H':
        return 140 * np.random.weibull(a=1.2)       
    elif outcome == 'T':
        return -220 * np.random.weibull(a=0.7)      
    return 15 * np.random.weibull(a=1.0)

# Functional strategy mapping dictionary
STRATEGIES = {
    'Normal': normal_step,
    'Exponential': exponential_step,
    'Weibull': weibull_step
}

def simulate_walk(start_balance, steps, step_func):
    """Simulates a path with asymmetry, bounded by Stop-Loss, Take-Profit, and Tail Drag."""
    balance = start_balance
    history = [balance]
    for _ in range(steps):
        outcome = get_coin_flip()
        
        # Apply distribution payout adjusted for ongoing friction
        balance += (step_func(outcome) - TAIL_DRAG_PER_STEP)
        
        if balance <= STOP_LOSS:
            history.append(STOP_LOSS)
            break
        if balance >= TAKE_PROFIT:
            history.append(TAKE_PROFIT)
            break
            
        history.append(balance)
    return np.array(history)

def calculate_ruin_probability(initial_balance, steps, step_func, num_sims=NUM_SIMULATIONS):
    """Calculates empirical probability of hitting stop-loss given a starting balance."""
    if initial_balance <= STOP_LOSS:
        return 1.0
        
    ruined_count = 0
    for _ in range(num_sims):
        walk = simulate_walk(initial_balance, steps, step_func)
        if walk[-1] <= STOP_LOSS:
            ruined_count += 1
    return ruined_count / num_sims

def run_game_iteration(retry_num):
    """Executes a full simulation cycle: sample paths, ruin probability testing, and AUC ranking."""
    print(f"\n--- Running Tail Drag Simulation (Run: #{retry_num}) ---")
    
    # Generate sample paths for plotting
    walks = {name: simulate_walk(START_BALANCE, MAX_STEPS, func) for name, func in STRATEGIES.items()}
    
    # Track metrics over a range of test initial balances
    balances_to_test = np.linspace(STOP_LOSS, START_BALANCE, 15)
    ruin_curves = {name: [] for name in STRATEGIES}
    
    for b in balances_to_test:
        for name, func in STRATEGIES.items():
            prob = calculate_ruin_probability(b, MAX_STEPS, func)
            ruin_curves[name].append(prob)
            
    # Calculate Simpson's Integration for Area Under the Curve (Lower AUC = Winner)
    aucs = {}
    domain_range = START_BALANCE - STOP_LOSS
    for name, curves in ruin_curves.items():
        auc_val = simpson(curves, x=balances_to_test) / domain_range
        aucs[name] = auc_val
        print(f"[{name}] Bounded Ruin AUC: {auc_val:.4f} | Final Sample Balance: ${walks[name][-1]:,.2f}")
        
    # Selection of winner via the most optimal survival metrics (minimum ruin risk surface area)
    winner = min(aucs, key=aucs.get)
    print(f"-> WINNER FOR RUN #{retry_num}: {winner} Distribution")
    
    # Visualization setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left Axis: Bounded Paths
    for name, path in walks.items():
        ax1.plot(path, label=f"{name} (Final: ${path[-1]:,.0f})")
    ax1.axhline(STOP_LOSS, color='red', linestyle='--', alpha=0.8, label=f'Stop-Loss (${STOP_LOSS:,})')
    ax1.axhline(TAKE_PROFIT, color='green', linestyle='--', alpha=0.8, label=f'Take-Profit (${TAKE_PROFIT:,})')
    ax1.set_title(f"Random Walks with Tail Drag Attrition (Run #{retry_num})")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Balance ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right Axis: Risk Surface Area Curves
    colors = {'Normal': 'blue', 'Exponential': 'orange', 'Weibull': 'green'}
    for name, curves in ruin_curves.items():
        ax2.plot(balances_to_test, curves, label=f"{name} (AUC: {aucs[name]:.2f})", color=colors[name])
        ax2.fill_between(balances_to_test, curves, color=colors[name], alpha=0.1)
        
    ax2.set_title(f"Risk of Ruin Curve via Friction (Run #{retry_num})")
    ax2.set_xlabel("Starting Balance ($)")
    ax2.set_ylabel("Probability of Hitting Stop-Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return winner

# 2. Outer Automation Engine (3 Independent Retries)
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
