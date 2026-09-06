import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# 1. Core Configuration & Boundaries
COIN_PROBS = [0.45, 0.45, 0.10]  # P(Heads), P(Tails), P(In-between)
START_BALANCE = 10000
MAX_STEPS = 500
NUM_SIMULATIONS = 200  # Paths generated to calculate empirical risk

# Boundary constraints
STOP_LOSS = 2000       # Early termination if balance falls to/below this threshold
TAKE_PROFIT = 25000    # Early victory if balance hits/exceeds this threshold

def get_coin_flip():
    """Functional generator for the 3-faced coin outcome."""
    return np.random.choice(['H', 'T', 'I'], p=COIN_PROBS)

def normal_step(outcome):
    """Normal distribution return mapping."""
    base = 150 if outcome == 'H' else (-130 if outcome == 'T' else 10)
    return np.random.normal(loc=base, scale=50)

def exponential_step(outcome):
    """Exponential distribution return mapping (shifted for losses)."""
    if outcome == 'H':
        return np.random.exponential(scale=180)
    elif outcome == 'T':
        return -np.random.exponential(scale=150)
    return np.random.exponential(scale=15)

def weibull_step(outcome):
    """Weibull distribution return mapping (shifted for losses)."""
    shape = 1.5
    if outcome == 'H':
        return 160 * np.random.weibull(shape)
    elif outcome == 'T':
        return -140 * np.random.weibull(shape)
    return 15 * np.random.weibull(shape)

# Dictionary mapping for clean functional execution loops
STRATEGIES = {
    'Normal': normal_step,
    'Exponential': exponential_step,
    'Weibull': weibull_step
}

def simulate_walk(start_balance, steps, step_func):
    """Simulates a random walk path bounded by Stop-Loss and Take-Profit."""
    balance = start_balance
    history = [balance]
    for _ in range(steps):
        outcome = get_coin_flip()
        balance += step_func(outcome)
        
        # Enforce boundary checks
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
    """Runs a full simulation suite: walks, risk curves, AUC mapping, and plotting."""
    print(f"\n--- Running Game Iteration (Retry Run: #{retry_num}) ---")
    
    # Generate example random walks for visual tracking
    walks = {name: simulate_walk(START_BALANCE, MAX_STEPS, func) for name, func in STRATEGIES.items()}
    
    # Calculate Risk of Ruin Curves from the Stop-Loss floor to the Start Balance ceiling
    balances_to_test = np.linspace(STOP_LOSS, START_BALANCE, 15)
    ruin_curves = {name: [] for name in STRATEGIES}
    
    for b in balances_to_test:
        for name, func in STRATEGIES.items():
            prob = calculate_ruin_probability(b, MAX_STEPS, func)
            ruin_curves[name].append(prob)
            
    # Calculate Area Under Curve (AUC) for the bounded Risk of Ruin
    aucs = {}
    domain_range = START_BALANCE - STOP_LOSS
    for name, curves in ruin_curves.items():
        auc_val = simpson(curves, x=balances_to_test) / domain_range
        aucs[name] = auc_val
        print(f"[{name}] Bounded Ruin AUC: {auc_val:.4f} | Path End: ${walks[name][-1]:,.2f}")
        
    # The winner has the lowest overall Area Under the Ruin Curve
    winner = min(aucs, key=aucs.get)
    print(f"-> WINNER FOR RUN #{retry_num}: {winner} Distribution")
    
    # Render Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left Plot: Bounded Random Walks
    for name, path in walks.items():
        ax1.plot(path, label=f"{name} (Final: ${path[-1]:,.0f})")
    ax1.axhline(STOP_LOSS, color='red', linestyle='--', alpha=0.8, label=f'Stop-Loss (${STOP_LOSS:,})')
    ax1.axhline(TAKE_PROFIT, color='green', linestyle='--', alpha=0.8, label=f'Take-Profit (${TAKE_PROFIT:,})')
    ax1.set_title(f"Bounded Random Walk Paths (Run #{retry_num})")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Balance ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right Plot: Risk of Ruin AUC
    colors = {'Normal': 'blue', 'Exponential': 'orange', 'Weibull': 'green'}
    for name, curves in ruin_curves.items():
        ax2.plot(balances_to_test, curves, label=f"{name} (AUC: {aucs[name]:.2f})", color=colors[name])
        ax2.fill_between(balances_to_test, curves, color=colors[name], alpha=0.1)
        
    ax2.set_title(f"Risk of Ruin Curve & AUC Area (Run #{retry_num})")
    ax2.set_xlabel("Starting Balance ($)")
    ax2.set_ylabel("Probability of Hitting Stop-Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return winner

# 2. Automated Loop with 3 Retries
if __name__ == "__main__":
    total_retries = 3
    results_summary = []
    
    for i in range(1, total_retries + 1):
        round_winner = run_game_iteration(retry_num=i)
        results_summary.append((i, round_winner))
        
    print("\n==========================================")
    print("FINAL TOURNAMENT SUMMARY")
    print("==========================================")
    for num_retries, winner in results_summary:
        print(f"Retry Run #{num_retries} -> Winner: {winner} Distribution")
