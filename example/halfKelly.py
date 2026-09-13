import numpy as np
import matplotlib.pyplot as plt

# --- Setup Constants & Frictions ---
p_normal, p_jackpot, p_loss = 0.49, 0.01, 0.50
payout_normal, payout_jackpot = 1000, 10000
ticket_price = 520
flat_fee = 10
tax_rate = 0.25
initial_bankroll = 2000
num_games = 150
num_sims = 5000

base_cost = ticket_price + flat_fee
mult_normal = ((payout_normal - ticket_price - flat_fee) * (1 - tax_rate)) / base_cost
mult_jackpot = ((payout_jackpot - ticket_price - flat_fee) * (1 - tax_rate)) / base_cost
mult_loss = -1.0

# =========================================================================
# 1. OPTIMIZATION SEARCH (Finding the Sweet Spot)
# =========================================================================
# We scan risk fractions from 0.5% to 8.0% to maximize a custom "Risk-Adjusted Payout" metric:
# Metric = (Maximum Potential Return) + (Minimum Worst-Case Return)
best_fraction = 0
max_metric_score = -np.inf
optimal_stopping_game = 0

# For reproducibility
np.random.seed(42)

# Generate a single standardized set of random trajectories to test fractions fairly
random_matrix = np.random.rand(num_sims, num_games)

risk_fractions_to_test = np.linspace(0.005, 0.08, 50)

for f in risk_fractions_to_test:
    # Track the peak performance across all games for this specific fraction
    game_peaks = []
    
    # Simulate all paths for this fraction
    balances = np.zeros((num_sims, num_games + 1))
    balances[:, 0] = initial_bankroll
    
    for game in range(num_games):
        wager = balances[:, game] * f
        rolls = random_matrix[:, game]
        
        changes = np.where(rolls < p_jackpot, wager * mult_jackpot,
                           np.where(rolls < (p_jackpot + p_normal), wager * mult_normal, 
                                    wager * mult_loss))
        balances[:, game + 1] = balances[:, game] + changes

    # Calculate worst-case and best-case boundaries at the end of the run
    final_returns = balances[:, -1] - initial_bankroll
    max_return = np.max(final_returns)
    min_return = np.min(final_returns)
    
    # Calculate a risk-adjusted score (we want high upside but heavy penalties for deep losses)
    score = max_return + (min_return * 2.0) 
    
    if score > max_metric_score:
        max_metric_score = score
        best_fraction = f
        # Find the game index where the average winner path reaches its maximum absolute peak
        mean_top_10_percent_path = np.mean(np.sort(balances, axis=0)[-int(num_sims*0.1):, :], axis=0)
        optimal_stopping_game = np.argmax(mean_top_10_percent_path)

print(f"=== MATHEMATICAL OPTIMIZATION RESULTS ===")
print(f"Optimal Degraded Kelly Risk Percentage: {best_fraction*100:.2f}% of balance")
print(f"Optimal Strategy Stopping Point: Game {optimal_stopping_game}")

# =========================================================================
# 2. RERUNNING SCENARIO WITH THE OPTIMAL VALUE
# =========================================================================
plt.figure(figsize=(12, 6))
opt_balances = np.zeros((num_sims, num_games + 1))
opt_balances[:, 0] = initial_bankroll

for game in range(num_games):
    wager = opt_balances[:, game] * best_fraction
    rolls = random_matrix[:, game]
    changes = np.where(rolls < p_jackpot, wager * mult_jackpot,
                       np.where(rolls < (p_jackpot + p_normal), wager * mult_normal, 
                                wager * mult_loss))
    opt_balances[:, game + 1] = opt_balances[:, game] + changes

# Plot a clear visual sample of paths
for i in range(150):
    final_bal = opt_balances[i, -1]
    color = 'emerald' if final_bal > initial_bankroll else 'slate'
    plt.plot(opt_balances[i, :], color='teal' if final_bal > initial_bankroll else 'gray', alpha=0.2)

opt_max = np.max(opt_balances) - initial_bankroll
opt_min = np.min(opt_balances) - initial_bankroll

# Chart Layout Customization
plt.axvline(x=optimal_stopping_game, color='darkorange', linestyle='--', linewidth=2, 
            label=f'Optimal Peak Stopping Game (Game {optimal_stopping_game})')
plt.axhline(y=initial_bankroll, color='black', linestyle=':', label='Starting Capital ($2,000)')
plt.title(f"Optimized Geometric Kelly Strategy ({best_fraction*100:.2f}% Dynamic Risk)\nMax Peak Generated: +${opt_max:.2f} | Absolute Minimum Floor: ${opt_min:.2f}", fontsize=12)
plt.xlabel("Games Played", fontsize=11)
plt.ylabel("Player Balance ($)", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
