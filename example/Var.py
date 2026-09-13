import numpy as np

# Simulation & Game Parameters
p_normal, p_jackpot, p_loss = 0.49, 0.01, 0.50
payout_normal, payout_jackpot = 1000, 10000
ticket_price = 520
flat_fee = 10
tax_rate = 0.25

# Net profits per single trial
net_normal = (payout_normal - ticket_price - flat_fee) * (1 - tax_rate)
net_jackpot = (payout_jackpot - ticket_price - flat_fee) * (1 - tax_rate)
net_loss = -ticket_price - flat_fee

# 1. ANALYTICAL DOWNSIDE SEMI-VARIANCE (Target = 0 profit)
returns = np.array([net_normal, net_jackpot, net_loss])
probs = np.array([p_normal, p_jackpot, p_loss])
mean_return = np.sum(returns * probs)

# Only calculate squared deviation for outcomes below 0 net profit
downside_elements = np.where(returns < 0, probs * (returns - 0)**2, 0)
downside_semi_variance = np.sum(downside_elements)
downside_std_dev = np.sqrt(downside_semi_variance)

print(f"--- Single-Game Downside Profile ---")
print(f"Expected Net Return per Game: ${mean_return:.2f}")
print(f"Downside Semi-Standard Deviation: ${downside_std_dev:.2f}")

# 2. MULTI-GAME VALUE-AT-RISK (VaR) VIA MONTE CARLO
num_games_horizon = 20  # 20-game block horizon
num_simulations = 50000
horizon_returns = []

for _ in range(num_simulations):
    # Simulate a block of 20 games
    rolls = np.random.rand(num_games_horizon)
    outcomes = np.where(rolls < p_jackpot, net_jackpot, 
                        np.where(rolls < (p_jackpot + p_normal), net_normal, net_loss))
    horizon_returns.append(np.sum(outcomes))

horizon_returns = np.array(horizon_returns)

# Calculate VaR thresholds
var_95 = -np.percentile(horizon_returns, 5)
var_99 = -np.percentile(horizon_returns, 1)

print(f"\n--- {num_games_horizon}-Game Block Value-at-Risk (VaR) ---")
print(f"95% Confidence VaR: ${var_95:.2f}")
print(f"99% Confidence VaR: ${var_99:.2f}")
print(f"Interpretation: There is only a 5% chance of losing more than ${var_95:.2f} over a {num_games_horizon}-game run.")
