import numpy as np
import matplotlib.pyplot as plt

# --- Setup Parameters ---
p_normal, p_jackpot, p_loss = 0.49, 0.01, 0.50
payout_normal, payout_jackpot = 1000, 10000
ticket_price = 520
flat_fee = 10
tax_rate = 0.25
initial_bankroll = 2000
num_games = 150
num_sims = 10000

f_kelly = 0.0508  # 5.08% Kelly Fraction

# Net multipliers relative to a $1 base wager
base_cost = ticket_price + flat_fee
mult_normal = ((payout_normal - ticket_price - flat_fee) * (1 - tax_rate)) / base_cost
mult_jackpot = ((payout_jackpot - ticket_price - flat_fee) * (1 - tax_rate)) / base_cost
mult_loss = -1.0

np.random.seed(42)
final_geo_balances = []
plt.figure(figsize=(11, 5))

for sim in range(num_sims):
    geo_balance = [initial_bankroll]
    
    for game in range(num_games):
        current_bal = geo_balance[-1]
        
        # Calculate dynamic bet size based on current capital
        wager_amount = current_bal * f_kelly
        
        # Run random draw
        roll = np.random.rand()
        if roll < p_jackpot:
            change = wager_amount * mult_jackpot
        elif roll < (p_jackpot + p_normal):
            change = wager_amount * mult_normal
        else:
            change = wager_amount * mult_loss
            
        geo_balance.append(current_bal + change)
        
    final_geo_balances.append(geo_balance[-1])
    
    # Plot first 100 trajectories for visualization
    if sim < 100:
        plt.plot(geo_balance, color='teal', alpha=0.15)

final_geo_balances = np.array(final_geo_balances)
geo_max_return = np.max(final_geo_balances) - initial_bankroll
geo_min_return = np.min(final_geo_balances) - initial_bankroll
geo_ruin_rate = (np.sum(final_geo_balances < base_cost * f_kelly) / num_sims) * 100

# Plot presentation
plt.axhline(y=initial_bankroll, color='black', linestyle=':', alpha=0.5, label='Starting Capital ($2,000)')
plt.title(f"Degraded Geometric Kelly Simulation (Dynamic 5.08% Risk)\nMax: +${geo_max_return:.2f} | Min: ${geo_min_return:.2f} | Ruin Rate: {geo_ruin_rate:.2f}%")
plt.xlabel("Games Played")
plt.ylabel("Player Balance ($)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
