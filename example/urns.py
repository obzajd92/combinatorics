import numpy as np
import matplotlib.pyplot as plt

def simulate_frictions(initial_bankroll=2000, num_games=100, p=0.48, tax_rate=0.25):
    ticket_price = 480
    payout = 1000
    
    # Calculate regular Kelly vs Taxed/Shifted Kelly
    q = 1 - p
    net_win_taxed = (payout - ticket_price) * (1 - tax_rate)
    b_taxed = net_win_taxed / ticket_price
    
    # Mathematical Kelly Percentage
    kelly_frac = (p * b_taxed - q) / b_taxed if b_taxed > 0 else 0
    
    # Force bet sizing to 0 if Kelly is negative (rational player stops)
    bet_fraction = max(0, kelly_frac)
    
    balances = [initial_bankroll]
    for _ in range(num_games):
        current_bal = balances[-1]
        if current_bal < ticket_price or bet_fraction == 0:
            balances.append(current_bal)
            continue
            
        bet_amount = current_bal * bet_fraction
        # Pro-rata scaling of ticket and payouts based on bet fraction
        scaled_ticket = bet_amount
        scaled_payout = bet_amount * (payout / ticket_price)
        
        if np.random.rand() < p:
            # Win scenario (apply tax to net profit)
            net_profit = scaled_payout - scaled_ticket
            taxed_profit = net_profit * (1 - tax_rate)
            balances.append(current_bal + taxed_profit)
        else:
            # Loss scenario
            balances.append(current_bal - scaled_ticket)
            
    return balances, kelly_frac

# Plotting the impact
plt.figure(figsize=(12, 6))
for run in range(10):
    bal_bad_prob, k1 = simulate_frictions(p=0.48, tax_rate=0.0)
    bal_taxed, k2 = simulate_frictions(p=0.50, tax_rate=0.25)
    
    if run == 0:
        plt.plot(bal_bad_prob, color='crimson', alpha=0.6, label=f'48% Win Rate (Kelly: {k1:.2%})')
        plt.plot(bal_taxed, color='darkorange', alpha=0.6, label=f'50% Win + 25% Tax (Kelly: {k2:.2%})')
    else:
        plt.plot(bal_bad_prob, color='crimson', alpha=0.2)
        plt.plot(bal_taxed, color='darkorange', alpha=0.2)

plt.title("The Destruction of Compounding via Frictions & Negative Kelly Constraints", fontsize=14)
plt.xlabel("Games Played", fontsize=12)
plt.ylabel("Bankroll ($)", fontsize=12)
plt.axhline(y=2000, color='gray', linestyle='--', alpha=0.5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
