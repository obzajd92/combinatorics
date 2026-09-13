import numpy as np

def analyze_game_boundaries(ticket_price, tax_rate, jackpot_multiplier=1.0):
    # Probabilities
    p_normal = 0.49
    p_jackpot = 0.01
    p_loss = 0.50
    
    # Payouts
    payout_normal = 1000
    payout_jackpot = 10000 * jackpot_multiplier
    
    # Calculate net returns per dollar risked
    # We define returns relative to the ticket price
    net_normal = (payout_normal - ticket_price) / ticket_price
    net_jackpot = (payout_jackpot - ticket_price) / ticket_price
    net_loss = -1.0
    
    # Apply taxes only to positive net returns
    if net_normal > 0: net_normal *= (1 - tax_rate)
    if net_jackpot > 0: net_jackpot *= (1 - tax_rate)
    
    # Calculate Expected Value (Breakeven check)
    ev = (p_normal * (net_normal * ticket_price)) + \
         (p_jackpot * (net_jackpot * ticket_price)) + \
         (p_loss * (net_loss * ticket_price))
         
    # Numerical Optimization to find the precise Kelly Fraction (f)
    # We maximize log-growth: E[ln(1 + f*R)]
    best_f = 0
    max_log_growth = 0
    
    # Scan possible allocation percentages from 0% to 50% of bankroll
    for f in np.linspace(0, 0.50, 1000):
        # Calculate log utility across all three possible states
        g_normal = np.log(1 + f * net_normal) if (1 + f * net_normal) > 0 else -np.inf
        g_jackpot = np.log(1 + f * net_jackpot) if (1 + f * net_jackpot) > 0 else -np.inf
        g_loss = np.log(1 + f * net_loss) if (1 + f * net_loss) > 0 else -np.inf
        
        expected_log_growth = (p_normal * g_normal) + (p_jackpot * g_jackpot) + (p_loss * g_loss)
        
        if expected_log_growth > max_log_growth:
            max_log_growth = expected_log_growth
            best_f = f
            
    return ev, best_f

# Test Scenario A: Flat game (No Jackpot multiplier) at $520 ticket
ev_a, kelly_a = analyze_game_boundaries(ticket_price=520, tax_rate=0.25, jackpot_multiplier=0.1) # Jackpot is just a normal ball
print(f"Standard Game EV: ${ev_a:.2f} | Recommended Kelly Size: {kelly_a:.2%}")

# Test Scenario B: Jackpot enabled (10x payout) at the exact same $520 ticket cost
ev_b, kelly_b = analyze_game_boundaries(ticket_price=520, tax_rate=0.25, jackpot_multiplier=1.0)
print(f"Jackpot Game EV: ${ev_b:.2f} | Recommended Kelly Size: {kelly_b:.2%}")
