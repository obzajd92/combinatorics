import numpy as np
import scipy.stats as stats

# Parameters
p_normal = 0.49
p_jackpot = 0.01
p_loss = 0.50

payout_normal = 1000
payout_jackpot = 10000
ticket_price = 520
flat_fee = 10
tax_rate = 0.25

# Flat fee scenario returns
ret_normal_fee = payout_normal - ticket_price - flat_fee
ret_jackpot_fee = payout_jackpot - ticket_price - flat_fee
ret_loss_fee = -ticket_price - flat_fee

# Tax scenario returns (no flat fee)
ret_normal_tax = (payout_normal - ticket_price) * (1 - tax_rate)
ret_jackpot_tax = (payout_jackpot - ticket_price) * (1 - tax_rate)
ret_loss_tax = -ticket_price

# Expected values
ev_fee = p_normal * ret_normal_fee + p_jackpot * ret_jackpot_fee + p_loss * ret_loss_fee
ev_tax = p_normal * ret_normal_tax + p_jackpot * ret_jackpot_tax + p_loss * ret_loss_tax

print(f"EV with flat fee: {ev_fee}")
print(f"EV with tax: {ev_tax}")

# Skewness calculation for jackpot game
returns = np.array([ret_normal_fee, ret_jackpot_fee, ret_loss_fee])
probs = np.array([p_normal, p_jackpot, p_loss])
mean = np.sum(returns * probs)
variance = np.sum(probs * (returns - mean)**2)
std_dev = np.sqrt(variance)
skewness = np.sum(probs * ((returns - mean) / std_dev)**3)

print(f"Mean: {mean}, Std Dev: {std_dev}, Skewness: {skewness}")
