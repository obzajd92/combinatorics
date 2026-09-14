import numpy as np
import pandas as pd
import scipy.stats as si

class AdvancedGreekRiskEngine:
    def __init__(self, S0=100, K=105, T=0.25, r=0.045, sigma=0.25, 
                 init_cash=10000, num_options=500, sweep_threshold=1.15):
        """
        Advanced Dynamic Greek Hedging & Leverage Control Engine
        K                : Strike price of the options contract
        T                : Time to maturity (in years)
        r                : Risk-free rate
        num_options      : Number of options contracts held long (positive delta buffer)
        sweep_threshold  : Vault sweep multiplier above historical ATH
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.init_cash = init_cash
        self.num_options = num_options
        self.sweep_threshold = sweep_threshold
        
    def _bs_greeks(self, S, t, total_steps):
        """Calculates Black-Scholes exact option value, delta, and gamma."""
        time_to_mat = max(1e-5, self.T - (t / total_steps) * self.T)
        
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * time_to_mat) / (self.sigma * np.sqrt(time_to_mat))
        d2 = d1 - self.sigma * np.sqrt(time_to_mat)
        
        call_price = S * si.norm.cdf(d1) - self.K * np.exp(-self.r * time_to_mat) * si.norm.cdf(d2)
        delta = si.norm.cdf(d1)
        gamma = si.norm.pdf(d1) / (S * self.sigma * np.sqrt(time_to_mat))
        
        return call_price, delta, gamma

    def generate_short_squeeze_path(self, steps=100, mu=0.05, jump_size=0.35, jump_step=65):
        """Generates an asset trajectory featuring a massive short squeeze event."""
        np.random.seed(88)
        dt = 1 / steps
        S = np.zeros(steps)
        S[0] = self.S0
        
        for t in range(1, steps):
            epsilon = np.random.normal(0, 1)
            S[t] = S[t-1] * np.exp((mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * epsilon)
            
            # Inject instantaneous right-tail jump event (Short Squeeze)
            if t == jump_step:
                S[t] = S[t] * (1 + jump_size)
        return S

    def run_backtest(self, steps=100, max_allowable_drawdown=0.20):
        S = self.generate_short_squeeze_path(steps=steps)
        
        # State tracking arrays
        cash_vault = np.zeros(steps)
        options_val = np.zeros(steps)
        short_liability = np.zeros(steps)
        total_equity = np.zeros(steps)
        short_units = np.zeros(steps)
        drawdown_mult = np.ones(steps)
        
        # Initial State
        opt_price, initial_delta, _ = self._bs_greeks(S[0], 0, steps)
        options_val[0] = opt_price * self.num_options
        cash_vault[0] = self.init_cash
        
        # Target an unhedged short architecture baseline adjusted by delta
        short_units[0] = self.num_options * initial_delta * 1.10 # Intentional slight over-short
        short_liability[0] = short_units[0] * S[0]
        total_equity[0] = cash_vault[0] + options_val[0] - short_liability[0]
        
        peak_equity = total_equity[0]
        options_ath = options_val[0]
        sweep_count = 0
        
        for t in range(1, steps):
            opt_price, live_delta, live_gamma = self._bs_greeks(S[t], t, steps)
            options_val[t] = opt_price * self.num_options
            
            # 1. Update Portfolio Peak and Current Peak-to-Trough Drawdown
            current_mid_equity = cash_vault[t-1] + options_val[t] - (short_units[t-1] * S[t])
            if current_mid_equity > peak_equity:
                peak_equity = current_mid_equity
            
            current_drawdown = max(0.0, (peak_equity - current_mid_equity) / peak_equity)
            
            # 2. Compute Drawdown-Dependent Multiplier
            # Linear decay scaling down leverage as drawdown approaches maximum limit
            if current_drawdown > 0:
                drawdown_mult[t] = max(0.0, 1.0 - (current_drawdown / max_allowable_drawdown))
            else:
                drawdown_mult[t] = 1.0
                
            # 3. Dynamic Delta-Hedging Execution Loop with Gamma adjustments
            # The short units target tracks options delta, boosted by a look-ahead gamma buffer
            gamma_buffer = 0.5 * live_gamma * (S[t] * 0.05)
            target_short_units = self.num_options * (live_delta + gamma_buffer)
            
            # Apply the drawdown risk circuit-breaker to actively shrink liability exposure
            short_units[t] = target_short_units * drawdown_mult[t]
            short_liability[t] = short_units[t] * S[t]
            
            # Rebalance transactional cash layer based on short adjustments
            short_trade_adjustment = (short_units[t] - short_units[t-1]) * S[t]
            cash_vault[t] = cash_vault[t-1] + short_trade_adjustment
            
            # 4. Auto-Sweep Vault Trigger
            if options_val[t] > (options_ath * self.sweep_threshold):
                excess_profit = options_val[t] - options_ath
                options_val[t] -= excess_profit
                cash_vault[t] += excess_profit
                options_ath = options_val[t]
                sweep_count += 1
                
            # 5. Composite Portfolio Net Asset Value
            total_equity[t] = cash_vault[t] + options_val[t] - short_liability[t]
            
        return pd.DataFrame({
            'Spot': S,
            'Delta_Exposure': short_units,
            'Drawdown_Multiplier': drawdown_mult,
            'Cash_Vault': cash_vault,
            'Options_Layer': options_val,
            'Total_Portfolio_Equity': total_equity
        }), sweep_count

# Execute dual risk mitigators
engine = AdvancedGreekRiskEngine()
df_results, total_sweeps = engine.run_backtest()

print(f"Simulation Complete.")
print(f"Total dynamic rebalancing auto-sweeps triggered: {total_sweeps}")
print(df_results.iloc[[0, 64, 65, 66, -1]].to_string())
