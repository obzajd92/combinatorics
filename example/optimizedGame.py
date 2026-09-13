import numpy as np
import matplotlib.pyplot as plt

class DynamicMacroRiskEngine:
    def __init__(self, initial_bankroll=2000.0, base_ticket_price=520.0, flat_fee=10.0):
        self.initial_bankroll = initial_bankroll
        self.base_cost = base_ticket_price + flat_fee
        
        # Initial Payout Structure
        self.payout_normal = 1000.0
        self.payout_jackpot_nominal = 10000.0
        
        # Progressive Tax Frictions
        self.tax_normal = 0.15
        self.tax_jackpot = 0.35
        
        # Baseline Sizing & Strategy Controls
        self.f_base = 0.0245
        self.vault_threshold = 3000.0
        self.sliding_lock_pct = 0.75
        
        # Vasicek Bond Yield Curve Parameters (Macro Interest Rate Simulation)
        self.r0 = 0.045     # Starting interest rate (4.5%)
        self.k = 0.15       # Speed of mean reversion
        self.theta = 0.05   # Long-term mean interest rate (5.0%)
        self.sigma = 0.015  # Volatility of interest rates

    def _simulate_vasicek_curve(self, num_games, dt=1/252):
        """Generates a stochastic interest rate path over time via Vasicek model."""
        rates = np.zeros(num_games)
        rates[0] = self.r0
        for t in range(1, num_games):
            dr = self.k * (self.theta - rates[t-1]) * dt + self.sigma * np.sqrt(dt) * np.random.normal()
            rates[t] = rates[t-1] + dr
        return rates

    def run_stressed_backtest(self, num_games=500, num_sims=5000, 
                              alpha_decay_rate=0.0002, counterparty_default_prob=0.02):
        """
        Executes a multi-year backtest subjecting the portfolio to:
        1. Stochastic changing yield curves (Bonds)
        2. Dynamic alpha decay (Win probability shrinks each game)
        3. Counterparty jackpot default layers
        """
        # Effective net payout multiples relative to ticket cost
        effective_jackpot = self.payout_jackpot_nominal * (1.0 - counterparty_default_prob)
        mult_loss = -1.0
        mult_normal = ((self.payout_normal - self.base_cost) * (1.0 - self.tax_normal)) / self.base_cost
        mult_jackpot = ((effective_jackpot - self.base_cost) * (1.0 - self.tax_jackpot)) / self.base_cost
        
        np.random.seed(42)
        terminal_net_returns = []
        sample_trajectories = []
        macro_yield_paths = []
        
        dt = 1 / 252  # Daily step proxy per game
        
        for sim in range(num_sims):
            active_balance = self.initial_bankroll
            vault_balance = 0.0
            max_balance = self.initial_bankroll
            jackpot_hit = False
            consecutive_losses = 0
            path = [self.initial_bankroll]
            
            # Simulate this simulation run's macro interest rate environment
            yield_curve = self._simulate_vasicek_curve(num_games, dt)
            if sim < 5:  # Store a few yield curve paths for the dashboard
                macro_yield_paths.append(yield_curve)
            
            # Initial Probabilities
            p_normal_init = 0.49
            p_jackpot = 0.01
            
            for game in range(num_games):
                # 1. Update Dynamic Alpha Decay (Probability declines mid-sequence)
                current_p_normal = max(0.35, p_normal_init - (game * alpha_decay_rate))
                
                # 2. Vault Yield Accrual (Dynamic rate pulled from Vasicek curve)
                if vault_balance > 0:
                    vault_balance *= (1.0 + (yield_curve[game] * dt))
                
                # 3. 75% Trailing Profit Floor Check
                if jackpot_hit:
                    accumulated_profit = max_balance - self.initial_bankroll
                    trailing_floor = self.initial_bankroll + (self.sliding_lock_pct * accumulated_profit)
                    if active_balance <= trailing_floor:
                        active_balance = trailing_floor
                        path.extend([active_balance + vault_balance] * (num_games - game))
                        break
                
                # 4. Standard Capital Ruin Barrier Check
                if active_balance < self.base_cost * self.f_base:
                    path.extend([active_balance + vault_balance] * (num_games - game))
                    break
                
                # 5. Circuit Breaker Sizing Adjuster
                if consecutive_losses >= 5:
                    leverage_modifier = 0.25
                elif consecutive_losses >= 3:
                    leverage_modifier = 0.50
                else:
                    leverage_modifier = 1.0
                    
                wager = active_balance * self.f_base * leverage_modifier
                
                # 6. Core Game Allocation Check
                roll = np.random.rand()
                if roll < p_jackpot:
                    if np.random.rand() < counterparty_default_prob:
                        active_balance += wager * mult_loss
                        consecutive_losses += 1
                    else:
                        active_balance += wager * mult_jackpot
                        jackpot_hit = True
                        consecutive_losses = 0
                elif roll < (p_jackpot + current_p_normal):
                    active_balance += wager * mult_normal
                    consecutive_losses = 0
                else:
                    active_balance += wager * mult_loss
                    consecutive_losses += 1
                
                # 7. Automated Sweep Execution
                if active_balance > self.vault_threshold:
                    excess = active_balance - self.vault_threshold
                    vault_balance += excess
                    active_balance = self.vault_threshold
                    
                if (active_balance + vault_balance) > max_balance:
                    max_balance = active_balance + vault_balance
                    
                path.append(active_balance + vault_balance)
                
            terminal_net_returns.append(active_balance + vault_balance - self.initial_bankroll)
            if sim < 50:
                sample_trajectories.append(path)
                
        return np.array(terminal_net_returns), sample_trajectories, macro_yield_paths

    def generate_stress_dashboard(self, num_games=500, num_sims=5000, alpha_decay_rate=0.0004):
        """Computes stress metrics and renders dual visual panels for portfolios and macro rates."""
        net_returns, paths, yield_paths = self.run_stressed_backtest(num_games, num_sims, alpha_decay_rate)
        
        mean_ret = np.mean(net_returns)
        std_dev = np.std(net_returns)
        sharpe = mean_ret / std_dev if std_dev > 0 else 0
        
        ruined_paths = np.sum(net_returns <= (self.initial_bankroll * -0.7))
        survival_rate = (1 - (ruined_paths / num_sims)) * 100

        print(f"=====================================================")
        print(f"         MACRO STRESS-TEST ENGINE PROFILE            ")
        print(f"=====================================================")
        print(f"Stressed Expected Net Return : ${mean_ret:.2f}")
        print(f"Stressed Volatility Deviation: ${std_dev:.2f}")
        print(f"Stressed Sharpe Ratio        : {sharpe:.3f}")
        print(f"System Survival Rate Floor    : {survival_rate:.2f}%")
        print(f"Ultimate Stressed Max Winner : +${np.max(net_returns):.2f}")
        print(f"Ultimate Stressed Max Loser  : ${np.min(net_returns):.2f}")
        print(f"=====================================================")

        # Multi-panel charting execution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Panel 1: Stressed Equity Trajectories
        for p in paths:
            ax1.plot(p, color='crimson' if p[-1] < self.initial_bankroll else 'teal', alpha=0.2)
        ax1.axhline(y=self.initial_bankroll, color='black', linestyle=':', label='Starting Principal')
        ax1.set_title("Stressed Long-Term Portfolio Trajectories")
        ax1.set_xlabel("Extended Game Horizon Blocks (Years Equivalent)")
        ax1.set_ylabel("Total Wealth Pool ($)")
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Simulated Vasicek Interest Rate Yield Curves
        for y in yield_paths:
            ax2.plot(y * 100, color='darkorange', alpha=0.7)
        ax2.set_title("Simulated Vasicek Macro Yield Curve Evolution")
        ax2.set_xlabel("Time Horizon Steps")
        ax2.set_ylabel("Annualized Bond Yield (%)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# --- Execution Entry Point ---
if __name__ == "__main__":
    engine = DynamicMacroRiskEngine()
    # Stressing the engine across an extended 500-game window with active probability degradation
    engine.generate_stress_dashboard(num_games=500, num_sims=5000, alpha_decay_rate=0.0004)
