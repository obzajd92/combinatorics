import numpy as np
import matplotlib.pyplot as plt

class InstitutionalRiskEngine:
    def __init__(self, initial_bankroll=2000.0, base_ticket_price=520.0, flat_fee=10.0):
        # Initial Capital Parameters
        self.initial_bankroll = initial_bankroll
        self.base_cost = base_ticket_price + flat_fee
        
        # Probabilities (Normal Win, Jackpot Win, Loss)
        self.p_normal = 0.49
        self.p_jackpot = 0.01
        self.p_loss = 0.50
        
        # Payout Multipliers (Before Frictions)
        self.payout_normal = 1000.0
        self.payout_jackpot_nominal = 10000.0
        
        # Progressive Tax Brackets
        self.tax_normal = 0.15
        self.tax_jackpot = 0.35
        
        # Risk Management Rules
        self.f_base = 0.0245               # Defensive 2.45% baseline fraction
        self.vault_threshold = 3000.0      # Target to sweep excess profits
        self.sliding_lock_pct = 0.75       # 75% trailing stop-loss threshold
        self.annual_bond_yield = 0.045     # 4.5% annualized secondary asset yield
        
    def _calculate_frictional_multipliers(self, counterparty_default_prob=0.02):
        """Calculates effective net payout multipliers after taxes, fees, and defaults."""
        effective_jackpot = self.payout_jackpot_nominal * (1.0 - counterparty_default_prob)
        
        mult_loss = -1.0
        mult_normal = ((self.payout_normal - self.base_cost) * (1.0 - self.tax_normal)) / self.base_cost
        mult_jackpot = ((effective_jackpot - self.base_cost) * (1.0 - self.tax_jackpot)) / self.base_cost
        return mult_loss, mult_normal, mult_jackpot

    def run_backtest(self, num_games=150, num_sims=10000, counterparty_default_prob=0.02):
        """Executes a multi-layered Monte Carlo analysis under full real-world constraints."""
        mult_loss, mult_normal, mult_jackpot = self._calculate_frictional_multipliers(counterparty_default_prob)
        
        # Pro-rata bond yield earned per game horizon (assuming 252 trading days equivalent context)
        yield_per_game = self.annual_bond_yield / 252
        
        np.random.seed(42)
        terminal_net_returns = []
        sample_trajectories = []
        
        for sim in range(num_sims):
            active_balance = self.initial_bankroll
            vault_balance = 0.0
            max_balance = self.initial_bankroll
            jackpot_hit = False
            consecutive_losses = 0
            path = [self.initial_bankroll]
            
            for game in range(num_games):
                # 1. Yield Accrual on the Secondary Asset Class
                if vault_balance > 0:
                    vault_balance *= (1.0 + yield_per_game)
                
                # 2. 75% Sliding Profit Trailing Floor Check
                if jackpot_hit:
                    accumulated_profit = max_balance - self.initial_bankroll
                    trailing_floor = self.initial_bankroll + (self.sliding_lock_pct * accumulated_profit)
                    if active_balance <= trailing_floor:
                        active_balance = trailing_floor
                        path.extend([active_balance + vault_balance] * (num_games - game))
                        break
                
                # 3. Capital Ruin Ceiling Check
                if active_balance < self.base_cost * self.f_base:
                    path.extend([active_balance + vault_balance] * (num_games - game))
                    break
                
                # 4. Drawdown Circuit Breaker Sizing Adjustment
                if consecutive_losses >= 5:
                    leverage_modifier = 0.25
                elif consecutive_losses >= 3:
                    leverage_modifier = 0.50
                else:
                    leverage_modifier = 1.0
                    
                wager = active_balance * self.f_base * leverage_modifier
                
                # 5. Core Game Engine Execution Loop
                roll = np.random.rand()
                if roll < self.p_jackpot:
                    if np.random.rand() < counterparty_default_prob:
                        active_balance += wager * mult_loss
                        consecutive_losses += 1
                    else:
                        active_balance += wager * mult_jackpot
                        jackpot_hit = True
                        consecutive_losses = 0
                elif roll < (self.p_jackpot + self.p_normal):
                    active_balance += wager * mult_normal
                    consecutive_losses = 0
                else:
                    active_balance += wager * mult_loss
                    consecutive_losses += 1
                
                # 6. Automated Rebalancing Sweep Routine Execution
                if active_balance > self.vault_threshold:
                    excess = active_balance - self.vault_threshold
                    vault_balance += excess
                    active_balance = self.vault_threshold
                    
                if (active_balance + vault_balance) > max_balance:
                    max_balance = active_balance + vault_balance
                    
                path.append(active_balance + vault_balance)
                
            terminal_net_returns.append(active_balance + vault_balance - self.initial_bankroll)
            if sim < 100:
                sample_trajectories.append(path)
                
        return np.array(terminal_net_returns), sample_trajectories

    def generate_dashboard(self, num_games=150, num_sims=10000):
        """Computes institutional risk metrics and renders full visualization suite."""
        net_returns, paths = self.run_backtest(num_games, num_sims)
        
        mean_ret = np.mean(net_returns)
        std_dev = np.std(net_returns)
        sharpe = mean_ret / std_dev if std_dev > 0 else 0
        
        downside_returns = net_returns[net_returns < 0]
        sortino = mean_ret / np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        print(f"=====================================================")
        print(f"      INSTITUTIONAL PERFORMANCE DASHBOARD WIDGET      ")
        print(f"=====================================================")
        print(f"Expected Portfolio Net Return : ${mean_ret:.2f}")
        print(f"Terminal Wealth Volatility    : ${std_dev:.2f}")
        print(f"Optimized Sharpe Ratio        : {sharpe:.3f}")
        print(f"Optimized Sortino Ratio       : {sortino:.3f}")
        print(f"Absolute System Max Winner    : +${np.max(net_returns):.2f}")
        print(f"Absolute System Max Loser     : ${np.min(net_returns):.2f}")
        print(f"=====================================================")

        # Render Visual Framework
        plt.figure(figsize=(10, 5))
        for p in paths:
            plt.plot(p, color='navy' if p[-1] > self.initial_bankroll else 'dimgray', alpha=0.2)
        plt.axhline(y=self.initial_bankroll, color='black', linestyle=':', label='Initial Capital')
        plt.title("Production Backtest: Dual-Asset Sweep Routine & Circuit Breaker Model")
        plt.xlabel("Games Horizon Block")
        plt.ylabel("Total Portfolio Net Worth ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

# --- Execution Entry Point ---
if __name__ == "__main__":
    engine = InstitutionalRiskEngine()
    engine.generate_dashboard(num_games=150, num_sims=10000)
