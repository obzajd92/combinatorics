import numpy as np
import matplotlib.pyplot as plt

class AdaptiveInstitutionalEngine:
    def __init__(self, initial_bankroll=2000.0, base_ticket_price=520.0, flat_fee=10.0):
        self.initial_bankroll = initial_bankroll
        self.base_cost = base_ticket_price + flat_fee
        
        # Nominal Payout Matrix
        self.payout_normal = 1000.0
        self.payout_jackpot_nominal = 10000.0
        
        # Progressive Tax Vectors
        self.tax_normal = 0.15
        self.tax_jackpot = 0.35
        
        # Risk Management Settings
        self.f_base_max = 0.0245
        self.vault_threshold = 3000.0
        self.sliding_lock_pct = 0.75
        
        # Macro Frictions (Vasicek Yields & Hard Inflation Floor)
        self.r0 = 0.045
        self.k = 0.15
        self.theta = 0.05
        self.sigma = 0.015
        self.annual_inflation_rate = 0.030  # 3.0% structural inflation drag
        
        # Adaptive Threshold Rules
        self.tracking_window_size = 40      # Trailing 40-game window
        self.critical_win_threshold = 0.44  # 44% win rate kill-switch boundary

    def _simulate_vasicek_curve(self, num_games, dt=1/252):
        rates = np.zeros(num_games)
        rates[0] = self.r0
        for t in range(1, num_games):
            dr = self.k * (self.theta - rates[t-1]) * dt + self.sigma * np.sqrt(dt) * np.random.normal()
            rates[t] = rates[t-1] + dr
        return rates

    def run_adaptive_backtest(self, num_games=500, num_sims=5000, 
                              alpha_decay_rate=0.0004, counterparty_default_prob=0.02):
        effective_jackpot = self.payout_jackpot_nominal * (1.0 - counterparty_default_prob)
        mult_loss = -1.0
        mult_normal = ((self.payout_normal - self.base_cost) * (1.0 - self.tax_normal)) / self.base_cost
        mult_jackpot = ((effective_jackpot - self.base_cost) * (1.0 - self.tax_jackpot)) / self.base_cost
        
        np.random.seed(42)
        terminal_real_returns = []
        sample_trajectories = []
        
        dt = 1 / 252  # Time step proxy per game iteration
        inflation_factor_per_game = self.annual_inflation_rate / 252
        
        for sim in range(num_sims):
            active_balance = self.initial_bankroll
            vault_balance = 0.0
            max_balance = self.initial_bankroll
            jackpot_hit = False
            consecutive_losses = 0
            
            # Tracking history buffer for win/loss (1 for win, 0 for loss)
            trade_history = []
            kill_switch_triggered = False
            
            path = [self.initial_bankroll]
            yield_curve = self._simulate_vasicek_curve(num_games, dt)
            
            p_normal_init = 0.49
            p_jackpot = 0.01
            
            for game in range(num_games):
                # 1. Real Purchasing Power Inflation Adjustment on Vault
                if vault_balance > 0:
                    # Nominal yield addition minus the inflation decay drag
                    net_growth_rate = yield_curve[game] - self.annual_inflation_rate
                    vault_balance *= (1.0 + (net_growth_rate * dt))
                
                # Active decay scaling
                current_p_normal = max(0.25, p_normal_init - (game * alpha_decay_rate))
                
                # 2. Adaptive Allocation Window Verification Loop
                if len(trade_history) >= self.tracking_window_size and not kill_switch_triggered:
                    trailing_wins = sum(trade_history[-self.tracking_window_size:])
                    realized_win_rate = trailing_wins / self.tracking_window_size
                    
                    if realized_win_rate < self.critical_win_threshold:
                        kill_switch_triggered = True  # Trigger structural system lock
                
                # 3. 75% Trailing Profit Stop Lock
                if jackpot_hit:
                    accumulated_profit = max_balance - self.initial_bankroll
                    trailing_floor = self.initial_bankroll + (self.sliding_lock_pct * accumulated_profit)
                    if active_balance <= trailing_floor:
                        active_balance = trailing_floor
                        path.extend([active_balance + vault_balance] * (num_games - game))
                        break
                
                # 4. Standard Capital Ruin Barrier Check
                if active_balance < self.base_cost * self.f_base_max:
                    path.extend([active_balance + vault_balance] * (num_games - game))
                    break
                
                # 5. Adaptive Betting Sizing Engine
                if kill_switch_triggered:
                    f_adaptive = 0.0  # Dynamic protection allocation forced to 0%
                else:
                    if consecutive_losses >= 5:
                        leverage_modifier = 0.25
                    elif consecutive_losses >= 3:
                        leverage_modifier = 0.50
                    else:
                        leverage_modifier = 1.0
                    f_adaptive = self.f_base_max * leverage_modifier
                    
                wager = active_balance * f_adaptive
                
                # 6. Execution Step Loop
                if f_adaptive == 0:
                    # Capital stands idle, eroded solely by inflation drag
                    active_balance *= (1.0 - (inflation_factor_per_game * dt))
                    path.append(active_balance + vault_balance)
                    continue
                    
                roll = np.random.rand()
                if roll < p_jackpot:
                    if np.random.rand() < counterparty_default_prob:
                        active_balance += wager * mult_loss
                        consecutive_losses += 1
                        trade_history.append(0)
                    else:
                        active_balance += wager * mult_jackpot
                        jackpot_hit = True
                        consecutive_losses = 0
                        trade_history.append(1)
                elif roll < (p_jackpot + current_p_normal):
                    active_balance += wager * mult_normal
                    consecutive_losses = 0
                    trade_history.append(1)
                else:
                    active_balance += wager * mult_loss
                    consecutive_losses += 1
                    trade_history.append(0)
                
                # 7. Automated Sweep Routine
                if active_balance > self.vault_threshold:
                    excess = active_balance - self.vault_threshold
                    vault_balance += excess
                    active_balance = self.vault_threshold
                    
                total_portfolio_wealth = active_balance + vault_balance
                if total_portfolio_wealth > max_balance:
                    max_balance = total_portfolio_wealth
                    
                path.append(total_portfolio_wealth)
                
            terminal_real_returns.append(active_balance + vault_balance - self.initial_bankroll)
            if sim < 100:
                sample_trajectories.append(path)
                
        return np.array(terminal_net_returns), sample_trajectories

    def generate_adaptive_dashboard(self, num_games=500, num_sims=5000):
        net_real_returns, paths = self.run_adaptive_backtest(num_games, num_sims)
        
        mean_ret = np.mean(net_real_returns)
        std_dev = np.std(net_real_returns)
        sharpe = mean_ret / std_dev if std_dev > 0 else 0
        
        ruined_paths = np.sum(net_real_returns <= (self.initial_bankroll * -0.7))
        survival_rate = (1 - (ruined_paths / num_sims)) * 100

        print(f"=====================================================")
        print(f"      ADAPTIVE WINDOW & INFLATION ENGINE METRICS      ")
        print(f"=====================================================")
        print(f"Real Expected Net Return     : ${mean_ret:.2f}")
        print(f"Real Wealth Volatility (SD)  : ${std_dev:.2f}")
        print(f"Real Space Sharpe Ratio       : {sharpe:.3f}")
        print(f"Portfolio Survival Rate Floor: {survival_rate:.2f}%")
        print(f"Ultimate System Max Winner   : +${np.max(net_real_returns):.2f}")
        print(f"Ultimate System Max Loser    : ${np.min(net_real_returns):.2f}")
        print(f"=====================================================")

        plt.figure(figsize=(10, 5))
        for p in paths:
            plt.plot(p, color='darkgreen' if p[-1] > self.initial_bankroll else 'darkred', alpha=0.15)
        plt.axhline(y=self.initial_bankroll, color='black', linestyle=':', label='Starting Capital')
        plt.title("Adaptive Backtest: 40-Game Rolling Win Tracking & Inflation Drag Module")
        plt.xlabel("Extended Game Horizon Blocks (500 Round Runway)")
        plt.ylabel("Real Purchasing Power Value ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

# --- Execution Entry Point ---
if __name__ == "__main__":
    engine = AdaptiveInstitutionalEngine()
    engine.generate_adaptive_dashboard(num_games=500, num_sims=5000)
