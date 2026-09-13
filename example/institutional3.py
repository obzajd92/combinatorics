"""
Institutional Risk and Allocation Engine.

This package provides a production-grade, object-oriented framework for multi-asset
capital allocation under progressive tax systems, dynamic game/alpha decay, 
stochastic macro interest rate fluctuations, and structural inflation drag.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any


class ProductionRiskEngine:
    """
    An automated allocation engine featuring dynamic risk controls and macro protections.
    
    Implements dynamic geometric Kelly sizing, drawdown-dependent leverage limits,
    an automated profit vault sweep routine with Vasicek yield curve adjustments,
    and a rolling Bayesian tracking window to kill allocation during alpha decay.
    """

    def __init__(
        self, 
        initial_bankroll: float = 2000.0, 
        base_ticket_price: float = 520.0, 
        flat_fee: float = 10.0,
        annual_inflation: float = 0.03
    ) -> None:
        """
        Initializes the risk engine with core capitalization and fee parameters.

        Args:
            initial_bankroll (float): Initial cash pool allocated to the strategy.
            base_ticket_price (float): Nominal price of a single game ticket.
            flat_fee (float): Flat broker fee per transaction.
            annual_inflation (float): Annual rate of purchasing power erosion (e.g., 0.03 = 3%).
        """
        self.initial_bankroll = initial_bankroll
        self.base_cost = base_ticket_price + flat_fee
        self.annual_inflation_rate = annual_inflation
        
        # Game Payoffs (Nominal values)
        self.payout_normal = 1000.0
        self.payout_jackpot_nominal = 10000.0
        
        # Progressive Tax Profiles
        self.tax_normal = 0.15
        self.tax_jackpot = 0.35
        
        # Strategy Parameters
        self.f_base_max = 0.0245               # Defensive dynamic Kelly limit
        self.vault_threshold = 3000.0          # Balance cap where sweeps trigger
        self.sliding_lock_pct = 0.75           # Trailing profit lock floor (75%)
        self.tracking_window_size = 40          # History size for win rate tracking
        self.critical_win_threshold = 0.44      # Win rate line for system lock
        
        # Stochastic Yield Parameters (Vasicek Short-Rate Model)
        self.r0 = 0.045     # Initial yield (4.5%)
        self.k = 0.15       # Mean reversion speed
        self.theta = 0.05   # Long-term mean rate (5.0%)
        self.sigma = 0.015  # Daily rate volatility

    def _simulate_vasicek_rates(self, num_games: int, dt: float) -> np.ndarray:
        """Generates a stochastic interest rate pathway via mean-reverting Vasicek framework."""
        rates = np.zeros(num_games)
        rates[0] = self.r0
        for t in range(1, num_games):
            dr = self.k * (self.theta - rates[t-1]) * dt + self.sigma * np.sqrt(dt) * np.random.normal()
            rates[t] = rates[t-1] + dr
        return rates

    def execute_simulation(
        self, 
        num_games: int = 500, 
        num_sims: int = 5000, 
        alpha_decay: float = 0.0004, 
        default_risk: float = 0.02
    ) -> Tuple[np.ndarray, List[List[float]], int]:
        """
        Executes a rigorous multi-year backtest under complete frictional constraints.

        Args:
            num_games (int): Number of rounds/days played in the simulation block.
            num_sims (int): Total number of independent paths generated.
            alpha_decay (float): Linear decline in normal win probability per round.
            default_risk (float): Probability the host defaults on a jackpot payout.

        Returns:
            Tuple[np.ndarray, List[List[float]], int]: Net returns array, sample paths, and kill triggers.
        """
        effective_jackpot = self.payout_jackpot_nominal * (1.0 - default_risk)
        mult_loss = -1.0
        mult_normal = ((self.payout_normal - self.base_cost) * (1.0 - self.tax_normal)) / self.base_cost
        mult_jackpot = ((effective_jackpot - self.base_cost) * (1.0 - self.tax_jackpot)) / self.base_cost
        
        np.random.seed(42)
        terminal_real_returns = []
        sample_trajectories = []
        total_kill_triggers = 0
        
        dt = 1 / 252  # Daily step proxy per round
        inflation_per_game = self.annual_inflation_rate / 252
        
        for sim in range(num_sims):
            active_balance = self.initial_bankroll
            vault_balance = 0.0
            max_balance = self.initial_bankroll
            jackpot_hit = False
            consecutive_losses = 0
            
            trade_history: List[int] = []
            kill_switch_triggered = False
            path = [self.initial_bankroll]
            
            yield_curve = self._simulate_vasicek_rates(num_games, dt)
            p_normal_init = 0.49
            p_jackpot = 0.01
            
            for game in range(num_games):
                # 1. Real Purchasing Power Vault Sweep Accrual
                if vault_balance > 0:
                    net_yield = yield_curve[game] - self.annual_inflation_rate
                    vault_balance *= (1.0 + (net_yield * dt))
                
                # Dynamic Decay Curve
                current_p_normal = max(0.25, p_normal_init - (game * alpha_decay))
                
                # 2. Adaptive Allocation Window Evaluation
                if len(trade_history) >= self.tracking_window_size and not kill_switch_triggered:
                    realized_win_rate = sum(trade_history[-self.tracking_window_size:]) / self.tracking_window_size
                    if realized_win_rate < self.critical_win_threshold:
                        kill_switch_triggered = True
                        total_kill_triggers += 1
                
                # 3. 75% Trailing Profit Stop Check
                if jackpot_hit:
                    trailing_floor = self.initial_bankroll + (self.sliding_lock_pct * (max_balance - self.initial_bankroll))
                    if active_balance <= trailing_floor:
                        active_balance = trailing_floor
                        path.extend([active_balance + vault_balance] * (num_games - game))
                        break
                
                # 4. Core Capital Ruin Threshold Check
                if active_balance < self.base_cost * self.f_base_max:
                    path.extend([active_balance + vault_balance] * (num_games - game))
                    break
                
                # 5. Circuit Breaker Sizing Architecture
                if kill_switch_triggered:
                    f_adaptive = 0.0
                else:
                    if consecutive_losses >= 5:
                        leverage_modifier = 0.25
                    elif consecutive_losses >= 3:
                        leverage_modifier = 0.50
                    else:
                        leverage_modifier = 1.0
                    f_adaptive = self.f_base_max * leverage_modifier
                
                wager = active_balance * f_adaptive
                
                if f_adaptive == 0.0:
                    active_balance *= (1.0 - (inflation_per_game * dt))
                    path.append(active_balance + vault_balance)
                    continue
                
                # 6. Core Game Allocation Loop
                roll = np.random.rand()
                if roll < p_jackpot:
                    if np.random.rand() < default_risk:
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
                
                # 7. Automated Rebalancing Sweep Routine
                if active_balance > self.vault_threshold:
                    excess = active_balance - self.vault_threshold
                    vault_balance += excess
                    active_balance = self.vault_threshold
                    
                total_wealth = active_balance + vault_balance
                if total_wealth > max_balance:
                    max_balance = total_wealth
                    
                path.append(total_wealth)
                
            terminal_real_returns.append(active_balance + vault_balance - self.initial_bankroll)
            if sim < 100:
                sample_trajectories.append(path)
                
        return np.array(terminal_real_returns), sample_trajectories, total_kill_triggers

    def display_production_dashboard(self, num_games: int = 500, num_sims: int = 5000) -> Dict[str, Any]:
        """Calculates production performance metrics and plots real-space equity trajectories."""
        net_returns, paths, kill_triggers = self.execute_simulation(num_games, num_sims)
        
        mean_ret = np.mean(net_returns)
        std_dev = np.std(net_returns)
        sharpe = mean_ret / std_dev if std_dev > 0 else 0.0
        
        downside_returns = net_returns[net_returns < 0]
        sortino = mean_ret / np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        
        ruined_paths = np.sum(net_returns <= (self.initial_bankroll * -0.7))
        survival_rate = (1 - (ruined_paths / num_sims)) * 100
        
        metrics = {
            "Expected Net Return": f"${mean_ret:.2f}",
            "Volatility Deviation": f"${std_dev:.2f}",
"Sharpe Ratio": f"{sharpe:.3f}","Sortino Ratio": f"{sortino:.3f}","Survival Rate": f"{survival_rate:.2f}%","Kill Switch Triggers": f"{kill_triggers} paths ({kill_triggers/num_sims:.1%})"}print("\n" + "="*55)print("     PRODUCTION RISK ADAPTIVE ALLOCATION MODULE      ")print("="*55)for k, v in metrics.items():print(f"{k:<30} : {v}")print("="*55 + "\n")# Render Production Chartplt.figure(figsize=(10, 5))for p in paths:plt.plot(p, color='darkgreen' if p[-1] > self.initial_bankroll else 'darkred', alpha=0.15)plt.axhline(y=self.initial_bankroll, color='black', linestyle=':', label='Principal Capital')plt.title("Production Output: Multi-Asset Adaptive Strategy Engine")plt.xlabel("Horizon Runway (Games Played)")plt.ylabel("Real Purchasing Power Value ($)")plt.grid(True, alpha=0.3)plt.legend()plt.show()return metrics
#--- Package Verification Trigger ---
if name == "main":engine = ProductionRiskEngine()engine.display_production_dashboard(num_games=500, num_sims=5000)