"""
Institutional Multi-Asset Portfolio Allocation & Macro Frictions Engine.

Provides an automated risk architecture managing high-skew alternative game setups 
under systemic alpha decay, progressive windfalls tax, stochastic interest rates,
and dynamic inflation-hedging asset swaps (Bonds vs. Commodities).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any


class InstitutionalProductionEngine:
    """
    Automated trading and capital preservation dashboard.
    
    Enforces dynamic geometric Kelly limits, consecutive-loss circuit breakers,
    automatic tracking windows, dynamic commodity-basket inflation hedging swaps,
    and a final liquidation engine converting nominal balances to absolute real cash.
    """

    def __init__(
        self,
        initial_bankroll: float = 2000.0,
        base_ticket_price: float = 520.0,
        flat_fee: float = 10.0,
        initial_inflation: float = 0.030,
        inflation_threshold: float = 0.045
    ) -> None:
        """Initializes capital baselines, progressive friction bounds, and macro triggers."""
        self.initial_bankroll = initial_bankroll
        self.base_cost = base_ticket_price + flat_fee
        
        # Nominal Payoff Parameters
        self.payout_normal = 1000.0
        self.payout_jackpot_nominal = 10000.0
        
        # Variable Progressive Tax Brackets
        self.tax_normal = 0.15
        self.tax_jackpot = 0.35
        
        # Sizing and Safety Limits
        self.f_base_max = 0.0245
        self.vault_threshold = 3000.0
        self.sliding_lock_pct = 0.75
        self.tracking_window_size = 40
        self.critical_win_threshold = 0.44
        
        # Macro Environment Setting Variables (Vasicek Model)
        self.r0 = 0.045
        self.k = 0.15
        self.theta = 0.05
        self.sigma = 0.015
        
        # Inflation & Commodity Swap Allocations
        self.initial_inflation = initial_inflation
        self.inflation_threshold = inflation_threshold  # Switch to commodities if inflation >= 4.5%
        self.commodity_alpha = 0.015                     # Outperformance premium over active inflation

    def _simulate_macro_paths(self, num_games: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generates interest rate and correlated macro inflation shocks stochastically."""
        rates = np.zeros(num_games)
        rates[0] = self.r0
        inflation = np.zeros(num_games)
        inflation[0] = self.initial_inflation
        
        for t in range(1, num_games):
            # Mean-reverting interest rate path
            dr = self.k * (self.theta - rates[t-1]) * dt + self.sigma * np.sqrt(dt) * np.random.normal()
            rates[t] = rates[t-1] + dr
            
            # Correlated macroeconomic inflation tracking drift
            d_inf = 0.1 * (0.035 - inflation[t-1]) * dt + 0.02 * np.sqrt(dt) * np.random.normal()
            inflation[t] = inflation[t-1] + d_inf
            
        return rates, inflation

    def execute_horizon_backtest(
        self, 
        num_games: int = 500, 
        num_sims: int = 5000, 
        alpha_decay: float = 0.0004, 
        default_risk: float = 0.02
    ) -> Tuple[np.ndarray, List[List[float]], Dict[str, float]]:
        """Runs the asset optimization engine across simulated multi-year horizons."""
        effective_jackpot = self.payout_jackpot_nominal * (1.0 - default_risk)
        mult_loss = -1.0
        mult_normal = ((self.payout_normal - self.base_cost) * (1.0 - self.tax_normal)) / self.base_cost
        mult_jackpot = ((effective_jackpot - self.base_cost) * (1.0 - self.tax_jackpot)) / self.base_cost
        
        np.random.seed(42)
        terminal_real_cash_payouts = []
        sample_paths = []
        
        dt = 1 / 252
        commodity_alloc_count = 0
        total_steps = num_sims * num_games
        
        for sim in range(num_sims):
            active_balance = self.initial_bankroll
            vault_balance = 0.0
            max_balance = self.initial_bankroll
            jackpot_hit = False
            consecutive_losses = 0
            
            trade_history: List[int] = []
            kill_switch_triggered = False
            path = [self.initial_bankroll]
            
            yield_curve, inflation_curve = self._simulate_macro_paths(num_games, dt)
            
            for game in range(num_games):
                current_inflation = inflation_curve[game]
                
                # 1. Dynamic Inflation Hedging Component
                if vault_balance > 0:
                    if current_inflation >= self.inflation_threshold:
                        # Asset Swap: Move vault capital out of Bonds and into Commodity Basket
                        nominal_vault_growth = current_inflation + self.commodity_alpha
                        commodity_alloc_count += 1
                    else:
                        # Standard Framework: Retain capital in stable Bond Vault
                        nominal_vault_growth = yield_curve[game]
                        
                    vault_balance *= (1.0 + (nominal_vault_growth * dt))
                
                # Alpha Decay probability shift
                current_p_normal = max(0.25, 0.49 - (game * alpha_decay))
                
                # 2. Adaptive Allocation Window Metric Check
                if len(trade_history) >= self.tracking_window_size and not kill_switch_triggered:
                    realized_win_rate = sum(trade_history[-self.tracking_window_size:]) / self.tracking_window_size
                    if realized_win_rate < self.critical_win_threshold:
                        kill_switch_triggered = True
                
                # 3. 75% Trailing Profit Stop Floor Check
                if jackpot_hit:
                    trailing_floor = self.initial_bankroll + (self.sliding_lock_pct * (max_balance - self.initial_bankroll))
                    if active_balance <= trailing_floor:
                        active_balance = trailing_floor
                        path.extend([active_balance + vault_balance] * (num_games - game))
                        break
                
                # 4. Standard Capital Ruin Barrier Check
                if active_balance < self.base_cost * self.f_base_max:
                    path.extend([active_balance + vault_balance] * (num_games - game))
                    break
                
                # 5. Circuit Breaker Sizing Implementation
                if kill_switch_triggered:
                    f_adaptive = 0.0
                else:
                    if consecutive_losses >= 5: leverage_modifier = 0.25
                    elif consecutive_losses >= 3: leverage_modifier = 0.50
                    else: leverage_modifier = 1.0
                    f_adaptive = self.f_base_max * leverage_modifier
                
                wager = active_balance * f_adaptive
                
                if f_adaptive == 0.0:
                    # Idle active funds degrade by inflation velocity
                    active_balance *= (1.0 - (current_inflation * dt))
                    path.append(active_balance + vault_balance)
                    continue
                
                # 6. Core Game Execution
                roll = np.random.rand()
                if roll < 0.01:
                    if np.random.rand() < default_risk:
                        active_balance += wager * mult_loss
                        consecutive_losses += 1
                        trade_history.append(0)
                    else:
                        active_balance += wager * mult_jackpot
                        jackpot_hit = True
                        consecutive_losses = 0
                        trade_history.append(1)
                elif roll < (0.01 + current_p_normal):
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
                
            # =========================================================================
            # 8. END-OF-HORIZON LIQUIDATION SUMMARY ENGINE
            # =========================================================================
            nominal_terminal_liquidation = active_balance + vault_balance
            
            # Discount the nominal total by cumulative inflation compound metrics to extract real cash value
            cumulative_inflation_drag = np.exp(np.sum(inflation_curve) * dt)
            absolute_real_cash_payout = nominal_terminal_liquidation / cumulative_inflation_drag
            
            terminal_real_cash_payouts.append(absolute_real_cash_payout)
            if sim < 100:
                sample_trajectories.append(path)
                
        allocation_analytics = {
            "commodity_utilization_rate": commodity_alloc_count / total_steps
        }
        
        return np.array(terminal_real_cash_payouts), sample_trajectories, allocation_analytics

    def render_system_dashboard(self, num_games: int = 500, num_sims: int = 5000) -> None:
        """Computes statistical metrics and displays the final production equity pathways."""
real_payouts, paths, analytics = self.execute_horizon_backtest(num_games, num_sims)mean_payout = np.mean(real_payouts)std_payout = np.std(real_payouts)sharpe = (mean_payout - self.initial_bankroll) / std_payout if std_payout > 0 else 0.0survival_rate = (np.sum(real_payouts >= (self.initial_bankroll * 0.3)) / num_sims) * 100print("\n" + "="*60)print("    PRODUCTION LIQUIDATION & DUAL-ASSET SWEP ENGINE    ")print("="*60)print(f"Expected Absolute Real Cash Payout : ${mean_payout:.2f}")print(f"Real Terminal Volatility Deviation  : ${std_payout:.2f}")print(f"Real Space Sharpe Metric Ratio     : {sharpe:.3f}")print(f"System Survival Rate Floor         : {survival_rate:.2f}%")print(f"Commodity Vault Swap Utilization   : {analytics['commodity_utilization_rate']:.2%}")print(f"Ultimate Absolute Max Winner Case  : +${np.max(real_payouts):.2f}")print(f"Ultimate Absolute Max Loser Case   : ${np.min(real_payouts):.2f}")print("="*60 + "\n")plt.figure(figsize=(11, 5))for p in paths:plt.plot(p, color='darkgreen' if p[-1] > self.initial_bankroll else 'firebrick', alpha=0.15)plt.axhline(y=self.initial_bankroll, color='black', linestyle=':', label='Starting Principal ($2,000)')plt.title("Production Backtest: Dynamic Inflation Hedging & Terminal Close-Out Summary")plt.xlabel("Extended Horizon Runway (Games Played / Daily Multi-Year Steps)")plt.ylabel("Real Purchasing Power Cash Equivalence ($)")plt.grid(True, alpha=0.3)plt.legend()plt.show()if name == "main":engine = InstitutionalProductionEngine()engine.render_system_dashboard(num_games=500, num_sims=5000)