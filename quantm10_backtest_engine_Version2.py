"""
Backtesting module for QuantM10

Provides backtesting and optimization functionality for trading strategies.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import itertools

from quantm10.exceptions import BacktestError
from quantm10.utils.logging import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, df: pd.DataFrame, params: Optional[Dict[str, Any]] = None, initial_capital: float = 100000.0):
        """
        Initialize backtesting engine
        
        Args:
            df: DataFrame with OHLCV data
            params: Backtest parameters
            initial_capital: Initial capital for backtest
        """
        self.df = df.copy()
        self.params = params or {}
        self.initial_capital = initial_capital
        self.results = None
        
        # For storing trades
        self.trades = []
        
        # For storing equity curve
        self.equity_curve = []
        
        # For storing drawdowns
        self.drawdowns = []
    
    def _get_param(self, param_name: str, default: Any = None) -> Any:
        """
        Get parameter value with fallback to default
        
        Args:
            param_name: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value
        """
        try:
            return self.params.get('backtest', {}).get(param_name, default)
        except (AttributeError, KeyError):
            return default
    
    def run_backtest(self, 
                    strategy_func: Callable, 
                    start_date: Optional[Union[str, datetime]] = None,
                    end_date: Optional[Union[str, datetime]] = None,
                    **strategy_params) -> Dict[str, Any]:
        """
        Run backtest for a strategy
        
        Args:
            strategy_func: Strategy function that returns signals
            start_date: Start date for backtest
            end_date: End date for backtest
            **strategy_params: Additional parameters for the strategy
            
        Returns:
            Dictionary with backtest results
            
        Raises:
            BacktestError: If backtest fails
        """
        try:
            logger.info("Running backtest...")
            
            # Convert dates to pandas datetime if provided
            if start_date is not None:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                self.df = self.df[self.df.index >= start_date]
            
            if end_date is not None:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                self.df = self.df[self.df.index <= end_date]
            
            # Reset index if it got changed
            if not isinstance(self.df.index, pd.DatetimeIndex):
                self.df.reset_index(inplace=True)
                self.df['date'] = pd.to_datetime(self.df['date'])
                self.df.set_index('date', inplace=True)
            
            # Check if enough data
            if len(self.df) < 10:
                raise BacktestError("Insufficient data for backtest (minimum 10 candles required)")
            
            # Set up backtest parameters
            position_size_pct = self._get_param('position_size_pct', 10.0)
            max_open_positions = self._get_param('max_open_positions', 5)
            include_commissions = self._get_param('include_commissions', True)
            commission_per_trade = self._get_param('commission_per_trade', 0.05) / 100  # Convert to decimal
            
            # Initialize variables
            self.equity_curve = [self.initial_capital]
            capital = self.initial_capital
            max_capital = self.initial_capital
            open_positions = []
            self.trades = []
            
            # Apply strategy function to generate signals
            self.df['signal'] = self.df.apply(lambda row: strategy_func(self.df.loc[:row.name], **strategy_params), axis=1)
            
            # Shift signals to avoid look-ahead bias
            self.df['signal'] = self.df['signal'].shift(1)
            
            # Main backtest loop
            for i in range(1, len(self.df)):
                current_date = self.df.index[i]
                current_row = self.df.iloc[i]
                
                # Process open positions first
                new_open_positions = []
                for pos in open_positions:
                    # Check stop loss
                    if (pos['type'] == 'LONG' and current_row['low'] <= pos['stop_loss']) or \
                       (pos['type'] == 'SHORT' and current_row['high'] >= pos['stop_loss']):
                        # Stop loss hit
                        exit_price = pos['stop_loss']
                        exit_reason = 'STOP_LOSS'
                        
                        # Calculate P&L
                        if pos['type'] == 'LONG':
                            pnl = (exit_price - pos['entry_price']) / pos['entry_price'] * pos['position_size']
                        else:
                            pnl = (pos['entry_price'] - exit_price) / pos['entry_price'] * pos['position_size']
                        
                        # Apply commission if enabled
                        if include_commissions:
                            commission = pos['position_size'] * commission_per_trade
                            pnl -= commission
                        
                        # Update capital
                        capital += pos['position_size'] + pnl
                        
                        # Record trade
                        self.trades.append({
                            'entry_date': pos['entry_date'],
                            'exit_date': current_date,
                            'entry_price': pos['entry_price'],
                            'exit_price': exit_price,
                            'position_size': pos['position_size'],
                            'type': pos['type'],
                            'pnl': pnl,
                            'pnl_pct': pnl / pos['position_size'] * 100,
                            'exit_reason': exit_reason
                        })
                    
                    # Check take profit
                    elif (pos['type'] == 'LONG' and current_row['high'] >= pos['take_profit']) or \
                         (pos['type'] == 'SHORT' and current_row['low'] <= pos['take_profit']):
                        # Take profit hit
                        exit_price = pos['take_profit']
                        exit_reason = 'TAKE_PROFIT'
                        
                        # Calculate P&L
                        if pos['type'] == 'LONG':
                            pnl = (exit_price - pos['entry_price']) / pos['entry_price'] * pos['position_size']
                        else:
                            pnl = (pos['entry_price'] - exit_price) / pos['entry_price'] * pos['position_size']
                        
                        # Apply commission if enabled
                        if include_commissions:
                            commission = pos['position_size'] * commission_per_trade
                            pnl -= commission
                        
                        # Update capital
                        capital += pos['position_size'] + pnl
                        
                        # Record trade
                        self.trades.append({
                            'entry_date': pos['entry_date'],
                            'exit_date': current_date,
                            'entry_price': pos['entry_price'],
                            'exit_price': exit_price,
                            'position_size': pos['position_size'],
                            'type': pos['type'],
                            'pnl': pnl,
                            'pnl_pct': pnl / pos['position_size'] * 100,
                            'exit_reason': exit_reason
                        })
                    
                    else:
                        # Position still open
                        new_open_positions.append(pos)
                
                open_positions = new_open_positions
                
                # Check for new signals
                current_signal = current_row['signal']
                
                # Enter new positions if conditions met
                if (current_signal == 1 or current_signal == -1) and len(open_positions) < max_open_positions:
                    # Calculate position size
                    pos_size = capital * (position_size_pct / 100) / max_open_positions
                    
                    if current_signal == 1:  # BUY signal
                        pos_type = 'LONG'
                        entry_price = current_row['open']
                        
                        # Calculate stop loss and take profit based on ATR if available
                        if 'atr' in current_row:
                            atr = current_row['atr']
                            stop_loss = entry_price - (atr * 1.5)
                            take_profit = entry_price + (atr * 2.5)
                        else:
                            # Default to fixed percentage
                            stop_loss = entry_price * 0.97
                            take_profit = entry_price * 1.05
                    
                    else:  # SELL signal
                        pos_type = 'SHORT'
                        entry_price = current_row['open']
                        
                        # Calculate stop loss and take profit based on ATR if available
                        if 'atr' in current_row:
                            atr = current_row['atr']
                            stop_loss = entry_price + (atr * 1.5)
                            take_profit = entry_price - (atr * 2.5)
                        else:
                            # Default to fixed percentage
                            stop_loss = entry_price * 1.03
                            take_profit = entry_price * 0.95
                    
                    # Apply commission if enabled
                    if include_commissions:
                        commission = pos_size * commission_per_trade
                        capital -= commission
                        pos_size -= commission
                    
                    # Create new position
                    position = {
                        'entry_date': current_date,
                        'entry_price': entry_price,
                        'position_size': pos_size,
                        'type': pos_type,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    
                    # Remove position size from available capital
                    capital -= pos_size
                    
                    # Add to open positions
                    open_positions.append(position)
                
                # Update equity curve
                total_equity = capital
                for pos in open_positions:
                    # Calculate current value of position
                    if pos['type'] == 'LONG':
                        current_value = pos['position_size'] * (current_row['close'] / pos['entry_price'])
                    else:
                        current_value = pos['position_size'] * (2 - current_row['close'] / pos['entry_price'])
                    
                    total_equity += current_value
                
                self.equity_curve.append(total_equity)
                
                # Update max capital for drawdown calculation
                if total_equity > max_capital:
                    max_capital = total_equity
                
                # Calculate drawdown
                drawdown = (max_capital - total_equity) / max_capital if max_capital > 0 else 0
                self.drawdowns.append(drawdown)
            
            # Close any remaining positions at the end
            for pos in open_positions:
                exit_price = self.df['close'].iloc[-1]
                exit_reason = 'END_OF_PERIOD'
                
                # Calculate P&L
                if pos['type'] == 'LONG':
                    pnl = (exit_price - pos['entry_price']) / pos['entry_price'] * pos['position_size']
                else:
                    pnl = (pos['entry_price'] - exit_price) / pos['entry_price'] * pos['position_size']
                
                # Apply commission if enabled
                if include_commissions:
                    commission = pos['position_size'] * commission_per_trade
                    pnl -= commission
                
                # Update capital
                capital += pos['position_size'] + pnl
                
                # Record trade
                self.trades.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': self.df.index[-1],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'position_size': pos['position_size'],
                    'type': pos['type'],
                    'pnl': pnl,
                    'pnl_pct': pnl / pos['position_size'] * 100,
                    'exit_reason': exit_reason
                })
            
            # Calculate performance metrics
            self.results = self.calculate_metrics()
            
            logger.info(f"Backtest completed with {len(self.trades)} trades")
            
            return self.results
        
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise BacktestError(str(e))
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.equity_curve:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0
            }
        
        # Total return
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital * 100
        
        # Annualized return
        days = (self.df.index[-1] - self.df.index[0]).days
        years = days / 365
        annualized_return = ((1 + total_return / 100) ** (1 / max(years, 0.01))) - 1
        
        # Sharpe ratio
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max drawdown
        max_drawdown = max(self.drawdowns) * 100 if self.drawdowns else 0
        
        # Win rate
        if self.trades:
            winning_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
            win_rate = winning_trades / len(self.trades) * 100
        else:
            win_rate = 0
        
        # Average profit and loss
        if self.trades:
            profits = [trade['pnl'] for trade in self.trades if trade['pnl'] > 0]
            losses = [trade['pnl'] for trade in self.trades if trade['pnl'] < 0]
            
            avg_profit = sum(profits) / len(profits) if profits else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            # Profit factor
            profit_factor = sum(profits) / abs(sum(losses)) if losses and sum(losses) != 0 else 0
        else:
            avg_profit = 0
            avg_loss = 0
            profit_factor = 0
        
        # Average trade duration
        if self.trades:
            durations = [(trade['exit_date'] - trade['entry_date']).days for trade in self.trades]
            avg_duration = sum(durations) / len(durations) if durations else 0
        else:
            avg_duration = 0
        
        return {
            'total_return': f"{total_return:.2f}%",
            'annualized_return': f"{annualized_return * 100:.2f}%",
            'sharpe_ratio': f"{sharpe_ratio:.2f}",
            'max_drawdown': f"{max_drawdown:.2f}%",
            'win_rate': f"{win_rate:.2f}%",
            'total_trades': len(self.trades),
            'winning_trades': sum(1 for trade in self.trades if trade['pnl'] > 0),
            'losing_trades': sum(1 for trade in self.trades if trade['pnl'] < 0),
            'avg_profit': f"{avg_profit:.2f}",
            'avg_loss': f"{avg_loss:.2f}",
            'profit_factor': f"{profit_factor:.2f}",
            'avg_trade_duration': f"{avg_duration:.1f} days",
            'initial_capital': self.initial_capital,
            'final_capital': self.equity_curve[-1] if self.equity_curve else self.initial_capital,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'drawdowns': self.drawdowns
        }
    
    def monte_carlo_simulation(self, num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation to assess strategy robustness
        
        Args:
            num_simulations: Number of simulations to run
            
        Returns:
            Dictionary with Monte Carlo results
        """
        if not self.trades:
            return {
                'return_range': '0% - 0%',
                'drawdown_range': '0% - 0%',
                'failure_rate': '0%'
            }
        
        logger.info(f"Running Monte Carlo simulation with {num_simulations} iterations")
        
        # Extract trade returns
        returns = [trade['pnl_pct'] / 100 for trade in self.trades]
        
        # Run simulations
        simulation_results = []
        
        for _ in range(num_simulations):
            # Shuffle returns
            shuffled_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate equity curve
            equity = [self.initial_capital]
            for ret in shuffled_returns:
                # Apply return to previous equity
                new_equity = equity[-1] * (1 + ret)
                equity.append(new_equity)
            
            # Calculate total return
            total_return = (equity[-1] - self.initial_capital) / self.initial_capital
            
            # Calculate max drawdown
            max_equity = 0
            max_dd = 0
            
            for eq in equity:
                if eq > max_equity:
                    max_equity = eq
                
                dd = (max_equity - eq) / max_equity if max_equity > 0 else 0
                if dd > max_dd:
                    max_dd = dd
            
            simulation_results.append({
                'total_return': total_return,
                'max_drawdown': max_dd
            })
        
        # Extract return and drawdown distributions
        returns = [result['total_return'] * 100 for result in simulation_results]
        drawdowns = [result['max_drawdown'] * 100 for result in simulation_results]
        
        # Calculate percentiles
        return_5th = np.percentile(returns, 5)
        return_95th = np.percentile(returns, 95)
        
        drawdown_5th = np.percentile(drawdowns, 5)
        drawdown_95th = np.percentile(drawdowns, 95)
        
        # Calculate failure rate (negative returns)
        failure_rate = sum(1 for ret in returns if ret < 0) / len(returns) * 100
        
        return {
            'return_range': f"{return_5th:.2f}% - {return_95th:.2f}%",
            'drawdown_range': f"{drawdown_5th:.2f}% - {drawdown_95th:.2f}%",
            'failure_rate': f"{failure_rate:.2f}%",
            'avg_return': f"{np.mean(returns):.2f}%",
            'avg_drawdown': f"{np.mean(drawdowns):.2f}%",
            'worst_case_return': f"{min(returns):.2f}%",
            'worst_case_drawdown': f"{max(drawdowns):.2f}%"
        }
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all possible parameter combinations from grid
        
        Args:
            param_grid: Dictionary of parameter names and possible values
            
        Returns:
            List of dictionaries with parameter combinations
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = []
        for values in itertools.product(*param_values):
            combination = dict(zip(param_names, values))
            combinations.append(combination)
        
        return combinations
    
    def _find_best_parameters(self, 
                            data: pd.DataFrame, 
                            strategy_func: Callable, 
                            param_combinations: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float, float]:
        """
        Find best parameters for a strategy on given data
        
        Args:
            data: DataFrame with OHLCV data
            strategy_func: Strategy function that returns signals
            param_combinations: List of parameter combinations to test
            
        Returns:
            Tuple of (best_params, best_metric, in_sample_return)
        """
        best_params = None
        best_metric = -float('inf')
        best_return = 0
        
        for params in param_combinations:
            # Initialize backtesting engine with this data
            engine = BacktestEngine(data, self.params, self.initial_capital)
            
            # Run backtest with current parameters
            try:
                result = engine.run_backtest(strategy_func, **params)
                
                # Extract key metrics
                total_return = float(result['total_return'].replace('%', '')) / 100
                sharpe = float(result['sharpe_ratio'])
                drawdown = float(result['max_drawdown'].replace('%', '')) / 100
                
                # Combine metrics into single score (higher is better)
                # We prioritize Sharpe ratio but also consider return and drawdown
                metric = sharpe * (1 + total_return) * (1 - drawdown)
                
                # Update best parameters if this is better
                if metric > best_metric:
                    best_metric = metric
                    best_params = params
                    best_return = total_return
                    
            except Exception as e:
                logger.warning(f"Parameter combination {params} failed: {str(e)}")
                continue
        
        return best_params, best_metric, best_return
    
    def walk_forward_optimization(self, 
                                strategy_func: Callable, 
                                param_grid: Dict[str, List[Any]],
                                in_sample_pct: float = 0.6,
                                step_size: float = 0.2) -> Dict[str, Any]:
        """
        Run walk-forward optimization to find optimal parameters
        
        Args:
            strategy_func: Strategy function that returns signals
            param_grid: Dictionary of parameter names and possible values
            in_sample_pct: Percentage of data to use for in-sample optimization
            step_size: Step size as percentage of data
            
        Returns:
            Dictionary with optimization results
            
        Raises:
            BacktestError: If optimization fails
        """
        try:
            logger.info("Running walk-forward optimization")
            
            # Generate all parameter combinations
            param_combinations = self._generate_param_combinations(param_grid)
            
            if not param_combinations:
                raise BacktestError("No parameter combinations to test")
            
            logger.info(f"Testing {len(param_combinations)} parameter combinations")
            
            # Calculate period boundaries
            total_periods = max(int((1 - in_sample_pct) / step_size), 1)
            period_results = []
            all_results = []
            
            # Store period boundaries for reference
            periods = []
            
            # Start with first chunk for training
            start_idx = 0
            period_size = int(len(self.df) * in_sample_pct)
            train_end_idx = period_size
            
            # Walk forward through the data
            for period in range(total_periods):
                logger.info(f"Optimizing period {period+1}/{total_periods}")
                
                # Define training and testing periods
                train_data = self.df.iloc[start_idx:train_end_idx]
                
                # Calculate test period size
                test_size = int(len(self.df) * step_size)
                test_start_idx = train_end_idx
                test_end_idx = min(test_start_idx + test_size, len(self.df))
                
                test_data = self.df.iloc[test_start_idx:test_end_idx]
                
                # Store period boundaries
                periods.append((
                    train_data.index[0].strftime('%Y-%m-%d'),
                    test_data.index[-1].strftime('%Y-%m-%d')
                ))
                
                if len(train_data) < 30 or len(test_data) < 10:
                    logger.warning(f"Insufficient data for period {period+1}, skipping")
                    continue
                
                # Find best parameters on training data
                best_params, best_metric, in_sample_return = self._find_best_parameters(
                    train_data, strategy_func, param_combinations
                )
                
                if best_params is None:
                    logger.warning(f"No valid parameters found for period {period+1}, skipping")
                    continue
                
                # Test best parameters on test data
                test_engine = BacktestEngine(test_data, self.params, self.initial_capital)
                test_result = test_engine.run_backtest(strategy_func, **best_params)
                
                # Extract key metrics
                out_sample_return = float(test_result['total_return'].replace('%', '')) / 100
                out_sample_drawdown = float(test_result['max_drawdown'].replace('%', '')) / 100
                out_sample_sharpe = float(test_result['sharpe_ratio'])
                
                # Combine metrics into single score (higher is better)
                out_sample_metric = out_sample_sharpe * (1 + out_sample_return) * (1 - out_sample_drawdown)
                
                # Store result for this period
                period_results.append({
                    'period': period + 1,
                    'in_sample_return': in_sample_return,
                    'out_sample_return': out_sample_return,
                    'out_sample_drawdown': out_sample_drawdown,
                    'out_sample_metric': out_sample_metric,
                    'best_params': best_params
                })
                
                all_results.append({
                    'period': period + 1,
                    'in_sample_return': in_sample_return,
                    'out_sample_return': out_sample_return,
                    'out_sample_metric': out_sample_metric,
                    'best_params': best_params
                })
                
                # Move to next period
                start_idx = int(start_idx + len(self.df) * step_size)
                train_end_idx = int(start_idx + len(self.df) * in_sample_pct)
                
                if train_end_idx >= len(self.df):
                    break
            
            if not period_results:
                raise BacktestError("No valid optimization periods")
            
            # Calculate average out-of-sample performance
            avg_out_sample_return = sum(result['out_sample_return'] for result in period_results) / len(period_results)
            avg_out_sample_drawdown = sum(result['out_sample_drawdown'] for result in period_results) / len(period_results)
            
            # Find the most common parameter values
            param_frequency = {}
            for result in period_results:
                for param, value in result['best_params'].items():
                    if param not in param_frequency:
                        param_frequency[param] = {}
                    
                    if value not in param_frequency[param]:
                        param_frequency[param][value] = 0
                    
                    param_frequency[param][value] += 1
            
            # Determine recommended parameters based on frequency
            recommended_params = {}
            for param, value_counts in param_frequency.items():
                # Find value with highest frequency
                recommended_params[param] = max(value_counts.items(), key=lambda x: x[1])[0]
            
            # Return all results
            return {
                'recommended_params': recommended_params,
                'avg_out_sample_return': avg_out_sample_return,
                'avg_out_sample_drawdown': avg_out_sample_drawdown,
                'periods': periods,
                'period_results': period_results,
                'all_results': all_results,
                'param_frequency': param_frequency
            }
        
        except Exception as e:
            logger.error(f"Error in walk-forward optimization: {str(e)}")
            raise BacktestError(f"Walk-forward optimization failed: {str(e)}")