"""
Main entry point for QuantM10 trading bot

Handles command-line arguments and schedules bot execution.

Author: Rahul Reddy Allu
Version: 2.0.0
Date: 2025-05-20
"""
import sys
import os
import logging
import argparse
import asyncio
import time
import yaml
from datetime import datetime, timedelta
import traceback
from pathlib import Path

# Set up base paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Add project to path if running as script
if __name__ == "__main__" and str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from quantm10.app.trading_bot import TradingBot
from quantm10.config import ConfigManager
from quantm10.utils.logging import setup_logging, get_logger
from quantm10.utils.database import init_db


# Configure logger
logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="QuantM10 Algorithmic Trading Bot")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # 'run' command
    run_parser = subparsers.add_parser("run", help="Run the trading bot once")
    run_parser.add_argument("--config", type=str, help="Path to configuration file")
    run_parser.add_argument("--stock", type=str, help="Run for a specific stock only")
    
    # 'scheduled' command
    scheduled_parser = subparsers.add_parser("scheduled", help="Run the bot on a schedule")
    scheduled_parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # 'backtest' command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest for a stock")
    backtest_parser.add_argument("instrument_key", type=str, help="Instrument key to backtest")
    backtest_parser.add_argument("--days", type=int, default=250, help="Number of days for backtest")
    backtest_parser.add_argument("--config", type=str, help="Path to configuration file")
    backtest_parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    
    # 'multi' command
    multi_parser = subparsers.add_parser("multi", help="Run backtest for multiple stocks")
    multi_parser.add_argument("--days", type=int, default=250, help="Number of days for backtest")
    multi_parser.add_argument("--config", type=str, help="Path to configuration file")
    multi_parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    
    # 'optimize' command
    optimize_parser = subparsers.add_parser("optimize", help="Run parameter optimization")
    optimize_parser.add_argument("instrument_key", type=str, help="Instrument key to optimize")
    optimize_parser.add_argument("--config", type=str, help="Path to configuration file")
    optimize_parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    
    return parser.parse_args()


async def run_bot(config_file: str = None, stock: str = None):
    """
    Run the trading bot once
    
    Args:
        config_file: Path to configuration file
        stock: Specific stock to analyze (None for all)
    """
    # Load configuration
    config_manager = ConfigManager(config_file)
    config = config_manager.get_config()
    
    # Initialize bot
    bot = TradingBot(config)
    
    try:
        # Initialize components
        await bot.initialize()
        
        if stock:
            # Run for a specific stock
            signal = await bot.generate_signal(stock)
            await bot.send_signal_alert(signal)
            
            # Run backtest if signal is actionable
            if signal.get('actionable', False):
                backtest_results = await bot.run_backtest(stock)
                await bot.send_backtest_report(backtest_results)
                
            return [signal] if signal.get('actionable', False) else []
        else:
            # Run for all stocks
            return await bot.analyze_all_stocks()
    
    except Exception as e:
        logger.error(f"Error running trading bot: {str(e)}")
        logger.error(traceback.format_exc())
        return []


async def run_scheduled(config_file: str = None):
    """
    Run the trading bot on a schedule
    
    Args:
        config_file: Path to configuration file
    """
    # Load configuration
    config_manager = ConfigManager(config_file)
    config = config_manager.get_config()
    
    logger.info("Starting scheduled execution")
    
    # Get schedule times
    market_open = config.schedule.market_open
    mid_day = config.schedule.mid_day
    pre_close = config.schedule.pre_close
    post_market = config.schedule.post_market
    
    # Run on startup if configured
    if config.schedule.run_on_startup:
        logger.info("Running initial analysis on startup")
        await run_bot(config_file)
    
    # Main scheduling loop
    while True:
        try:
            # Get current time
            current_time = datetime.now().strftime('%H:%M')
            
            # Skip weekends if configured
            if not config.schedule.run_on_weekends:
                current_day = datetime.now().weekday()
                if current_day >= 5:  # 5=Saturday, 6=Sunday
                    logger.info("Weekend detected, sleeping for 1 hour")
                    await asyncio.sleep(3600)  # Sleep for 1 hour
                    continue
            
            # Check schedule times
            if current_time in [market_open, mid_day, pre_close, post_market]:
                logger.info(f"Scheduled run at {current_time}")
                await run_bot(config_file)
                
                # Sleep for 61 seconds to avoid duplicate runs
                await asyncio.sleep(61)
            else:
                # Sleep for 30 seconds before checking again
                await asyncio.sleep(30)
        
        except KeyboardInterrupt:
            logger.info("Scheduled execution stopped by user")
            break
        
        except Exception as e:
            logger.error(f"Error in scheduled execution: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Sleep for 5 minutes before retrying
            await asyncio.sleep(300)


async def run_backtest(instrument_key: str, days: int = 250, config_file: str = None, output_file: str = None):
    """
    Run backtest for a stock
    
    Args:
        instrument_key: Instrument key to backtest
        days: Number of days for backtest
        config_file: Path to configuration file
        output_file: Output file for results
    """
    # Load configuration
    config_manager = ConfigManager(config_file)
    config = config_manager.get_config()
    
    # Initialize bot
    bot = TradingBot(config)
    
    try:
        # Initialize components
        await bot.initialize()
        
        # Run backtest
        results = await bot.run_backtest(instrument_key, days)
        
        # Save to file if specified
        if output_file and not 'error' in results:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Backtest results saved to {output_file}")
        
        # Print summary to console
        print("\n" + "="*50)
        print(f"BACKTEST RESULTS: {results.get('stock_name', instrument_key)}")
        print("="*50)
        
        if 'error' in results:
            print(f"\nBacktest failed: {results['error']}")
        else:
            # Print key metrics
            print("\nPERFORMANCE METRICS:")
            for key in ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
                print(f"{key.replace('_', ' ').title()}: {results.get(key, 'N/A')}")
            
            # Print trade summary
            print(f"\nTotal Trades: {results.get('total_trades', 0)}")
            print(f"Winning Trades: {results.get('winning_trades', 0)}")
            print(f"Losing Trades: {results.get('losing_trades', 0)}")
            
            # Print Monte Carlo results if available
            if 'monte_carlo' in results:
                print("\nMONTE CARLO SIMULATION:")
                monte_carlo = results['monte_carlo']
                print(f"Return Range (95% CI): {monte_carlo.get('return_range', 'N/A')}")
                print(f"Drawdown Range (95% CI): {monte_carlo.get('drawdown_range', 'N/A')}")
                print(f"Failure Rate: {monte_carlo.get('failure_rate', 'N/A')}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Print error to console
        print(f"\nError: {str(e)}")
        
        return {'error': str(e)}


async def run_multi_backtest(days: int = 250, config_file: str = None, output_file: str = None):
    """
    Run backtest for multiple stocks
    
    Args:
        days: Number of days for backtest
        config_file: Path to configuration file
        output_file: Output file for results
    """
    # Load configuration
    config_manager = ConfigManager(config_file)
    config = config_manager.get_config()
    
    # Initialize bot
    bot = TradingBot(config)
    
    try:
        # Initialize components
        await bot.initialize()
        
        # Container for all results
        all_results = {}
        summary = []
        
        # Run backtest for each stock
        for instrument_key in config.stock_list:
            logger.info(f"Running backtest for {instrument_key}")
            
            try:
                # Run backtest
                result = await bot.run_backtest(instrument_key, days)
                all_results[instrument_key] = result
                
                # Add to summary if successful
                if 'error' not in result:
                    summary.append({
                        'instrument_key': instrument_key,
                        'stock_name': result.get('stock_name', 'Unknown'),
                        'stock_symbol': result.get('stock_symbol', instrument_key),
                        'total_return': result.get('total_return', 'N/A'),
                        'win_rate': result.get('win_rate', 'N/A'),
                        'sharpe_ratio': result.get('sharpe_ratio', 'N/A'),
                        'max_drawdown': result.get('max_drawdown', 'N/A'),
                        'total_trades': result.get('total_trades', 0)
                    })
            
            except Exception as e:
                logger.error(f"Error running backtest for {instrument_key}: {str(e)}")
                all_results[instrument_key] = {'error': str(e)}
        
        # Sort summary by total return
        summary.sort(
            key=lambda x: float(x['total_return'].replace('%', '')) 
            if isinstance(x['total_return'], str) else -9999, 
            reverse=True
        )
        
        # Save to file if specified
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump({
                    'summary': summary,
                    'details': all_results,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'days': days
                }, f, indent=2)
            logger.info(f"Multi-backtest results saved to {output_file}")
        
        # Print summary to console
        print("\n" + "="*70)
        print("MULTI-STOCK BACKTEST SUMMARY")
        print("="*70)
        print(f"{'SYMBOL':<10} {'RETURN':<10} {'WIN RATE':<10} {'SHARPE':<8} {'DRAWDOWN':<10} {'TRADES':<7}")
        print("-"*70)
        
        for result in summary:
            print(f"{result['stock_symbol']:<10} {result['total_return']:<10} "
                  f"{result['win_rate']:<10} {result['sharpe_ratio']:<8} "
                  f"{result['max_drawdown']:<10} {result['total_trades']:<7}")
        
        return {
            'summary': summary,
            'details': all_results
        }
    
    except Exception as e:
        logger.error(f"Error running multi-backtest: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Print error to console
        print(f"\nError: {str(e)}")
        
        return {'error': str(e)}


async def run_optimization(instrument_key: str, config_file: str = None, output_file: str = None):
    """
    Run parameter optimization
    
    Args:
        instrument_key: Instrument key to optimize
        config_file: Path to configuration file
        output_file: Output file for results
    """
    # Load configuration
    config_manager = ConfigManager(config_file)
    config = config_manager.get_config()
    
    # Get optimization settings
    param_grid = config.backtest.walk_forward_optimization.get('param_grid', {})
    in_sample_pct = config.backtest.walk_forward_optimization.get('in_sample_pct', 0.6)
    step_size = config.backtest.walk_forward_optimization.get('step_size', 0.2)
    
    if not param_grid:
        print("No parameter grid defined in configuration")
        return {'error': "No parameter grid defined"}
    
    try:
        # Initialize bot
        bot = TradingBot(config)
        await bot.initialize()
        
        # Get historical data
        df = await bot.get_historical_data(
            instrument_key=instrument_key,
            interval=config.analysis.intervals.get('short_term', '1D'),
            days=config.backtest.lookback_days
        )
        
        # Get stock info
        stock_info = await bot.get_stock_info(instrument_key)
        
        # Import backtest engine directly
        from quantm10.backtest.engine import BacktestEngine
        
        # Initialize backtest engine
        backtest = BacktestEngine(df, config, config.backtest.initial_capital)
        
        # Define strategy function (same as in TradingBot.run_backtest)
        def strategy(data: pd.DataFrame, **kwargs) -> int:
            """Trading strategy based on indicators and patterns"""
            # Use only data up to current index
            current_data = data.copy()
            
            # Calculate indicators
            indicators = TechnicalIndicators(current_data, config)
            indicators.calculate_all()
            indicator_signals = indicators.get_overall_signal()
            
            # Detect patterns
            candlestick_patterns = CandlestickPatterns(current_data, config)
            candlestick_patterns.detect_all_patterns()
            pattern_signals = candlestick_patterns.get_overall_signal()
            
            # Determine signal
            indicator_strength = indicator_signals.get('strength', 0)
            pattern_strength = pattern_signals.get('strength', 0)
            
            # Apply parameters from optimization
            indicator_weight = kwargs.get('indicator_weight', 0.7)
            pattern_weight = kwargs.get('pattern_weight', 0.3)
            signal_threshold = kwargs.get('signal_threshold', 2.0)
            
            # Weighted combination with parameterized weights
            combined_strength = (indicator_strength * indicator_weight) + (pattern_strength * pattern_weight)
            
            if combined_strength >= signal_threshold:
                return 1  # Buy
            elif combined_strength <= -signal_threshold:
                return -1  # Sell
            else:
                return 0  # Neutral
        
        # Run walk-forward optimization
        results = backtest.walk_forward_optimization(
            strategy, param_grid, in_sample_pct, step_size
        )
        
        # Add stock info
        results['stock_name'] = stock_info.get('name', '')
        results['stock_symbol'] = stock_info.get('symbol', '')
        results['instrument_key'] = instrument_key
        
        # Save to file if specified
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Optimization results saved to {output_file}")
        
        # Print results to console
        print("\n" + "="*50)
        print(f"OPTIMIZATION RESULTS: {results.get('stock_name', instrument_key)}")
        print("="*50)
        
        if 'error' in results:
            print(f"\nOptimization failed: {results['error']}")
        else:
            print("\nRECOMMENDED PARAMETERS:")
            for param, value in results['recommended_params'].items():
                print(f"{param}: {value}")
            
            print(f"\nAverage Out-of-Sample Return: {results['avg_out_sample_return']*100:.2f}%")
            print(f"Average Out-of-Sample Drawdown: {results['avg_out_sample_drawdown']*100:.2f}%")
            
            # Print period results
            print("\nPERIOD RESULTS:")
            for i, (period_start, period_end) in enumerate(results['periods']):
                result = results['all_results'][i]
                print(f"Period {i+1}: {period_start} to {period_end}")
                print(f"  Parameters: {result['best_params']}")
                print(f"  Out-of-Sample Return: {result['out_sample_return']*100:.2f}%")
        
        return results
    
    except Exception as e:
        logger.error(f"Error running optimization: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Print error to console
        print(f"\nError: {str(e)}")
        
        return {'error': str(e)}


async def main():
    """Main entry point for the application"""
    # Initialize database
    init_db()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging()
    
    logger.info(f"Starting QuantM10 with command: {args.command}")
    
    # Execute based on command
    if args.command == "run":
        await run_bot(args.config, args.stock)
    
    elif args.command == "scheduled":
        await run_scheduled(args.config)
    
    elif args.command == "backtest":
        await run_backtest(args.instrument_key, args.days, args.config, args.output)
    
    elif args.command == "multi":
        await run_multi_backtest(args.days, args.config, args.output)
    
    elif args.command == "optimize":
        await run_optimization(args.instrument_key, args.config, args.output)
    
    else:
        # Default to single run
        await run_bot(args.config)
    
    logger.info("QuantM10 execution completed")


if __name__ == "__main__":
    # If no arguments provided, default to "run"
    if len(sys.argv) == 1:
        sys.argv.append("run")
    
    # Run async main
    asyncio.run(main())