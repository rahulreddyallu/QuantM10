"""
Main entry point for running the Master Trading Signal Bot.
Handles initialization, scheduling, and execution.

Author: rahulreddyallu
Version: 4.0.0 (Master)
Date: 2025-05-03
"""

import asyncio
import logging
import sys
import os
import time
import json
import datetime
import traceback
import argparse
import upstox_client
from pathlib import Path
from upstox_client.models.ohlc import Ohlc as OHLCInterval
from upstox_client.api_client import ApiClient
from upstox_client.api.login_api import LoginApi
from upstox_client.api.market_quote_api import MarketQuoteApi
from upstox_client.api.history_api import HistoryApi
import importlib

# Import the core logic from compute.py
from compute import TechnicalAnalysis, CandlestickPatterns, TradingSignalBot, BacktestEngine
import config

# Set up logging
def setup_logging():
    """Setup logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for log filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'trading_bot_{timestamp}.log'
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger()
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

async def run_scheduled():
    """Run the trading signal bot on a schedule."""
    logger = setup_logging()
    
    try:
        # Initialize the bot
        logger.info("Initializing Trading Signal Bot...")
        bot = TradingSignalBot(config.__dict__)
        
        # Schedule runs at specific times
        market_open = config.SCHEDULE_INTERVALS.get('market_open', '09:15')
        mid_day = config.SCHEDULE_INTERVALS.get('mid_day', '12:30') 
        pre_close = config.SCHEDULE_INTERVALS.get('pre_close', '15:00')
        
        logger.info(f"Bot scheduled to run at: {market_open}, {mid_day}, {pre_close}")
        
        # Run immediately on startup if configured
        if hasattr(config, 'RUN_ON_STARTUP') and config.RUN_ON_STARTUP:
            logger.info("Running initial analysis on startup...")
            await run_with_backtest_option(bot)
            logger.info("Initial analysis completed.")
        
        # Simple loop to check current time against schedule
        while True:
            current_time = datetime.datetime.now().strftime('%H:%M')
            
            # Skip on weekends if configured
            if hasattr(config, 'RUN_ON_WEEKENDS') and not config.RUN_ON_WEEKENDS:
                weekday = datetime.datetime.now().weekday()
                if weekday >= 5:  # 5 = Saturday, 6 = Sunday
                    logger.info("Weekend detected - sleeping for 1 hour")
                    await asyncio.sleep(3600)  # Sleep for 1 hour
                    continue
            
            # Check if it's time to run
            if current_time in [market_open, mid_day, pre_close]:
                logger.info(f"Scheduled run at {current_time}")
                await run_with_backtest_option(bot)
                # Sleep for 61 seconds to avoid duplicate runs
                await asyncio.sleep(61)
            else:
                # Sleep for 30 seconds before checking again
                await asyncio.sleep(30)
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

async def run_with_backtest_option(bot):
    """Run the bot with optional backtest before signal generation."""
    if hasattr(config, 'RUN_BACKTEST_BEFORE_SIGNAL') and config.RUN_BACKTEST_BEFORE_SIGNAL:
        logger.info("Running backtest before generating signals...")
        for instrument_key in config.STOCK_LIST:
            try:
                # Run backtest for the instrument
                backtest_result = await bot.run_backtest(instrument_key)
                
                # Log backtest summary
                if 'error' not in backtest_result:
                    logger.info(f"Backtest for {instrument_key} complete: "
                                f"Win Rate: {backtest_result['metrics'].get('Win Rate', 'N/A')}, "
                                f"Total Return: {backtest_result['metrics'].get('Total Return', 'N/A')}")
                else:
                    logger.warning(f"Backtest for {instrument_key} failed: {backtest_result['error']}")
            except Exception as e:
                logger.error(f"Error running backtest for {instrument_key}: {str(e)}")
    
    # Run the main signal generation
    return await bot.run()

async def run_adhoc():
    """Run the trading signal bot once and exit."""
    logger = setup_logging()
    
    try:
        # Initialize the bot
        logger.info("Initializing Trading Signal Bot...")
        bot = TradingSignalBot(config.__dict__)
        
        # Run once
        logger.info("Running adhoc analysis...")
        await run_with_backtest_option(bot)
        logger.info("Adhoc analysis completed.")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

async def run_backtest_command(instrument_key, days=250, output_file=None):
    """Run a backtest for a specific instrument."""
    logger = setup_logging()
    
    try:
        # Initialize the bot
        logger.info(f"Initializing Trading Signal Bot for backtesting {instrument_key}...")
        bot = TradingSignalBot(config.__dict__)
        
        # Run backtest
        logger.info(f"Running backtest for {instrument_key} with {days} days of history...")
        backtest_results = await bot.run_backtest(instrument_key, days)
        
        # Print results to console in a readable format
        if 'error' not in backtest_results:
            print("\n" + "="*50)
            print(f"BACKTEST RESULTS: {backtest_results.get('stock_name', instrument_key)}")
            print("="*50)
            
            # Print metrics
            print("\nPERFORMANCE METRICS:")
            for key, value in backtest_results.get('metrics', {}).items():
                print(f"{key}: {value}")
            
            # Print Monte Carlo results if available
            if 'monte_carlo' in backtest_results:
                print("\nMONTE CARLO SIMULATION:")
                for key, value in backtest_results['monte_carlo'].items():
                    print(f"{key}: {value}")
            
            # Print trade summary
            trade_count = len(backtest_results.get('trades', []))
            print(f"\nTRADES: {trade_count} total trades")
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(backtest_results, f, indent=2)
                print(f"\nFull results saved to {output_file}")
        else:
            print(f"\nBacktest failed: {backtest_results['error']}")
        
        logger.info("Backtest completed.")
    
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

async def run_walk_forward_optimization(instrument_key, output_file=None):
    """Run walk-forward optimization for a specific instrument."""
    logger = setup_logging()
    
    try:
        # Initialize the bot
        logger.info(f"Initializing Trading Signal Bot for walk-forward optimization on {instrument_key}...")
        bot = TradingSignalBot(config.__dict__)
        
        # Fetch historical data
        days = config.BACKTEST.get("lookback_days", 500)
        df = await bot.fetch_historical_data(instrument_key, days=days)
        
        # Initialize backtest engine
        backtest = BacktestEngine(df, bot.params)
        
        # Get parameter grid from config
        param_grid = config.BACKTEST.get("walk_forward_optimization", {}).get("param_grid", {})
        in_sample_pct = config.BACKTEST.get("walk_forward_optimization", {}).get("in_sample_pct", 0.6)
        step_size = config.BACKTEST.get("walk_forward_optimization", {}).get("step_size", 0.2)
        
        # Define test strategy
        def strategy_func(data, **kwargs):
            # Initialize indicators with provided parameters
            indicators = TechnicalAnalysis(data, bot.params)
            indicators.calculate_all()
            
            # Get overall signal
            signal = indicators.get_overall_signal()
            
            # Convert to numeric
            if signal['signal'] in ['BUY', 'STRONG BUY']:
                return 1
            elif signal['signal'] in ['SELL', 'STRONG SELL']:
                return -1
            else:
                return 0
        
        # Run walk-forward optimization
        logger.info(f"Running walk-forward optimization...")
        wfo_results = backtest.walk_forward_optimization(
            strategy_func, param_grid, in_sample_pct, step_size
        )
        
        # Print results
        print("\n" + "="*50)
        print(f"WALK-FORWARD OPTIMIZATION RESULTS: {instrument_key}")
        print("="*50)
        
        if 'error' not in wfo_results:
            print("\nRECOMMENDED PARAMETERS:")
            for param, value in wfo_results['recommended_params'].items():
                print(f"{param}: {value}")
            
            print(f"\nAverage Out-of-Sample Return: {wfo_results['avg_out_sample_return']*100:.2f}%")
            print(f"Average Out-of-Sample Drawdown: {wfo_results['avg_out_sample_drawdown']*100:.2f}%")
            
            # Print period results
            print("\nPERIOD RESULTS:")
            for i, period in enumerate(wfo_results['periods']):
                result = wfo_results['all_results'][i]
                print(f"Period {i+1}: {period[0]} to {period[1]}")
                print(f"  Parameters: {result['best_params']}")
                print(f"  Out-of-Sample Return: {result['out_sample_return']*100:.2f}%")
                print(f"  Out-of-Sample Metric: {result['out_sample_metric']:.4f}")
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(wfo_results, f, indent=2)
                print(f"\nFull results saved to {output_file}")
        else:
            print(f"\nOptimization failed: {wfo_results['error']}")
        
        logger.info("Walk-forward optimization completed.")
    
    except Exception as e:
        logger.error(f"Error in walk-forward optimization: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

async def run_multi_stock_backtest(stock_list=None, days=250, output_file=None):
    """Run backtests for multiple stocks and generate a summary report."""
    logger = setup_logging()
    
    try:
        # Initialize the bot
        logger.info("Initializing Trading Signal Bot for multi-stock backtest...")
        bot = TradingSignalBot(config.__dict__)
        
        # Use provided stock list or default from config
        stock_list = stock_list or config.STOCK_LIST
        
        # Prepare container for results
        all_results = {}
        summary = []
        
        # Run backtest for each stock
        for instrument_key in stock_list:
            logger.info(f"Running backtest for {instrument_key}...")
            try:
                backtest_results = await bot.run_backtest(instrument_key, days)
                all_results[instrument_key] = backtest_results
                
                # Add to summary if successful
                if 'error' not in backtest_results:
                    summary.append({
                        'instrument_key': instrument_key,
                        'stock_name': backtest_results.get('stock_name', 'Unknown'),
                        'stock_symbol': backtest_results.get('stock_symbol', instrument_key),
                        'total_return': backtest_results['metrics'].get('Total Return', 'N/A'),
                        'win_rate': backtest_results['metrics'].get('Win Rate', 'N/A'),
                        'sharpe_ratio': backtest_results['metrics'].get('Sharpe Ratio', 'N/A'),
                        'max_drawdown': backtest_results['metrics'].get('Max Drawdown', 'N/A'),
                        'total_trades': backtest_results['metrics'].get('Total Trades', 0),
                    })
                else:
                    logger.warning(f"Backtest failed for {instrument_key}: {backtest_results['error']}")
            except Exception as e:
                logger.error(f"Error in backtest for {instrument_key}: {str(e)}")
                all_results[instrument_key] = {'error': str(e)}
        
        # Sort summary by total return
        summary.sort(key=lambda x: float(x['total_return'].replace('%', '')) if isinstance(x['total_return'], str) else -9999, reverse=True)
        
        # Print summary
        print("\n" + "="*70)
        print("MULTI-STOCK BACKTEST SUMMARY")
        print("="*70)
        print(f"{'SYMBOL':<10} {'RETURN':<10} {'WIN RATE':<10} {'SHARPE':<8} {'DRAWDOWN':<10} {'TRADES':<7}")
        print("-"*70)
        
        for result in summary:
            print(f"{result['stock_symbol']:<10} {result['total_return']:<10} {result['win_rate']:<10} "
                  f"{result['sharpe_ratio']:<8} {result['max_drawdown']:<10} {result['total_trades']:<7}")
        
        # Save to file if requested
        if output_file:
            report = {
                'summary': summary,
                'details': all_results,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'days': days
            }
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nFull results saved to {output_file}")
        
        logger.info("Multi-stock backtest completed.")
    
    except Exception as e:
        logger.error(f"Error in multi-stock backtest: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Master Trading Signal Bot')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Scheduled mode
    scheduled_parser = subparsers.add_parser('scheduled', help='Run bot on a schedule')
    
    # Adhoc mode
    adhoc_parser = subparsers.add_parser('adhoc', help='Run bot once and exit')
    
    # Backtest mode
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest for a specific instrument')
    backtest_parser.add_argument('instrument_key', help='Instrument key to backtest')
    backtest_parser.add_argument('--days', type=int, default=250, help='Number of days for backtest')
    backtest_parser.add_argument('--output', help='Output file for detailed results (JSON)')
    
    # Walk-forward optimization mode
    wfo_parser = subparsers.add_parser('optimize', help='Run walk-forward optimization')
    wfo_parser.add_argument('instrument_key', help='Instrument key to optimize')
    wfo_parser.add_argument('--output', help='Output file for detailed results (JSON)')
    
    # Multi-stock backtest mode
    multi_parser = subparsers.add_parser('multi', help='Run backtest for multiple stocks')
    multi_parser.add_argument('--days', type=int, default=250, help='Number of days for backtest')
    multi_parser.add_argument('--output', help='Output file for detailed results (JSON)')
    
    return parser.parse_args()

async def main_async():
    """Async main entry point."""
    args = parse_arguments()
    
    if args.command == 'backtest':
        return await run_backtest_command(args.instrument_key, args.days, args.output)
    elif args.command == 'optimize':
        return await run_walk_forward_optimization(args.instrument_key, args.output)
    elif args.command == 'multi':
        return await run_multi_stock_backtest(None, args.days, args.output)
    elif args.command == 'adhoc':
        return await run_adhoc()
    else:  # Default to scheduled mode
        return await run_scheduled()

def main():
    """Main entry point."""
    # Handle case with no arguments (default to scheduled mode)
    if len(sys.argv) == 1:
        sys.argv.append('scheduled')
        
    return asyncio.run(main_async())

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
