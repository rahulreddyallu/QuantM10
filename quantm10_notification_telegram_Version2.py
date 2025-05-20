"""
Telegram notification service for QuantM10

Provides functionality for sending signal alerts and reports via Telegram.
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
import telegram
from telegram import ParseMode

from quantm10.exceptions import NotificationError
from quantm10.utils.logging import get_logger

logger = get_logger(__name__)


class TelegramNotifier:
    """Telegram notification service for sending alerts"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID to send messages to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None
    
    async def initialize(self) -> None:
        """
        Initialize Telegram bot
        
        Raises:
            NotificationError: If initialization fails
        """
        try:
            self.bot = telegram.Bot(token=self.bot_token)
            await self.bot.get_me()
            logger.info("Telegram bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {str(e)}")
            raise NotificationError("Telegram", str(e))
    
    async def send_message(self, message: str, parse_mode: str = ParseMode.MARKDOWN) -> bool:
        """
        Send a message via Telegram
        
        Args:
            message: Message to send
            parse_mode: Message parsing mode
            
        Returns:
            True if message was sent successfully
            
        Raises:
            NotificationError: If message sending fails
        """
        if not self.bot:
            await self.initialize()
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            logger.info("Message sent successfully to Telegram")
            return True
        except Exception as e:
            logger.error(f"Failed to send message to Telegram: {str(e)}")
            raise NotificationError("Telegram", str(e))
    
    def _escape_markdown(self, text: str) -> str:
        """
        Escape special markdown characters
        
        Args:
            text: Text to escape
            
        Returns:
            Escaped text
        """
        escape_chars = r'_*[]()~`>#+-=|{}.!'
        return ''.join(f'\\{c}' if c in escape_chars else c for c in text)
    
    async def send_signal_alert(self, signal_data: Dict[str, Any], 
                               stock_info: Dict[str, Any],
                               template: Optional[str] = None) -> bool:
        """
        Send trading signal alert
        
        Args:
            signal_data: Signal data
            stock_info: Stock information
            template: Optional template for message formatting
            
        Returns:
            True if alert was sent successfully
            
        Raises:
            NotificationError: If alert sending fails
        """
        try:
            # Extract signal info
            signal_type = signal_data.get('signal_text', 'NEUTRAL')
            signal_strength = abs(signal_data.get('signal_strength', 0))
            
            # Skip sending alerts for neutral signals
            if signal_type == 'NEUTRAL':
                logger.info("Not sending alert for NEUTRAL signal")
                return False
            
            # Format signal strength as stars
            signal_strength_stars = '⭐' * signal_strength
            
            # Format stop loss and target percentages
            entry_price = signal_data.get('entry_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            target_price = signal_data.get('target_price', 0)
            
            stop_loss_pct = ((stop_loss - entry_price) / entry_price * 100) if entry_price > 0 else 0
            target_pct = ((target_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            
            # Format RRR
            risk_reward_ratio = signal_data.get('risk_reward_ratio', 0)
            
            # Get component signals
            indicators = signal_data.get('components', {}).get('indicators', {})
            candlestick_patterns = signal_data.get('components', {}).get('candlestick_patterns', {})
            chart_patterns = signal_data.get('components', {}).get('chart_patterns', {})
            
            # Format primary indicators
            indicator_signal = indicators.get('signal', 'NEUTRAL')
            primary_indicators = [
                f"Moving Averages: {indicator_signal}",
                f"RSI: {indicators.get('rsi', 'NEUTRAL')}",
                f"MACD: {indicators.get('macd', 'NEUTRAL')}"
            ]
            
            # Format patterns
            pattern_list = []
            candlestick_pattern_names = candlestick_patterns.get('patterns', [])
            chart_pattern_names = chart_patterns.get('patterns', [])
            
            for pattern in candlestick_pattern_names:
                pattern_list.append(f"• {pattern}")
            
            for pattern in chart_pattern_names:
                pattern_list.append(f"• {pattern}")
            
            # Generate message using template if provided
            if template:
                # Replace placeholders in template
                message = template
                
                # Stock info
                message = message.replace('{stock_name}', self._escape_markdown(stock_info.get('name', '')))
                message = message.replace('{stock_symbol}', self._escape_markdown(stock_info.get('symbol', '')))
                message = message.replace('{current_price}', f"{entry_price:.2f}")
                message = message.replace('{industry}', self._escape_markdown(stock_info.get('industry', 'Unknown')))
                
                # Signal info
                message = message.replace('{signal_type}', signal_type)
                message = message.replace('{signal_strength_stars}', signal_strength_stars)
                
                # Price levels
                message = message.replace('{entry_price}', f"{entry_price:.2f}")
                message = message.replace('{stop_loss}', f"{stop_loss:.2f}")
                message = message.replace('{stop_loss_pct}', f"{abs(stop_loss_pct):.2f}")
                message = message.replace('{target_price}', f"{target_price:.2f}")
                message = message.replace('{target_pct}', f"{abs(target_pct):.2f}")
                message = message.replace('{risk_reward_ratio}', f"{risk_reward_ratio:.2f}")
                
                # Components
                message = message.replace('{primary_indicators}', '\n'.join(primary_indicators))
                message = message.replace('{patterns}', '\n'.join(pattern_list) if pattern_list else 'No patterns detected')
                
                # Timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                timestamp_short = datetime.now().strftime('%H:%M')
                message = message.replace('{timestamp}', timestamp)
                message = message.replace('{timestamp_short}', timestamp_short)
                
                # Trend analysis
                trend_strength = "Strong Uptrend" if signal_type == "STRONG BUY" else \
                                 "Uptrend" if signal_type == "BUY" else \
                                 "Strong Downtrend" if signal_type == "STRONG SELL" else \
                                 "Downtrend" if signal_type == "SELL" else \
                                 "Sideways/Neutral"
                message = message.replace('{trend_strength}', trend_strength)
                
                # Buy/Sell summary
                if "BUY" in signal_type:
                    buy_sell_summary = "👉 Consider buying at current market price with stop loss set at specified level."
                elif "SELL" in signal_type:
                    buy_sell_summary = "👉 Consider selling at current market price with stop loss set at specified level."
                else:
                    buy_sell_summary = "👉 Current market conditions unclear. Wait for stronger signal."
                
                message = message.replace('{buy_sell_summary}', buy_sell_summary)
                
            else:
                # Generate default message format
                message = f"🔍 *TRADING SIGNAL ALERT* 🔍\n\n"
                message += f"*{self._escape_markdown(stock_info.get('name', ''))}* ({self._escape_markdown(stock_info.get('symbol', ''))})\n"
                message += f"💰 *Current Price:* {entry_price:.2f}\n\n"
                message += f"🚨 *{signal_type} SIGNAL* 🚨\n"
                message += f"Signal Strength: {signal_strength_stars}\n\n"
                
                message += f"🎯 *TRADE SETUP:*\n"
                message += f"Entry: {entry_price:.2f}\n"
                message += f"Stop Loss: {stop_loss:.2f} ({abs(stop_loss_pct):.2f}%)\n"
                message += f"Target: {target_price:.2f} ({abs(target_pct):.2f}%)\n"
                message += f"R:R Ratio: {risk_reward_ratio:.2f}\n\n"
                
                if pattern_list:
                    message += f"📈 *PATTERNS:*\n"
                    message += '\n'.join(pattern_list) + "\n\n"
                
                # Timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                message += f"⏰ *Generated:* {timestamp}"
            
            # Send message
            return await self.send_message(message)
        
        except Exception as e:
            logger.error(f"Failed to send signal alert: {str(e)}")
            raise NotificationError("Telegram", f"Failed to send signal alert: {str(e)}")
    
    async def send_backtest_report(self, backtest_results: Dict[str, Any], 
                                 stock_info: Dict[str, Any],
                                 template: Optional[str] = None) -> bool:
        """
        Send backtest report
        
        Args:
            backtest_results: Backtest results
            stock_info: Stock information
            template: Optional template for message formatting
            
        Returns:
            True if report was sent successfully
            
        Raises:
            NotificationError: If report sending fails
        """
        try:
            # Extract metrics
            total_return = backtest_results.get('total_return', '0%')
            annualized_return = backtest_results.get('annualized_return', '0%')
            sharpe_ratio = backtest_results.get('sharpe_ratio', '0')
            max_drawdown = backtest_results.get('max_drawdown', '0%')
            win_rate = backtest_results.get('win_rate', '0%')
            total_trades = backtest_results.get('total_trades', 0)
            profit_factor = backtest_results.get('profit_factor', '0')
            avg_profit = backtest_results.get('avg_profit', '0')
            avg_loss = backtest_results.get('avg_loss', '0')
            
            # Monte Carlo data if available
            monte_carlo = backtest_results.get('monte_carlo', {})
            return_range = monte_carlo.get('return_range', 'N/A')
            drawdown_range = monte_carlo.get('drawdown_range', 'N/A')
            
            # Format dates
            start_date = backtest_results.get('start_date', 'N/A')
            end_date = backtest_results.get('end_date', 'N/A')
            
            # Generate message using template if provided
            if template:
                # Replace placeholders in template
                message = template
                
                # Stock info
                message = message.replace('{stock_name}', self._escape_markdown(stock_info.get('name', '')))
                message = message.replace('{stock_symbol}', self._escape_markdown(stock_info.get('symbol', '')))
                
                # Performance metrics
                message = message.replace('{total_return}', total_return)
                message = message.replace('{annualized_return}', annualized_return)
                message = message.replace('{sharpe_ratio}', sharpe_ratio)
                message = message.replace('{max_drawdown}', max_drawdown)
                
                # Trading statistics
                message = message.replace('{total_trades}', str(total_trades))
                message = message.replace('{win_rate}', win_rate)
                message = message.replace('{profit_factor}', profit_factor)
                message = message.replace('{avg_profit}', avg_profit)
                message = message.replace('{avg_loss}', avg_loss)
                
                # Monte Carlo
                message = message.replace('{return_range}', return_range)
                message = message.replace('{drawdown_range}', drawdown_range)
                
                # Dates
                message = message.replace('{start_date}', start_date)
                message = message.replace('{end_date}', end_date)
                
            else:
                # Generate default message format
                message = f"📊 *BACKTEST RESULTS* 📊\n\n"
                message += f"*{self._escape_markdown(stock_info.get('name', ''))}* ({self._escape_markdown(stock_info.get('symbol', ''))})\n\n"
                
                message += f"💰 *Performance:*\n"
                message += f"Total Return: {total_return}\n"
                message += f"Annualized Return: {annualized_return}\n"
                message += f"Sharpe Ratio: {sharpe_ratio}\n"
                message += f"Max Drawdown: {max_drawdown}\n\n"
                
                message += f"📈 *Trading Statistics:*\n"
                message += f"Total Trades: {total_trades}\n"
                message += f"Win Rate: {win_rate}\n"
                message += f"Profit Factor: {profit_factor}\n"
                message += f"Avg Profit: {avg_profit}\n"
                message += f"Avg Loss: {avg_loss}\n\n"
                
                if monte_carlo:
                    message += f"🎯 *Monte Carlo Analysis:*\n"
                    message += f"Return Range (95% CI): {return_range}\n"
                    message += f"Drawdown Range (95% CI): {drawdown_range}\n\n"
                
                message += f"⏰ *Period:* {start_date} to {end_date}"
            
            # Send message
            return await self.send_message(message)
        
        except Exception as e:
            logger.error(f"Failed to send backtest report: {str(e)}")
            raise NotificationError("Telegram", f"Failed to send backtest report: {str(e)}")
    
    async def send_daily_report(self, results: List[Dict[str, Any]]) -> bool:
        """
        Send daily summary report of all signals
        
        Args:
            results: List of signal results
            
        Returns:
            True if report was sent successfully
            
        Raises:
            NotificationError: If report sending fails
        """
        try:
            # Skip if no results
            if not results:
                logger.info("No results to send in daily report")
                return False
            
            # Count signals by type
            buy_signals = [r for r in results if "BUY" in r.get('signal_text', '')]
            sell_signals = [r for r in results if "SELL" in r.get('signal_text', '')]
            neutral_signals = [r for r in results if "NEUTRAL" in r.get('signal_text', '')]
            
            # Generate message
            from datetime import datetime
            date_str = datetime.now().strftime('%Y-%m-%d')
            
            message = f"📅 *DAILY SIGNAL REPORT - {date_str}* 📅\n\n"
            
            message += f"*SUMMARY:*\n"
            message += f"Total Stocks Analyzed: {len(results)}\n"
            message += f"Buy Signals: {len(buy_signals)}\n"
            message += f"Sell Signals: {len(sell_signals)}\n"
            message += f"Neutral: {len(neutral_signals)}\n\n"
            
            # Add strong buy signals
            strong_buys = [r for r in results if r.get('signal_text', '') == 'STRONG BUY']
            if strong_buys:
                message += f"🟢 *STRONG BUY SIGNALS:*\n"
                for signal in strong_buys:
                    stock_info = signal.get('stock_info', {})
                    stock_name = stock_info.get('name', '')
                    stock_symbol = stock_info.get('symbol', '')
                    
                    message += f"• {self._escape_markdown(stock_name)} ({self._escape_markdown(stock_symbol)})\n"
                message += "\n"
            
            # Add strong sell signals
            strong_sells = [r for r in results if r.get('signal_text', '') == 'STRONG SELL']
            if strong_sells:
                message += f"🔴 *STRONG SELL SIGNALS:*\n"
                for signal in strong_sells:
                    stock_info = signal.get('stock_info', {})
                    stock_name = stock_info.get('name', '')
                    stock_symbol = stock_info.get('symbol', '')
                    
                    message += f"• {self._escape_markdown(stock_name)} ({self._escape_markdown(stock_symbol)})\n"
                message += "\n"
            
            # Add normal buy signals
            normal_buys = [r for r in results if r.get('signal_text', '') == 'BUY']
            if normal_buys:
                message += f"🟡 *BUY SIGNALS:*\n"
                for signal in normal_buys:
                    stock_info = signal.get('stock_info', {})
                    stock_name = stock_info.get('name', '')
                    stock_symbol = stock_info.get('symbol', '')
                    
                    message += f"• {self._escape_markdown(stock_name)} ({self._escape_markdown(stock_symbol)})\n"
            
            # Send message
            return await self.send_message(message)
        
        except Exception as e:
            logger.error(f"Failed to send daily report: {str(e)}")
            raise NotificationError("Telegram", f"Failed to send daily report: {str(e)}")