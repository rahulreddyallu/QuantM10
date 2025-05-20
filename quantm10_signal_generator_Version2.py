"""
Signal generation module for QuantM10

Combines technical indicators and pattern signals to generate 
trading signals with risk management parameters.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union

from quantm10.exceptions import SignalGenerationError
from quantm10.utils.logging import get_logger

logger = get_logger(__name__)


class SignalGenerator:
    """Generate trading signals from indicators and patterns"""
    
    def __init__(self, df: pd.DataFrame, params: Optional[Dict[str, Any]] = None):
        """
        Initialize with price data and parameters
        
        Args:
            df: DataFrame with OHLCV data
            params: Signal parameters
        """
        self.df = df.copy()
        self.params = params or {}
        
        # Get latest prices
        self.current_price = df['close'].iloc[-1] if not df.empty else None
        self.current_high = df['high'].iloc[-1] if not df.empty else None
        self.current_low = df['low'].iloc[-1] if not df.empty else None
        
        # Get ATR for stop loss and target calculation
        self.atr = self._calculate_atr()
    
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
            return self.params.get('signals', {}).get(param_name, default)
        except (AttributeError, KeyError):
            return default
    
    def _calculate_atr(self, period: int = 14) -> float:
        """
        Calculate Average True Range
        
        Args:
            period: ATR period
            
        Returns:
            ATR value
        """
        try:
            # Need at least period+1 candles
            if len(self.df) < period + 1:
                return 0
            
            # Calculate true range
            high_low = self.df['high'] - self.df['low']
            high_close = abs(self.df['high'] - self.df['close'].shift())
            low_close = abs(self.df['low'] - self.df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            # Calculate ATR
            atr = true_range.rolling(window=period).mean().iloc[-1]
            return atr
        
        except Exception as e:
            logger.warning(f"Error calculating ATR: {str(e)}")
            return 0
    
    def generate_signals(self, 
                       indicators: Dict[str, Any], 
                       candlestick_patterns: Dict[str, Any],
                       chart_patterns: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate combined trading signals
        
        Args:
            indicators: Indicator signals
            candlestick_patterns: Candlestick pattern signals
            chart_patterns: Chart pattern signals
            
        Returns:
            Dictionary with combined signals and trade parameters
            
        Raises:
            SignalGenerationError: If signal generation fails
        """
        try:
            # Get parameters
            min_signal_strength = self._get_param('min_signal_strength', 3)
            target_multiplier = self._get_param('target_multiplier', 1.5)
            stop_multiplier = self._get_param('stop_multiplier', 1.0)
            min_rrr = self._get_param('min_rrr', 1.5)
            
            # Get overall signals from each component
            indicator_signal = indicators.get('overall', {})
            candlestick_signal = candlestick_patterns.get('overall', {})
            chart_signal = chart_patterns.get('overall', {}) if chart_patterns else {}
            
            # Extract signal strengths
            indicator_strength = indicator_signal.get('strength', 0)
            candlestick_strength = candlestick_signal.get('strength', 0)
            chart_strength = chart_signal.get('strength', 0)
            
            # Get signal types
            indicator_type = indicator_signal.get('signal', 'NEUTRAL')
            candlestick_type = candlestick_signal.get('signal', 'NEUTRAL')
            chart_type = chart_signal.get('signal', 'NEUTRAL')
            
            # Default weightings for each signal source
            indicator_weight = 0.5
            candlestick_weight = 0.3
            chart_weight = 0.2
            
            # Calculate weighted strength
            weighted_strength = (
                indicator_strength * indicator_weight +
                candlestick_strength * candlestick_weight +
                chart_strength * chart_weight
            )
            
            # Get integer strength for easier comparison
            total_strength = int(round(weighted_strength))
            
            # Determine signal type
            if total_strength >= 2:
                signal_type = "BUY"
            elif total_strength <= -2:
                signal_type = "SELL"
            else:
                signal_type = "NEUTRAL"
            
            # Determine signal strength descriptor
            if total_strength >= 3:
                signal_text = "STRONG BUY"
            elif total_strength >= 2:
                signal_text = "BUY"
            elif total_strength >= 1:
                signal_text = "WEAK BUY"
            elif total_strength <= -3:
                signal_text = "STRONG SELL"
            elif total_strength <= -2:
                signal_text = "SELL"
            elif total_strength <= -1:
                signal_text = "WEAK SELL"
            else:
                signal_text = "NEUTRAL"
            
            # Calculate risk parameters only for actionable signals
            entry_price = stop_loss = target_price = None
            risk_reward_ratio = 0
            
            if signal_type in ["BUY", "SELL"] and self.atr > 0:
                # Set entry price
                entry_price = self.current_price
                
                if signal_type == "BUY":
                    # For buy signals
                    stop_loss = entry_price - (self.atr * stop_multiplier)
                    target_price = entry_price + (self.atr * target_multiplier)
                else:
                    # For sell signals
                    stop_loss = entry_price + (self.atr * stop_multiplier)
                    target_price = entry_price - (self.atr * target_multiplier)
                
                # Calculate reward-to-risk ratio
                if entry_price and stop_loss and target_price:
                    if signal_type == "BUY":
                        risk = entry_price - stop_loss
                        reward = target_price - entry_price
                    else:
                        risk = stop_loss - entry_price
                        reward = entry_price - target_price
                    
                    risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Prepare signal data
            signal_data = {
                'signal_type': signal_type,
                'signal_text': signal_text,
                'signal_strength': total_strength,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'risk_reward_ratio': risk_reward_ratio,
                'atr': self.atr,
                'components': {
                    'indicators': {
                        'signal': indicator_type,
                        'strength': indicator_strength
                    },
                    'candlestick_patterns': {
                        'signal': candlestick_type,
                        'strength': candlestick_strength,
                        'patterns': candlestick_signal.get('patterns', [])
                    },
                    'chart_patterns': {
                        'signal': chart_type,
                        'strength': chart_strength,
                        'patterns': chart_signal.get('patterns', [])
                    }
                },
                'confidence': self._calculate_confidence(
                    indicator_signal.get('confidence', 0),
                    candlestick_signal.get('confidence', 0),
                    chart_signal.get('confidence', 0)
                )
            }
            
            # Check if signal meets minimum requirements
            signal_data['actionable'] = (
                abs(total_strength) >= min_signal_strength and
                (signal_type == "NEUTRAL" or risk_reward_ratio >= min_rrr)
            )
            
            return signal_data
        
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise SignalGenerationError(str(e))
    
    def _calculate_confidence(self, 
                            indicator_confidence: float, 
                            candlestick_confidence: float,
                            chart_confidence: float) -> float:
        """
        Calculate overall confidence score
        
        Args:
            indicator_confidence: Confidence score for indicators
            candlestick_confidence: Confidence score for candlestick patterns
            chart_confidence: Confidence score for chart patterns
            
        Returns:
            Overall confidence score (0-100)
        """
        # Default weightings
        indicator_weight = 0.5
        candlestick_weight = 0.3
        chart_weight = 0.2
        
        # Calculate weighted confidence
        weighted_confidence = (
            indicator_confidence * indicator_weight +
            candlestick_confidence * candlestick_weight +
            chart_confidence * chart_weight
        )
        
        return weighted_confidence
    
    def create_trade_checklist(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create pre-trade checklist from signal data
        
        Args:
            signal_data: Signal data dictionary
            
        Returns:
            Dictionary with trade checklist items
        """
        signal_type = signal_data.get('signal_type', 'NEUTRAL')
        
        if signal_type == 'NEUTRAL':
            return {'items': []}
        
        # Common checklist items
        checklist = [
            f"Signal strength: {abs(signal_data.get('signal_strength', 0))}/5",
            f"Confidence: {signal_data.get('confidence', 0):.1f}%",
            f"Risk-reward ratio: {signal_data.get('risk_reward_ratio', 0):.2f}"
        ]
        
        # Add indicator confirmations
        indicators = signal_data.get('components', {}).get('indicators', {})
        indicator_signal = indicators.get('signal', 'NEUTRAL')
        
        if signal_type == 'BUY' and 'BUY' in indicator_signal:
            checklist.append("Technical indicators confirm bullish trend")
        elif signal_type == 'SELL' and 'SELL' in indicator_signal:
            checklist.append("Technical indicators confirm bearish trend")
        
        # Add pattern confirmations
        candlestick_patterns = signal_data.get('components', {}).get('candlestick_patterns', {}).get('patterns', [])
        chart_patterns = signal_data.get('components', {}).get('chart_patterns', {}).get('patterns', [])
        
        if candlestick_patterns:
            pattern_text = ', '.join(candlestick_patterns)
            checklist.append(f"Candlestick patterns: {pattern_text}")
        
        if chart_patterns:
            pattern_text = ', '.join(chart_patterns)
            checklist.append(f"Chart patterns: {pattern_text}")
        
        # Add risk management items
        if signal_data.get('entry_price') and signal_data.get('stop_loss'):
            stop_pct = abs(signal_data['stop_loss'] - signal_data['entry_price']) / signal_data['entry_price'] * 100
            checklist.append(f"Stop-loss distance: {stop_pct:.2f}%")
            
            if signal_data.get('target_price'):
                target_pct = abs(signal_data['target_price'] - signal_data['entry_price']) / signal_data['entry_price'] * 100
                checklist.append(f"Target distance: {target_pct:.2f}%")
        
        return {'items': checklist}