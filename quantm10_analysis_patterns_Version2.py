"""
Chart pattern detection module for QuantM10

Implements candlestick pattern and chart pattern detection.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

from quantm10.exceptions import PatternDetectionError
from quantm10.utils.logging import get_logger

logger = get_logger(__name__)


class CandlestickPatterns:
    """Candlestick pattern detection and interpretation"""
    
    def __init__(self, df: pd.DataFrame, params: Optional[Dict[str, Any]] = None):
        """
        Initialize with price data and parameters
        
        Args:
            df: DataFrame with OHLCV data
            params: Parameters for pattern detection
        """
        self.df = df.copy()
        self.params = params or {}
        
        # Container for detected patterns
        self.patterns = {}
        
        # Calculate candle properties
        self._calculate_candle_dimensions()
        self._calculate_trend_context()
    
    def _get_param(self, pattern_name: str, param_key: str, default: Any = None) -> Any:
        """
        Get parameter value with fallback to default
        
        Args:
            pattern_name: Pattern name (e.g., 'marubozu')
            param_key: Parameter key (e.g., 'body_threshold')
            default: Default value if parameter not found
            
        Returns:
            Parameter value
        """
        try:
            return self.params.get('patterns', {}).get('candlestick_patterns', {}).get(pattern_name, {}).get(param_key, default)
        except (AttributeError, KeyError):
            return default
    
    def _is_pattern_enabled(self, pattern_name: str) -> bool:
        """
        Check if pattern detection is enabled
        
        Args:
            pattern_name: Pattern name
            
        Returns:
            True if pattern detection is enabled
        """
        return self._get_param(pattern_name, 'enabled', True)
    
    def _calculate_candle_dimensions(self) -> None:
        """Calculate basic candle properties"""
        # Calculate body size
        self.df['body_size'] = abs(self.df['close'] - self.df['open'])
        
        # Calculate upper and lower shadows
        self.df['upper_shadow'] = self.df.apply(
            lambda x: x['high'] - max(x['open'], x['close']), axis=1
        )
        self.df['lower_shadow'] = self.df.apply(
            lambda x: min(x['open'], x['close']) - x['low'], axis=1
        )
        
        # Calculate total range
        self.df['range'] = self.df['high'] - self.df['low']
        
        # Calculate body percentage of range
        self.df['body_pct'] = self.df['body_size'] / self.df['range']
        
        # Calculate color (1 for bullish, -1 for bearish)
        self.df['candle_color'] = np.where(self.df['close'] >= self.df['open'], 1, -1)
        
        # Calculate relative size compared to recent candles
        self.df['rel_size'] = self.df['range'] / self.df['range'].rolling(10).mean()
    
    def _calculate_trend_context(self) -> None:
        """Calculate trend context for better pattern interpretation"""
        # Simple trend detection using moving averages
        self.df['sma10'] = self.df['close'].rolling(window=10).mean()
        self.df['sma20'] = self.df['close'].rolling(window=20).mean()
        
        # Calculate price position relative to recent range
        self.df['high_10d'] = self.df['high'].rolling(window=10).max()
        self.df['low_10d'] = self.df['low'].rolling(window=10).min()
        self.df['price_position'] = (self.df['close'] - self.df['low_10d']) / (self.df['high_10d'] - self.df['low_10d'])
        
        # Uptrend: Close > SMA20 and SMA10 > SMA20
        self.df['uptrend'] = (self.df['close'] > self.df['sma20']) & (self.df['sma10'] > self.df['sma20'])
        
        # Downtrend: Close < SMA20 and SMA10 < SMA20
        self.df['downtrend'] = (self.df['close'] < self.df['sma20']) & (self.df['sma10'] < self.df['sma20'])
    
    def detect_marubozu(self) -> Dict[str, Any]:
        """
        Detect Marubozu patterns
        
        Returns:
            Dictionary with pattern information
        """
        if not self._is_pattern_enabled('marubozu'):
            return {'detected': False}
        
        try:
            # Get parameters
            shadow_threshold = self._get_param('marubozu', 'shadow_threshold', 0.05)
            body_pct_min = self._get_param('marubozu', 'body_pct', 0.95)
            
            # Get current candle
            current = self.df.iloc[-1]
            
            # Check if current candle is a Marubozu
            small_shadows = (current['upper_shadow'] / current['range'] < shadow_threshold and
                            current['lower_shadow'] / current['range'] < shadow_threshold)
            large_body = current['body_pct'] > body_pct_min
            
            is_marubozu = small_shadows and large_body
            
            # Determine pattern type
            if is_marubozu:
                if current['candle_color'] == 1:
                    pattern_type = 'bullish_marubozu'
                    signal = 'BUY'
                    signal_strength = 3
                else:
                    pattern_type = 'bearish_marubozu'
                    signal = 'SELL'
                    signal_strength = -3
            else:
                pattern_type = None
                signal = 'NEUTRAL'
                signal_strength = 0
            
            return {
                'detected': is_marubozu,
                'type': pattern_type,
                'signal': signal,
                'signal_strength': signal_strength,
                'body_pct': current['body_pct'],
                'candle_color': 'bullish' if current['candle_color'] == 1 else 'bearish'
            }
        
        except Exception as e:
            logger.error(f"Error detecting Marubozu pattern: {str(e)}")
            raise PatternDetectionError("marubozu", str(e))
    
    # [Other candlestick pattern detection methods as already shown...]
    
    def detect_all_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect all candlestick patterns
        
        Returns:
            Dictionary with all pattern information
        """
        logger.info("Detecting all candlestick patterns")
        
        # Detect individual patterns
        self.patterns['marubozu'] = self.detect_marubozu()
        self.patterns['doji'] = self.detect_doji()
        self.patterns['hammer'] = self.detect_hammer()
        self.patterns['engulfing'] = self.detect_engulfing()
        
        # Add more pattern detection calls here for a complete implementation
        # self.patterns['harami'] = self.detect_harami()
        # self.patterns['morning_star'] = self.detect_morning_star()
        # ...
        
        return self.patterns
    
    def get_pattern_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Get signals from detected patterns
        
        Returns:
            Dictionary with pattern signals
        """
        signals = {}
        
        for pattern_name, pattern_data in self.patterns.items():
            if pattern_data.get('detected', False):
                signals[pattern_name] = {
                    'type': pattern_data.get('type'),
                    'signal': pattern_data.get('signal'),
                    'strength': pattern_data.get('signal_strength', 0)
                }
        
        return signals
    
    def get_overall_signal(self) -> Dict[str, Any]:
        """
        Calculate overall signal from all detected patterns
        
        Returns:
            Dictionary with overall signal information
        """
        # If no patterns detected, return neutral
        if not self.patterns:
            return {'signal': 'NEUTRAL', 'strength': 0, 'confidence': 0}
        
        # Count detected patterns
        detected_patterns = {}
        for pattern_name, pattern_data in self.patterns.items():
            if pattern_data.get('detected', False):
                detected_patterns[pattern_name] = pattern_data
        
        if not detected_patterns:
            return {'signal': 'NEUTRAL', 'strength': 0, 'confidence': 0}
        
        # Calculate weighted signal strength
        total_strength = 0
        highest_bullish = 0
        highest_bearish = 0
        
        # Get weights from params
        pattern_weights = self.params.get('signals', {}).get('pattern_strength_weights', {})
        
        # Use default weights if not provided
        default_weights = {
            'bullish_marubozu': 3, 'bearish_marubozu': 3,
            'hammer': 3, 'hanging_man': 3,
            'bullish_engulfing': 4, 'bearish_engulfing': 4,
            'doji': 1
        }
        
        for pattern_name, pattern_data in detected_patterns.items():
            pattern_type = pattern_data.get('type')
            strength = pattern_data.get('signal_strength', 0)
            
            # Apply weight to strength
            weight = pattern_weights.get(pattern_type, default_weights.get(pattern_type, 1))
            weighted_strength = strength * weight
            
            # Add to total
            total_strength += weighted_strength
            
            # Track highest bullish and bearish signals
            if weighted_strength > highest_bullish:
                highest_bullish = weighted_strength
            elif weighted_strength < highest_bearish:
                highest_bearish = weighted_strength
        
        # Calculate average strength
        avg_strength = total_strength / len(detected_patterns)
        
        # Determine signal
        if avg_strength >= 2:
            signal = 'STRONG BUY'
        elif avg_strength >= 1:
            signal = 'BUY'
        elif avg_strength <= -2:
            signal = 'STRONG SELL'
        elif avg_strength <= -1:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'
        
        # Calculate confidence (higher when patterns agree)
        if highest_bullish > 0 and highest_bearish < 0:
            # Mixed signals reduce confidence
            confidence = 50
        else:
            # Strong agreement increases confidence
            confidence = 80
        
        return {
            'signal': signal,
            'strength': int(round(avg_strength)),
            'confidence': confidence,
            'patterns': [data.get('type') for data in detected_patterns.values() if data.get('type')]
        }


class ChartPatterns:
    """Chart pattern detection and interpretation"""
    
    def __init__(self, df: pd.DataFrame, params: Optional[Dict[str, Any]] = None):
        """
        Initialize with price data and parameters
        
        Args:
            df: DataFrame with OHLCV data
            params: Parameters for pattern detection
        """
        self.df = df.copy()
        self.params = params or {}
        
        # Container for detected patterns
        self.patterns = {}
        
        # Precompute swing points
        self.swing_highs, self.swing_lows = self.find_swing_points()
    
    def _get_param(self, pattern_name: str, param_key: str, default: Any = None) -> Any:
        """
        Get parameter value with fallback to default
        
        Args:
            pattern_name: Pattern name (e.g., 'head_and_shoulders')
            param_key: Parameter key (e.g., 'tolerance')
            default: Default value if parameter not found
            
        Returns:
            Parameter value
        """
        try:
            return self.params.get('patterns', {}).get('chart_patterns', {}).get(pattern_name, {}).get(param_key, default)
        except (AttributeError, KeyError):
            return default
    
    def _is_pattern_enabled(self, pattern_name: str) -> bool:
        """
        Check if pattern detection is enabled
        
        Args:
            pattern_name: Pattern name
            
        Returns:
            True if pattern detection is enabled
        """
        return self._get_param(pattern_name, 'enabled', True)
    
    def find_swing_points(self, window_size: Optional[int] = None) -> Tuple[List[int], List[int]]:
        """
        Find swing highs and lows in price data
        
        Args:
            window_size: Number of candles to look before/after for confirmation
                        (None to use automatic sizing)
                        
        Returns:
            Tuple of (swing_high_indexes, swing_low_indexes)
        """
        # Use automatic window sizing if not specified
        if window_size is None:
            # Adjust window based on dataframe length
            if len(self.df) <= 30:
                window_size = 2
            elif len(self.df) <= 100:
                window_size = 3
            else:
                window_size = 5
        
        swing_highs = []
        swing_lows = []
        
        # Need at least 2*window_size+1 candles
        if len(self.df) < 2*window_size+1:
            return swing_highs, swing_lows
        
        # Find swing highs
        for i in range(window_size, len(self.df)-window_size):
            is_swing_high = True
            is_swing_low = True
            
            # Check if current point is a swing high
            for j in range(1, window_size+1):
                # If any point in window is higher, not a swing high
                if self.df['high'].iloc[i-j] > self.df['high'].iloc[i] or self.df['high'].iloc[i+j] > self.df['high'].iloc[i]:
                    is_swing_high = False
                    break
            
            # Check if current point is a swing low
            for j in range(1, window_size+1):
                # If any point in window is lower, not a swing low
                if self.df['low'].iloc[i-j] < self.df['low'].iloc[i] or self.df['low'].iloc[i+j] < self.df['low'].iloc[i]:
                    is_swing_low = False
                    break
            
            if is_swing_high:
                swing_highs.append(i)
            
            if is_swing_low:
                swing_lows.append(i)
        
        return swing_highs, swing_lows
    
    def detect_head_and_shoulders(self) -> Dict[str, Any]:
        """
        Detect Head and Shoulders pattern (or inverse)
        
        Returns:
            Dictionary with pattern information
        """
        if not self._is_pattern_enabled('head_and_shoulders'):
            return {'detected': False}
        
        try:
            # Get parameters
            head_tolerance = self._get_param('head_and_shoulders', 'head_tolerance', 0.03)
            shoulder_tolerance = self._get_param('head_and_shoulders', 'shoulder_tolerance', 0.05)
            
            # Need at least 5 swing points to form H&S
            if len(self.swing_highs) < 3 or len(self.swing_lows) < 2:
                return {'detected': False}
            
            # Check for regular Head and Shoulders (bearish)
            h_and_s_detected = False
            inv_h_and_s_detected = False
            
            # Check for regular H&S in the most recent swing highs
            recent_highs = self.swing_highs[-5:]
            if len(recent_highs) >= 3:
                for i in range(len(recent_highs)-2):
                    # Get potential left shoulder, head, right shoulder
                    left_idx = recent_highs[i]
                    head_idx = recent_highs[i+1]
                    right_idx = recent_highs[i+2]
                    
                    left_shoulder = self.df['high'].iloc[left_idx]
                    head = self.df['high'].iloc[head_idx]
                    right_shoulder = self.df['high'].iloc[right_idx]
                    
                    # Check if head is higher than shoulders
                    head_higher = head > left_shoulder and head > right_shoulder
                    
                    # Check if shoulders are at similar heights
                    shoulders_aligned = abs(left_shoulder - right_shoulder) / left_shoulder < shoulder_tolerance
                    
                    # Check if head is significantly higher than shoulders
                    head_significant = (head - max(left_shoulder, right_shoulder)) / max(left_shoulder, right_shoulder) > head_tolerance
                    
                    if head_higher and shoulders_aligned and head_significant:
                        h_and_s_detected = True
                        break
            
            # Check for inverse H&S in the most recent swing lows
            recent_lows = self.swing_lows[-5:]
            if len(recent_lows) >= 3:
                for i in range(len(recent_lows)-2):
                    # Get potential left shoulder, head, right shoulder
                    left_idx = recent_lows[i]
                    head_idx = recent_lows[i+1]
                    right_idx = recent_lows[i+2]
                    
                    left_shoulder = self.df['low'].iloc[left_idx]
                    head = self.df['low'].iloc[head_idx]
                    right_shoulder = self.df['low'].iloc[right_idx]
                    
                    # Check if head is lower than shoulders
                    head_lower = head < left_shoulder and head < right_shoulder
                    
                    # Check if shoulders are at similar heights
                    shoulders_aligned = abs(left_shoulder - right_shoulder) / left_shoulder < shoulder_tolerance
                    
                    # Check if head is significantly lower than shoulders
                    head_significant = (min(left_shoulder, right_shoulder) - head) / min(left_shoulder, right_shoulder) > head_tolerance
                    
                    if head_lower and shoulders_aligned and head_significant:
                        inv_h_and_s_detected = True
                        break
            
            # Determine pattern signal
            if h_and_s_detected:
                pattern_type = 'head_and_shoulders'
                signal = 'SELL'
                signal_strength = -3
            elif inv_h_and_s_detected:
                pattern_type = 'inverse_head_and_shoulders'
                signal = 'BUY'
                signal_strength = 3
            else:
                pattern_type = None
                signal = 'NEUTRAL'
                signal_strength = 0
            
            return {
                'detected': h_and_s_detected or inv_h_and_s_detected,
                'type': pattern_type,
                'signal': signal,
                'signal_strength': signal_strength
            }
        
        except Exception as e:
            logger.error(f"Error detecting Head and Shoulders pattern: {str(e)}")
            raise PatternDetectionError("head_and_shoulders", str(e))
    
    def detect_double_patterns(self) -> Dict[str, Any]:
        """
        Detect Double Top or Double Bottom patterns
        
        Returns:
            Dictionary with pattern information
        """
        if not self._is_pattern_enabled('double_pattern'):
            return {'detected': False}
        
        try:
            # Get parameters
            tolerance = self._get_param('double_pattern', 'tolerance', 0.03)
            lookback = self._get_param('double_pattern', 'lookback', 50)
            
            # Need at least lookback candles
            if len(self.df) < lookback:
                return {'detected': False}
            
            # Focus on recent part of the data
            recent_df = self.df.iloc[-lookback:]
            
            # Get recent high and low point indexes
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            
            # Find all local maxima and minima
            local_maxima = []
            local_minima = []
            
            for i in range(1, len(recent_df)-1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    local_maxima.append(i)
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    local_minima.append(i)
            
            # Check for double top
            double_top_detected = False
            double_bottom_detected = False
            
            # Need at least 2 maxima for double top
            if len(local_maxima) >= 2:
                for i in range(len(local_maxima)-1):
                    first_high = highs[local_maxima[i]]
                    second_high = highs[local_maxima[i+1]]
                    
                    # Check if highs are at similar levels
                    if abs(first_high - second_high) / first_high < tolerance:
                        # Must have a significant trough between
                        if local_maxima[i+1] - local_maxima[i] > 10:  # At least 10 candles apart
                            # Find lowest point between peaks
                            between_lows = lows[local_maxima[i]:local_maxima[i+1]+1]
                            min_between = min(between_lows)
                            
                            # Check if trough is significant (at least 3% below peaks)
                            if (first_high - min_between) / first_high > 0.03:
                                double_top_detected = True
                                break
            
            # Need at least 2 minima for double bottom
            if len(local_minima) >= 2:
                for i in range(len(local_minima)-1):
                    first_low = lows[local_minima[i]]
                    second_low = lows[local_minima[i+1]]
                    
                    # Check if lows are at similar levels
                    if abs(first_low - second_low) / first_low < tolerance:
                        # Must have a significant peak between
                        if local_minima[i+1] - local_minima[i] > 10:  # At least 10 candles apart
                            # Find highest point between troughs
                            between_highs = highs[local_minima[i]:local_minima[i+1]+1]
                            max_between = max(between_highs)
                            
                            # Check if peak is significant (at least 3% above troughs)
                            if (max_between - first_low) / first_low > 0.03:
                                double_bottom_detected = True
                                break
            
            # Determine pattern signal
            if double_top_detected:
                pattern_type = 'double_top'
                signal = 'SELL'
                signal_strength = -3
            elif double_bottom_detected:
                pattern_type = 'double_bottom'
                signal = 'BUY'
                signal_strength = 3
            else:
                pattern_type = None
                signal = 'NEUTRAL'
                signal_strength = 0
            
            return {
                'detected': double_top_detected or double_bottom_detected,
                'type': pattern_type,
                'signal': signal,
                'signal_strength': signal_strength
            }
        
        except Exception as e:
            logger.error(f"Error detecting Double pattern: {str(e)}")
            raise PatternDetectionError("double_pattern", str(e))
    
    def detect_all_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect all chart patterns
        
        Returns:
            Dictionary with all pattern information
        """
        logger.info("Detecting all chart patterns")
        
        # Detect individual patterns
        self.patterns['head_and_shoulders'] = self.detect_head_and_shoulders()
        self.patterns['double_pattern'] = self.detect_double_patterns()
        
        # Add more pattern detection calls here for a complete implementation
        # self.patterns['triple_pattern'] = self.detect_triple_patterns()
        # self.patterns['wedges'] = self.detect_wedges()
        # self.patterns['rectangle'] = self.detect_rectangle()
        # self.patterns['flags'] = self.detect_flags()
        # self.patterns['cup_and_handle'] = self.detect_cup_and_handle()
        # ...
        
        return self.patterns
    
    def get_pattern_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Get signals from detected patterns
        
        Returns:
            Dictionary with pattern signals
        """
        signals = {}
        
        for pattern_name, pattern_data in self.patterns.items():
            if pattern_data.get('detected', False):
                signals[pattern_name] = {
                    'type': pattern_data.get('type'),
                    'signal': pattern_data.get('signal'),
                    'strength': pattern_data.get('signal_strength', 0)
                }
        
        return signals
    
    def get_overall_signal(self) -> Dict[str, Any]:
        """
        Calculate overall signal from all detected patterns
        
        Returns:
            Dictionary with overall signal information
        """
        # If no patterns detected, return neutral
        if not self.patterns:
            return {'signal': 'NEUTRAL', 'strength': 0, 'confidence': 0}
        
        # Count detected patterns
        detected_patterns = {}
        for pattern_name, pattern_data in self.patterns.items():
            if pattern_data.get('detected', False):
                detected_patterns[pattern_name] = pattern_data
        
        if not detected_patterns:
            return {'signal': 'NEUTRAL', 'strength': 0, 'confidence': 0}
        
        # Determine highest priority pattern
        # Chart patterns are usually more significant than candlestick patterns
        highest_bullish = 0
        highest_bearish = 0
        
        for pattern_name, pattern_data in detected_patterns.items():
            strength = pattern_data.get('signal_strength', 0)
            
            if strength > highest_bullish:
                highest_bullish = strength
            elif strength < highest_bearish:
                highest_bearish = strength
        
        # Determine signal
        if highest_bullish > abs(highest_bearish):
            if highest_bullish >= 3:
                signal = 'STRONG BUY'
            else:
                signal = 'BUY'
            strength = highest_bullish
        elif abs(highest_bearish) > highest_bullish:
            if abs(highest_bearish) >= 3:
                signal = 'STRONG SELL'
            else:
                signal = 'SELL'
            strength = highest_bearish
        else:
            signal = 'NEUTRAL'
            strength = 0
        
        # Chart patterns usually have high confidence
        confidence = 80
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'patterns': [data.get('type') for data in detected_patterns.values() if data.get('type')]
        }