"""
Database utilities for QuantM10

Provides functions for storing and retrieving backtest results
and signal performance tracking data.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Default database path
DEFAULT_DB = DATA_DIR / "quantm10.db"


def connect_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Connect to SQLite database
    
    Args:
        db_path: Path to database file (None for default)
        
    Returns:
        Database connection
    """
    if db_path is None:
        db_path = DEFAULT_DB
    
    conn = sqlite3.connect(db_path)
    
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    
    # Use Row factory for dictionary-like rows
    conn.row_factory = sqlite3.Row
    
    return conn


def init_db(conn: Optional[sqlite3.Connection] = None, db_path: Optional[str] = None) -> None:
    """
    Initialize database schema
    
    Args:
        conn: Optional database connection
        db_path: Path to database file (None for default)
    """
    # Connect if connection not provided
    close_conn = False
    if conn is None:
        conn = connect_db(db_path)
        close_conn = True
    
    # Create tables
    conn.executescript("""
    -- Instruments table
    CREATE TABLE IF NOT EXISTS instruments (
        instrument_key TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        name TEXT NOT NULL,
        exchange TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Backtest runs table
    CREATE TABLE IF NOT EXISTS backtest_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instrument_key TEXT NOT NULL,
        run_date TIMESTAMP NOT NULL,
        start_date DATE NOT NULL,
        end_date DATE NOT NULL,
        initial_capital REAL NOT NULL,
        strategy_name TEXT NOT NULL,
        params TEXT NOT NULL,  -- JSON string
        metrics TEXT NOT NULL,  -- JSON string
        FOREIGN KEY (instrument_key) REFERENCES instruments(instrument_key)
    );
    
    -- Backtest trades table
    CREATE TABLE IF NOT EXISTS backtest_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        backtest_id INTEGER NOT NULL,
        entry_date TIMESTAMP NOT NULL,
        exit_date TIMESTAMP,
        entry_price REAL NOT NULL,
        exit_price REAL,
        position_size REAL NOT NULL,
        pnl REAL,
        pnl_pct REAL,
        trade_type TEXT NOT NULL,  -- 'LONG' or 'SHORT'
        exit_reason TEXT,
        FOREIGN KEY (backtest_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
    );
    
    -- Signal tracking table
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instrument_key TEXT NOT NULL,
        signal_date TIMESTAMP NOT NULL,
        signal_type TEXT NOT NULL,  -- 'BUY', 'SELL', 'NEUTRAL'
        entry_price REAL NOT NULL,
        stop_loss REAL,
        target_price REAL,
        strength INTEGER NOT NULL,
        indicators TEXT NOT NULL,  -- JSON string
        patterns TEXT NOT NULL,    -- JSON string
        result TEXT,               -- 'SUCCESS', 'FAILURE', 'UNKNOWN'
        close_date TIMESTAMP,
        close_price REAL,
        pnl_pct REAL,
        notes TEXT,
        FOREIGN KEY (instrument_key) REFERENCES instruments(instrument_key)
    );
    """)
    
    # Close connection if we opened it
    if close_conn:
        conn.commit()
        conn.close()


def save_backtest_result(
    instrument_key: str,
    symbol: str,
    name: str,
    exchange: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    initial_capital: float,
    strategy_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    trades: List[Dict[str, Any]],
    conn: Optional[sqlite3.Connection] = None
) -> int:
    """
    Save backtest result to database
    
    Args:
        instrument_key: Instrument identifier
        symbol: Instrument symbol
        name: Instrument name
        exchange: Exchange
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Initial capital
        strategy_name: Strategy name
        params: Strategy parameters
        metrics: Performance metrics
        trades: List of trades
        conn: Optional database connection
        
    Returns:
        Backtest run ID
    """
    # Connect if connection not provided
    close_conn = False
    if conn is None:
        conn = connect_db()
        close_conn = True
    
    try:
        # Convert dates to strings if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        # Insert or update instrument
        conn.execute(
            """
            INSERT OR REPLACE INTO instruments
            (instrument_key, symbol, name, exchange)
            VALUES (?, ?, ?, ?)
            """,
            (instrument_key, symbol, name, exchange)
        )
        
        # Insert backtest run
        cursor = conn.execute(
            """
            INSERT INTO backtest_runs
            (instrument_key, run_date, start_date, end_date, initial_capital, 
             strategy_name, params, metrics)
            VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
            """,
            (
                instrument_key, start_date, end_date, initial_capital,
                strategy_name, json.dumps(params), json.dumps(metrics)
            )
        )
        
        # Get backtest run ID
        backtest_id = cursor.lastrowid
        
        # Insert trades
        if trades:
            for trade in trades:
                conn.execute(
                    """
                    INSERT INTO backtest_trades
                    (backtest_id, entry_date, exit_date, entry_price, exit_price,
                     position_size, pnl, pnl_pct, trade_type, exit_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        backtest_id, trade['entry_date'], trade.get('exit_date'),
                        trade['entry_price'], trade.get('exit_price'),
                        trade['position_size'], trade.get('pnl'), trade.get('pnl_pct'),
                        trade['trade_type'], trade.get('exit_reason')
                    )
                )
        
        # Commit transaction
        conn.commit()
        
        return backtest_id
    
    except Exception as e:
        # Rollback on error
        conn.rollback()
        raise
    
    finally:
        # Close connection if we opened it
        if close_conn:
            conn.close()


def save_signal(
    instrument_key: str,
    symbol: str,
    name: str,
    exchange: str,
    signal_type: str,
    entry_price: float,
    stop_loss: Optional[float],
    target_price: Optional[float],
    strength: int,
    indicators: Dict[str, Any],
    patterns: Dict[str, Any],
    conn: Optional[sqlite3.Connection] = None
) -> int:
    """
    Save trading signal to database
    
    Args:
        instrument_key: Instrument identifier
        symbol: Instrument symbol
        name: Instrument name
        exchange: Exchange
        signal_type: Signal type (BUY, SELL, NEUTRAL)
        entry_price: Entry price
        stop_loss: Stop loss price
        target_price: Target price
        strength: Signal strength
        indicators: Indicator data
        patterns: Pattern data
        conn: Optional database connection
        
    Returns:
        Signal ID
    """
    # Connect if connection not provided
    close_conn = False
    if conn is None:
        conn = connect_db()
        close_conn = True
    
    try:
        # Insert or update instrument
        conn.execute(
            """
            INSERT OR REPLACE INTO instruments
            (instrument_key, symbol, name, exchange)
            VALUES (?, ?, ?, ?)
            """,
            (instrument_key, symbol, name, exchange)
        )
        
        # Insert signal
        cursor = conn.execute(
            """
            INSERT INTO signals
            (instrument_key, signal_date, signal_type, entry_price, stop_loss,
             target_price, strength, indicators, patterns)
            VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                instrument_key, signal_type, entry_price, stop_loss,
                target_price, strength, json.dumps(indicators), json.dumps(patterns)
            )
        )
        
        # Get signal ID
        signal_id = cursor.lastrowid
        
        # Commit transaction
        conn.commit()
        
        return signal_id
    
    except Exception as e:
        # Rollback on error
        conn.rollback()
        raise
    
    finally:
        # Close connection if we opened it
        if close_conn:
            conn.close()


def update_signal_result(
    signal_id: int,
    result: str,
    close_date: Union[str, datetime],
    close_price: float,
    pnl_pct: float,
    notes: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> None:
    """
    Update signal with result
    
    Args:
        signal_id: Signal ID
        result: Result (SUCCESS, FAILURE, UNKNOWN)
        close_date: Close date
        close_price: Close price
        pnl_pct: Profit/loss percentage
        notes: Optional notes
        conn: Optional database connection
    """
    # Connect if connection not provided
    close_conn = False
    if conn is None:
        conn = connect_db()
        close_conn = True
    
    try:
        # Convert date to string if needed
        if isinstance(close_date, datetime):
            close_date = close_date.strftime('%Y-%m-%d %H:%M:%S')
        
        # Update signal
        conn.execute(
            """
            UPDATE signals
            SET result = ?, close_date = ?, close_price = ?, pnl_pct = ?, notes = ?
            WHERE id = ?
            """,
            (result, close_date, close_price, pnl_pct, notes, signal_id)
        )
        
        # Commit transaction
        conn.commit()
    
    except Exception as e:
        # Rollback on error
        conn.rollback()
        raise
    
    finally:
        # Close connection if we opened it
        if close_conn:
            conn.close()


def get_backtest_results(
    instrument_key: Optional[str] = None,
    strategy_name: Optional[str] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    limit: int = 10,
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    Get backtest results from database
    
    Args:
        instrument_key: Filter by instrument
        strategy_name: Filter by strategy
        start_date: Filter by start date
        end_date: Filter by end date
        limit: Maximum number of results
        conn: Optional database connection
        
    Returns:
        List of backtest results
    """
    # Connect if connection not provided
    close_conn = False
    if conn is None:
        conn = connect_db()
        close_conn = True
    
    try:
        # Build query
        query = """
        SELECT br.*, i.symbol, i.name, i.exchange
        FROM backtest_runs br
        JOIN instruments i ON br.instrument_key = i.instrument_key
        WHERE 1=1
        """
        params = []
        
        # Add filters
        if instrument_key:
            query += " AND br.instrument_key = ?"
            params.append(instrument_key)
        
        if strategy_name:
            query += " AND br.strategy_name = ?"
            params.append(strategy_name)
        
        if start_date:
            # Convert date to string if needed
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            
            query += " AND br.run_date >= ?"
            params.append(start_date)
        
        if end_date:
            # Convert date to string if needed
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            
            query += " AND br.run_date <= ?"
            params.append(end_date)
        
        # Add ordering and limit
        query += " ORDER BY br.run_date DESC LIMIT ?"
        params.append(limit)
        
        # Execute query
        cursor = conn.execute(query, params)
        
        # Convert to list of dictionaries
        results = []
        for row in cursor:
            result = dict(row)
            
            # Parse JSON fields
            result['params'] = json.loads(result['params'])
            result['metrics'] = json.loads(result['metrics'])
            
            # Get trades for this backtest
            trades_cursor = conn.execute(
                """
                SELECT * FROM backtest_trades
                WHERE backtest_id = ?
                ORDER BY entry_date
                """,
                (result['id'],)
            )
            
            result['trades'] = [dict(trade) for trade in trades_cursor]
            
            results.append(result)
        
        return results
    
    finally:
        # Close connection if we opened it
        if close_conn:
            conn.close()


def get_signal_history(
    instrument_key: Optional[str] = None,
    signal_type: Optional[str] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    min_strength: Optional[int] = None,
    result: Optional[str] = None,
    limit: int = 100,
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    Get signal history from database
    
    Args:
        instrument_key: Filter by instrument
        signal_type: Filter by signal type
        start_date: Filter by start date
        end_date: Filter by end date
        min_strength: Filter by minimum strength
        result: Filter by result
        limit: Maximum number of results
        conn: Optional database connection
        
    Returns:
        List of signals
    """
    # Connect if connection not provided
    close_conn = False
    if conn is None:
        conn = connect_db()
        close_conn = True
    
    try:
        # Build query
        query = """
        SELECT s.*, i.symbol, i.name, i.exchange
        FROM signals s
        JOIN instruments i ON s.instrument_key = i.instrument_key
        WHERE 1=1
        """
        params = []
        
        # Add filters
        if instrument_key:
            query += " AND s.instrument_key = ?"
            params.append(instrument_key)
        
        if signal_type:
            query += " AND s.signal_type = ?"
            params.append(signal_type)
        
        if start_date:
            # Convert date to string if needed
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            
            query += " AND s.signal_date >= ?"
            params.append(start_date)
        
        if end_date:
            # Convert date to string if needed
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            
            query += " AND s.signal_date <= ?"
            params.append(end_date)
        
        if min_strength is not None:
            query += " AND s.strength >= ?"
            params.append(min_strength)
        
        if result:
            query += " AND s.result = ?"
            params.append(result)
        
        # Add ordering and limit
        query += " ORDER BY s.signal_date DESC LIMIT ?"
        params.append(limit)
        
        # Execute query
        cursor = conn.execute(query, params)
        
        # Convert to list of dictionaries
        results = []
        for row in cursor:
            result = dict(row)
            
            # Parse JSON fields
            result['indicators'] = json.loads(result['indicators'])
            result['patterns'] = json.loads(result['patterns'])
            
            results.append(result)
        
        return results
    
    finally:
        # Close connection if we opened it
        if close_conn:
            conn.close()


def convert_json_to_db(
    json_file: str,
    db_path: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> None:
    """
    Convert JSON results to database
    
    Args:
        json_file: Path to JSON file
        db_path: Path to database file (None for default)
        conn: Optional database connection
    """
    # Connect if connection not provided
    close_conn = False
    if conn is None:
        conn = connect_db(db_path)
        close_conn = True
    
    try:
        # Initialize database
        init_db(conn)
        
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Check if data is a collection or single item
        if isinstance(data, list):
            items = data
        else:
            items = [data]
        
        # Process each item
        for item in items:
            # Determine if it's a signal or backtest
            if 'signal_type' in item:
                # Save signal
                save_signal(
                    instrument_key=item['instrument_key'],
                    symbol=item.get('symbol', item.get('stock_symbol', '')),
                    name=item.get('name', item.get('stock_name', '')),
                    exchange=item.get('exchange', ''),
                    signal_type=item['signal_type'],
                    entry_price=item['entry_price'],
                    stop_loss=item.get('stop_loss'),
                    target_price=item.get('target_price'),
                    strength=item.get('strength', item.get('signal_strength', 0)),
                    indicators=item.get('indicators', {}),
                    patterns=item.get('patterns', {}),
                    conn=conn
                )
            elif 'metrics' in item:
                # Save backtest
                save_backtest_result(
                    instrument_key=item['instrument_key'],
                    symbol=item.get('symbol', item.get('stock_symbol', '')),
                    name=item.get('name', item.get('stock_name', '')),
                    exchange=item.get('exchange', ''),
                    start_date=item.get('start_date', ''),
                    end_date=item.get('end_date', ''),
                    initial_capital=item.get('initial_capital', 100000.0),
                    strategy_name=item.get('strategy_name', 'default'),
                    params=item.get('params', {}),
                    metrics=item['metrics'],
                    trades=item.get('trades', []),
                    conn=conn
                )
        
        # Commit transaction
        conn.commit()
    
    except Exception as e:
        # Rollback on error
        conn.rollback()
        raise
    
    finally:
        # Close connection if we opened it
        if close_conn:
            conn.close()


def get_performance_summary(
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    min_strength: int = 0,
    conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Any]:
    """
    Get signal performance summary
    
    Args:
        start_date: Start date for filtering
        end_date: End date for filtering
        min_strength: Minimum signal strength
        conn: Optional database connection
        
    Returns:
        Dictionary with performance summary
    """
    # Connect if connection not provided
    close_conn = False
    if conn is None:
        conn = connect_db()
        close_conn = True
    
    try:
        # Convert dates to strings if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        # Build query params
        params = [min_strength]
        date_condition = ""
        
        if start_date:
            date_condition += " AND signal_date >= ?"
            params.append(start_date)
        
        if end_date:
            date_condition += " AND signal_date <= ?"
            params.append(end_date)
        
        # Get overall stats
        cursor = conn.execute(
            f"""
            SELECT 
                COUNT(*) as total_signals,
                SUM(CASE WHEN result = 'SUCCESS' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN result = 'FAILURE' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN result IS NULL THEN 1 ELSE 0 END) as pending,
                AVG(CASE WHEN result = 'SUCCESS' THEN pnl_pct ELSE NULL END) as avg_win_pct,
                AVG(CASE WHEN result = 'FAILURE' THEN pnl_pct ELSE NULL END) as avg_loss_pct,
                MAX(CASE WHEN result = 'SUCCESS' THEN pnl_pct ELSE NULL END) as max_win_pct,
                MIN(CASE WHEN result = 'FAILURE' THEN pnl_pct ELSE NULL END) as max_loss_pct
            FROM signals
            WHERE strength >= ?{date_condition}
            """,
            params
        )
        
        overall = dict(cursor.fetchone())
        
        # Calculate win rate
        total_decided = overall['successful'] + overall['failed']
        overall['win_rate'] = (overall['successful'] / total_decided * 100) if total_decided > 0 else 0
        
        # Get stats by signal type
        cursor = conn.execute(
            f"""
            SELECT 
                signal_type,
                COUNT(*) as total,
                SUM(CASE WHEN result = 'SUCCESS' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN result = 'FAILURE' THEN 1 ELSE 0 END) as failed,
                AVG(CASE WHEN result = 'SUCCESS' THEN pnl_pct ELSE NULL END) as avg_win_pct,
                AVG(CASE WHEN result = 'FAILURE' THEN pnl_pct ELSE NULL END) as avg_loss_pct
            FROM signals
            WHERE strength >= ?{date_condition}
            GROUP BY signal_type
            """,
            params
        )
        
        by_type = {}
        for row in cursor:
            type_stats = dict(row)
            signal_type = type_stats.pop('signal_type')
            total_decided = type_stats['successful'] + type_stats['failed']
            type_stats['win_rate'] = (type_stats['successful'] / total_decided * 100) if total_decided > 0 else 0
            by_type[signal_type] = type_stats
        
        # Get stats by strength
        cursor = conn.execute(
            f"""
            SELECT 
                strength,
                COUNT(*) as total,
                SUM(CASE WHEN result = 'SUCCESS' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN result = 'FAILURE' THEN 1 ELSE 0 END) as failed
            FROM signals
            WHERE strength >= ?{date_condition}
            GROUP BY strength
            ORDER BY strength
            """,
            params
        )
        
        by_strength = {}
        for row in cursor:
            strength_stats = dict(row)
            strength = strength_stats.pop('strength')
            total_decided = strength_stats['successful'] + strength_stats['failed']
            strength_stats['win_rate'] = (strength_stats['successful'] / total_decided * 100) if total_decided > 0 else 0
            by_strength[strength] = strength_stats
        
        # Get top performing instruments
        cursor = conn.execute(
            f"""
            SELECT 
                s.instrument_key,
                i.symbol,
                i.name,
                COUNT(*) as total_signals,
                SUM(CASE WHEN result = 'SUCCESS' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN result = 'FAILURE' THEN 1 ELSE 0 END) as failed,
                AVG(CASE WHEN result = 'SUCCESS' THEN pnl_pct ELSE NULL END) as avg_win_pct
            FROM signals s
            JOIN instruments i ON s.instrument_key = i.instrument_key
            WHERE s.strength >= ?{date_condition} AND s.result IS NOT NULL
            GROUP BY s.instrument_key
            ORDER BY (SUM(CASE WHEN result = 'SUCCESS' THEN 1 ELSE 0 END) * 1.0 / 
                     (SUM(CASE WHEN result = 'SUCCESS' THEN 1 ELSE 0 END) + 
                      SUM(CASE WHEN result = 'FAILURE' THEN 1 ELSE 0 END))) DESC,
                     avg_win_pct DESC
            LIMIT 10
            """,
            params
        )
        
        top_instruments = []
        for row in cursor:
            inst_stats = dict(row)
            total_decided = inst_stats['successful'] + inst_stats['failed']
            inst_stats['win_rate'] = (inst_stats['successful'] / total_decided * 100) if total_decided > 0 else 0
            top_instruments.append(inst_stats)
        
        # Return combined results
        return {
            'overall': overall,
            'by_type': by_type,
            'by_strength': by_strength,
            'top_instruments': top_instruments
        }
    
    finally:
        # Close connection if we opened it
        if close_conn:
            conn.close()