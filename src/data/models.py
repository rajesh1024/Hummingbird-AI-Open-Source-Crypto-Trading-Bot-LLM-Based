from enum import Enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum as SQLEnum, JSON
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class PositionType(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"

class MarketStructure(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    SHIFTING = "SHIFTING"

class Position(Base):
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    position_type = Column(SQLEnum(PositionType), nullable=False)
    status = Column(SQLEnum(PositionStatus), default=PositionStatus.PENDING)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    pnl = Column(Float, default=0.0)
    timeframe = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    market_structures = relationship("MarketStructureData", back_populates="position")
    exit_signals = relationship("ExitSignal", back_populates="position")

class MarketStructureData(Base):
    __tablename__ = 'market_structures'
    
    id = Column(Integer, primary_key=True)
    position_id = Column(Integer, ForeignKey('positions.id'))
    timeframe = Column(String, nullable=False)
    structure_type = Column(SQLEnum(MarketStructure), nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    position = relationship("Position", back_populates="market_structures")
    order_blocks = relationship("OrderBlock", back_populates="market_structure")
    fair_value_gaps = relationship("FairValueGap", back_populates="market_structure")
    liquidity_levels = relationship("LiquidityLevel", back_populates="market_structure")
    smart_money_traps = Column(JSON)  # Store as JSON array

class OrderBlock(Base):
    __tablename__ = 'order_blocks'
    
    id = Column(Integer, primary_key=True)
    market_structure_id = Column(Integer, ForeignKey('market_structures.id'))
    block_type = Column(String, nullable=False)  # 'institutional', 'smart_money'
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    strength = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    market_structure = relationship("MarketStructureData", back_populates="order_blocks")

class FairValueGap(Base):
    __tablename__ = 'fair_value_gaps'
    
    id = Column(Integer, primary_key=True)
    market_structure_id = Column(Integer, ForeignKey('market_structures.id'))
    gap_type = Column(String, nullable=False)  # 'bullish', 'bearish'
    upper_price = Column(Float, nullable=False)
    lower_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    market_structure = relationship("MarketStructureData", back_populates="fair_value_gaps")

class LiquidityLevel(Base):
    __tablename__ = 'liquidity_levels'
    
    id = Column(Integer, primary_key=True)
    market_structure_id = Column(Integer, ForeignKey('market_structures.id'))
    level_type = Column(String, nullable=False)  # 'support', 'resistance'
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    strength = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    market_structure = relationship("MarketStructureData", back_populates="liquidity_levels")

class ExitSignal(Base):
    __tablename__ = 'exit_signals'
    
    id = Column(Integer, primary_key=True)
    position_id = Column(Integer, ForeignKey('positions.id'))
    signal_type = Column(String, nullable=False)  # 'structure_shift', 'strength_loss', 'stop_loss', 'take_profit'
    price = Column(Float, nullable=False)
    reason = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    position = relationship("Position", back_populates="exit_signals") 