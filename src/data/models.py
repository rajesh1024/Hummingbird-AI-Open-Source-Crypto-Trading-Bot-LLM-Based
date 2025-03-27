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
    closed_at = Column(DateTime, nullable=True)
    closed_reason = Column(String, nullable=True)
    
    # New fields for enhanced position management
    risk_reward_ratio = Column(Float, default=0.0)
    last_analysis_time = Column(DateTime)
    position_strength = Column(Float, default=0.0)
    last_adjustment_reason = Column(String)
    adjustment_history = Column(JSON, default=list)  # Store history of adjustments
    model_confidence = Column(Float, default=0.0)
    analysis_reasoning = Column(String) 