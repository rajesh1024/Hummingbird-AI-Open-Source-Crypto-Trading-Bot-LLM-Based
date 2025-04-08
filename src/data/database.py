from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv
from .models import Base, Position, PositionStatus
from typing import List, Dict, Optional
from datetime import datetime
import logging
import traceback

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

load_dotenv()

def get_database_url():
    """Get database URL from environment variables or use Docker default"""
    # Check for Docker environment variable first
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        return database_url
        
    # Fallback to individual components
    db_host = os.getenv('POSTGRES_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('POSTGRES_DB', 'hummingbird')
    db_user = os.getenv('POSTGRES_USER', 'postgres')
    db_password = os.getenv('POSTGRES_PASSWORD', '')
    
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

class DatabaseManager:
    def __init__(self):
        """Initialize database connection"""
        try:
            self.engine = create_engine(get_database_url())
            self.Session = sessionmaker(bind=self.engine)
            self._session = None
            
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise
    
    def get_session(self):
        """Get or create a database session"""
        if not self._session:
            self._session = self.Session()
        return self._session
    
    def commit(self):
        """Commit current transaction"""
        if self._session:
            self._session.commit()
    
    def rollback(self):
        """Rollback current transaction"""
        if self._session:
            self._session.rollback()
    
    def close(self):
        """Close current session"""
        if self._session:
            self._session.close()
            self._session = None
    
    def execute_query(self, query, params=None):
        """Execute a raw SQL query"""
        try:
            session = self.get_session()
            result = session.execute(query, params or {})
            return result
        except SQLAlchemyError as e:
            print(f"Error executing query: {e}")
            raise
    
    def add_position(self, position_data):
        """Add a new position"""
        try:
            session = self.get_session()
            session.add(position_data)
            session.flush()  # Ensure the position is persisted
            return position_data
        except SQLAlchemyError as e:
            print(f"Error adding position: {e}")
            raise
    
    def update_position(self, position_id, update_data):
        """Update an existing position"""
        try:
            session = self.get_session()
            position = session.query(Position).filter_by(id=position_id).first()
            if position:
                for key, value in update_data.items():
                    setattr(position, key, value)
                session.flush()  # Ensure updates are persisted
                return position
            return None
        except SQLAlchemyError as e:
            print(f"Error updating position: {e}")
            raise
    
    def get_active_positions(self) -> List[Dict]:
        """Get all active positions"""
        try:
            session = self.get_session()
            positions = session.query(Position).filter(
                Position.status.in_([PositionStatus.OPEN, PositionStatus.PENDING])
            ).all()
            
            return [
                {
                    'id': str(pos.id),
                    'symbol': pos.symbol,
                    'type': pos.position_type.value.upper(),
                    'entry_price': float(pos.entry_price),
                    'current_price': float(pos.current_price),
                    'pnl': float(pos.pnl),
                    'stop_loss': float(pos.stop_loss),
                    'take_profit': float(pos.take_profit),
                    'size': float(pos.size),
                    'risk_reward_ratio': float(pos.risk_reward_ratio),
                    'position_strength': float(pos.position_strength),
                    'status': pos.status.value,
                    'created_at': pos.created_at
                }
                for pos in positions
            ]
        except SQLAlchemyError as e:
            print(f"Error getting active positions: {e}")
            return []
    
    def get_position_history(self, position_id):
        """Get position history"""
        try:
            session = self.get_session()
            position = session.query(Position).filter_by(id=position_id).first()
            if position:
                return {
                    'position': position,
                    'adjustment_history': position.adjustment_history
                }
            return None
        except SQLAlchemyError as e:
            print(f"Error getting position history: {e}")
            raise

    def get_closed_positions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None,
        type: Optional[str] = None
    ) -> List[Dict]:
        """Get closed positions with optional filtering"""
        try:
            logger.info("Starting get_closed_positions query")
            session = self.get_session()
            query = session.query(Position).filter(Position.status == PositionStatus.CLOSED)
            
            if start_date:
                logger.debug(f"Applying start_date filter: {start_date}")
                query = query.filter(Position.closed_at >= start_date)
            if end_date:
                logger.debug(f"Applying end_date filter: {end_date}")
                query = query.filter(Position.closed_at <= end_date)
            if symbol:
                logger.debug(f"Applying symbol filter: {symbol}")
                query = query.filter(Position.symbol == symbol)
            if type:
                logger.debug(f"Applying type filter: {type}")
                query = query.filter(Position.position_type == type)
            
            logger.debug(f"Executing query: {query}")
            positions = query.order_by(Position.closed_at.desc()).all()
            logger.info(f"Found {len(positions)} positions")
            
            result = []
            for pos in positions:
                try:
                    position_dict = {
                        'symbol': pos.symbol,
                        'type': pos.position_type.value.upper(),
                        'entry_price': float(pos.entry_price),
                        'exit_price': float(pos.current_price),
                        'pnl': float(pos.pnl),
                        'closed_reason': pos.closed_reason or '',
                        'duration': self._calculate_duration(pos.created_at, pos.closed_at),
                        'created_at': pos.created_at,
                        'closed_at': pos.closed_at
                    }
                    result.append(position_dict)
                except Exception as e:
                    logger.error(f"Error processing position {pos.id}: {str(e)}")
                    logger.error(f"Position data: symbol={pos.symbol}, type={pos.position_type}, entry_price={pos.entry_price}, current_price={pos.current_price}, pnl={pos.pnl}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in get_closed_positions: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _calculate_duration(self, start: datetime, end: datetime) -> str:
        """Calculate the duration between two timestamps in a human-readable format"""
        if not start or not end:
            return ''
        
        duration = (end - start).total_seconds() / 60  # Duration in minutes
        if duration < 60:
            return f"{int(duration)}m"
        else:
            hours = int(duration / 60)
            minutes = int(duration % 60)
            return f"{hours}h {minutes}m" 