from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv
from .models import Base

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.Session = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection"""
        try:
            db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/hummingbird')
            self.engine = create_engine(db_url)
            self.Session = sessionmaker(bind=self.engine)
            Base.metadata.create_all(self.engine)
            self.session = self.Session()
        except SQLAlchemyError as e:
            print(f"Error connecting to database: {e}")
            raise
    
    def get_session(self):
        """Get the current database session"""
        if not self.session or not self.session.is_active:
            self.session = self.Session()
        return self.session
    
    def commit(self):
        """Commit the current transaction"""
        try:
            if self.session and self.session.is_active:
                self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e
    
    def rollback(self):
        """Rollback the current transaction"""
        if self.session and self.session.is_active:
            self.session.rollback()
    
    def close(self):
        """Close the current session"""
        if self.session:
            if self.session.is_active:
                self.session.close()
            self.session = None
    
    def __del__(self):
        self.close()
    
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
    
    def get_active_positions(self):
        """Get all active positions"""
        try:
            session = self.get_session()
            return session.query(Position).filter(
                Position.status.in_([PositionStatus.OPEN, PositionStatus.PENDING])
            ).all()
        except SQLAlchemyError as e:
            print(f"Error getting active positions: {e}")
            raise
    
    def add_market_structure(self, market_structure_data):
        """Add new market structure data"""
        try:
            session = self.get_session()
            session.add(market_structure_data)
            session.flush()  # Ensure the data is persisted
            return market_structure_data
        except SQLAlchemyError as e:
            print(f"Error adding market structure: {e}")
            raise
    
    def add_exit_signal(self, exit_signal_data):
        """Add a new exit signal"""
        try:
            session = self.get_session()
            session.add(exit_signal_data)
            session.flush()  # Ensure the signal is persisted
            return exit_signal_data
        except SQLAlchemyError as e:
            print(f"Error adding exit signal: {e}")
            raise
    
    def get_position_history(self, position_id):
        """Get complete position history including market structure and exit signals"""
        try:
            session = self.get_session()
            position = session.query(Position).filter_by(id=position_id).first()
            if position:
                return {
                    'position': position,
                    'market_structure': position.market_structure,
                    'order_blocks': position.order_blocks,
                    'exit_signals': position.exit_signals
                }
            return None
        except SQLAlchemyError as e:
            print(f"Error getting position history: {e}")
            raise 