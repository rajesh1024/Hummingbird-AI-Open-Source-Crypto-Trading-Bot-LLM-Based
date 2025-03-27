from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv
from .models import Base, Position, PositionStatus

load_dotenv()

def get_database_url():
    """Get database URL from environment variables"""
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'hummingbird')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', '')
    
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