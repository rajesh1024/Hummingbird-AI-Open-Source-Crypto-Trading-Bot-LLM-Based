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
        except SQLAlchemyError as e:
            print(f"Error connecting to database: {e}")
            raise
    
    def get_session(self):
        """Get a new database session"""
        return self.Session()
    
    def close_session(self, session):
        """Close a database session"""
        if session:
            session.close()
    
    def execute_query(self, query, params=None):
        """Execute a raw SQL query"""
        session = self.get_session()
        try:
            result = session.execute(query, params or {})
            session.commit()
            return result
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Error executing query: {e}")
            raise
        finally:
            self.close_session(session)
    
    def add_position(self, position_data):
        """Add a new position"""
        session = self.get_session()
        try:
            session.add(position_data)
            session.commit()
            return position_data
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Error adding position: {e}")
            raise
        finally:
            self.close_session(session)
    
    def update_position(self, position_id, update_data):
        """Update an existing position"""
        session = self.get_session()
        try:
            position = session.query(Position).filter_by(id=position_id).first()
            if position:
                for key, value in update_data.items():
                    setattr(position, key, value)
                session.commit()
                return position
            return None
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Error updating position: {e}")
            raise
        finally:
            self.close_session(session)
    
    def get_active_positions(self):
        """Get all active positions"""
        session = self.get_session()
        try:
            return session.query(Position).filter_by(status=PositionStatus.ACTIVE).all()
        finally:
            self.close_session(session)
    
    def add_market_structure(self, market_structure_data):
        """Add new market structure data"""
        session = self.get_session()
        try:
            session.add(market_structure_data)
            session.commit()
            return market_structure_data
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Error adding market structure: {e}")
            raise
        finally:
            self.close_session(session)
    
    def add_exit_signal(self, exit_signal_data):
        """Add a new exit signal"""
        session = self.get_session()
        try:
            session.add(exit_signal_data)
            session.commit()
            return exit_signal_data
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Error adding exit signal: {e}")
            raise
        finally:
            self.close_session(session)
    
    def get_position_history(self, position_id):
        """Get complete position history including market structure and exit signals"""
        session = self.get_session()
        try:
            position = session.query(Position).filter_by(id=position_id).first()
            if position:
                return {
                    'position': position,
                    'market_structure': position.market_structure,
                    'order_blocks': position.order_blocks,
                    'exit_signals': position.exit_signals
                }
            return None
        finally:
            self.close_session(session) 