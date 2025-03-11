from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from .models import Base
import os
from dotenv import load_dotenv

def reset_database():
    """Reset the database by dropping and recreating all tables"""
    load_dotenv()
    
    # Get database URL from environment or use default
    db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/hummingbird')
    
    # Create engine
    engine = create_engine(db_url)
    
    # Drop all tables with CASCADE
    with engine.connect() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO postgres"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public"))
        conn.commit()
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    print("Database reset complete!")

if __name__ == "__main__":
    reset_database() 