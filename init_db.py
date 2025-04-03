from src.data.models import Base
from src.data.database import get_database_url
from sqlalchemy import create_engine, inspect, text
import os
import time
import psycopg2
from alembic.config import Config
from alembic import command

def wait_for_db():
    """Wait for the database to be ready"""
    max_retries = 30
    retry_interval = 2

    for _ in range(max_retries):
        try:
            conn = psycopg2.connect(
                dbname=os.getenv('POSTGRES_DB', 'hummingbird'),
                user=os.getenv('POSTGRES_USER', 'hummingbird'),
                password=os.getenv('POSTGRES_PASSWORD', 'hummingbird'),
                host=os.getenv('POSTGRES_HOST', 'postgres'),
                port=os.getenv('DB_PORT', '5432')
            )
            conn.close()
            return True
        except psycopg2.OperationalError:
            print("Waiting for database to be ready...")
            time.sleep(retry_interval)
    
    raise Exception("Could not connect to database after maximum retries")

def tables_exist(engine):
    """Check if the required tables already exist"""
    try:
        inspector = inspect(engine)
        required_tables = {'account_balance', 'positions'}
        existing_tables = set(inspector.get_table_names())
        return required_tables.issubset(existing_tables)
    except Exception as e:
        print(f"Error checking tables: {e}")
        return False

def run_migrations(engine):
    """Run Alembic migrations"""
    try:
        print("Running database migrations...")
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        print("Database migrations completed successfully")
        return True
    except Exception as e:
        print(f"Error running migrations: {e}")
        return False

def init_db():
    """Initialize the database and run migrations"""
    # Wait for database to be ready
    wait_for_db()
    
    # Create database engine
    engine = create_engine(get_database_url())
    
    try:
        # Check if tables exist
        if not tables_exist(engine):
            print("Creating database tables...")
            Base.metadata.create_all(engine)
            print("Database tables created successfully")
            
            # Run Alembic migrations only if tables were just created
            run_migrations(engine)
        else:
            print("Database tables already exist, checking for pending migrations...")
            # Check if there are any pending migrations
            try:
                alembic_cfg = Config("alembic.ini")
                current = command.current(alembic_cfg)
                head = command.heads(alembic_cfg)
                if current != head:
                    print("Pending migrations found, running them...")
                    run_migrations(engine)
                else:
                    print("No pending migrations, database is up to date")
            except Exception as e:
                print(f"Error checking migrations: {e}")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

if __name__ == "__main__":
    init_db() 