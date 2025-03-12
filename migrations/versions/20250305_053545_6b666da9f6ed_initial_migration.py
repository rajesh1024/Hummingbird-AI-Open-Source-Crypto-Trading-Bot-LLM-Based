"""Initial migration

Revision ID: 6b666da9f6ed
Revises: 
Create Date: 2025-03-05 05:35:45.123456

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '6b666da9f6ed'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Drop existing tables if they exist
    connection = op.get_bind()
    connection.execute(text("""
        DROP TABLE IF EXISTS order_blocks CASCADE;
        DROP TABLE IF EXISTS liquidity_levels CASCADE;
        DROP TABLE IF EXISTS fair_value_gaps CASCADE;
        DROP TABLE IF EXISTS market_structures CASCADE;
        DROP TABLE IF EXISTS exit_signals CASCADE;
        DROP TABLE IF EXISTS positions CASCADE;
    """))
    
    # Create ENUM types if they don't exist
    result = connection.execute(text("SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'positiontype')"))
    if not result.scalar():
        connection.execute(text("CREATE TYPE positiontype AS ENUM ('LONG', 'SHORT')"))
    
    result = connection.execute(text("SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'positionstatus')"))
    if not result.scalar():
        connection.execute(text("CREATE TYPE positionstatus AS ENUM ('OPEN', 'CLOSED', 'PENDING')"))
    
    result = connection.execute(text("SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'marketstructure')"))
    if not result.scalar():
        connection.execute(text("CREATE TYPE marketstructure AS ENUM ('BULLISH', 'BEARISH', 'NEUTRAL', 'SHIFTING')"))
    
    # Create tables
    op.create_table('positions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(), nullable=False),
        sa.Column('position_type', postgresql.ENUM('LONG', 'SHORT', name='positiontype', create_type=False), nullable=False),
        sa.Column('status', postgresql.ENUM('OPEN', 'CLOSED', 'PENDING', name='positionstatus', create_type=False), nullable=True),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('current_price', sa.Float(), nullable=False),
        sa.Column('stop_loss', sa.Float(), nullable=False),
        sa.Column('take_profit', sa.Float(), nullable=False),
        sa.Column('size', sa.Float(), nullable=False),
        sa.Column('pnl', sa.Float(), nullable=True),
        sa.Column('timeframe', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('exit_signals',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('position_id', sa.Integer(), nullable=False),
        sa.Column('signal_type', sa.String(), nullable=False),
        sa.Column('status', postgresql.ENUM('OPEN', 'CLOSED', 'PENDING', name='positionstatus', create_type=False), nullable=True),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['position_id'], ['positions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('market_structures',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(), nullable=False),
        sa.Column('timeframe', sa.String(), nullable=False),
        sa.Column('structure_type', postgresql.ENUM('BULLISH', 'BEARISH', 'NEUTRAL', 'SHIFTING', name='marketstructure', create_type=False), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('fair_value_gaps',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(), nullable=False),
        sa.Column('timeframe', sa.String(), nullable=False),
        sa.Column('high_price', sa.Float(), nullable=False),
        sa.Column('low_price', sa.Float(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('liquidity_levels',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(), nullable=False),
        sa.Column('timeframe', sa.String(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('order_blocks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(), nullable=False),
        sa.Column('timeframe', sa.String(), nullable=False),
        sa.Column('high_price', sa.Float(), nullable=False),
        sa.Column('low_price', sa.Float(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade():
    op.drop_table('order_blocks')
    op.drop_table('liquidity_levels')
    op.drop_table('fair_value_gaps')
    op.drop_table('market_structures')
    op.drop_table('exit_signals')
    op.drop_table('positions')
    
    # Drop ENUM types
    connection = op.get_bind()
    connection.execute(text("DROP TYPE IF EXISTS marketstructure"))
    connection.execute(text("DROP TYPE IF EXISTS positionstatus"))
    connection.execute(text("DROP TYPE IF EXISTS positiontype")) 