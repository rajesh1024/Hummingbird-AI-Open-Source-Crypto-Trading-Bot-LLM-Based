"""Remove market structures and exit signals tables

Revision ID: remove_market_structures
Revises: add_position_management_fields
Create Date: 2025-03-05 05:35:47.623635+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'remove_market_structures'
down_revision = 'add_position_management_fields'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop tables in correct order (respecting foreign key constraints)
    op.drop_table('liquidity_levels')
    op.drop_table('fair_value_gaps')
    op.drop_table('order_blocks')
    op.drop_table('market_structures')
    op.drop_table('exit_signals')
    
    # Add closed_at column to positions table
    op.add_column('positions', sa.Column('closed_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    # Remove closed_at column from positions table
    op.drop_column('positions', 'closed_at')
    
    # Recreate tables in reverse order
    op.create_table('exit_signals',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('position_id', sa.Integer(), nullable=False),
        sa.Column('signal_type', sa.String(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('reason', sa.String(), nullable=False),
        sa.Column('confidence', sa.Float(), default=0.0),
        sa.Column('model_analysis', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['position_id'], ['positions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('market_structures',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('position_id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(), nullable=False),
        sa.Column('timeframe', sa.String(), nullable=False),
        sa.Column('structure_type', sa.String(), nullable=False),
        sa.Column('high', sa.Float(), nullable=False),
        sa.Column('low', sa.Float(), nullable=False),
        sa.Column('volume', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('smart_money_traps', postgresql.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['position_id'], ['positions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('order_blocks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('market_structure_id', sa.Integer(), nullable=False),
        sa.Column('block_type', sa.String(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('volume', sa.Float(), nullable=False),
        sa.Column('strength', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['market_structure_id'], ['market_structures.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('fair_value_gaps',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('market_structure_id', sa.Integer(), nullable=False),
        sa.Column('gap_type', sa.String(), nullable=False),
        sa.Column('upper_price', sa.Float(), nullable=False),
        sa.Column('lower_price', sa.Float(), nullable=False),
        sa.Column('volume', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['market_structure_id'], ['market_structures.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('liquidity_levels',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('market_structure_id', sa.Integer(), nullable=False),
        sa.Column('level_type', sa.String(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('volume', sa.Float(), nullable=False),
        sa.Column('strength', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['market_structure_id'], ['market_structures.id'], ),
        sa.PrimaryKeyConstraint('id')
    ) 