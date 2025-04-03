"""Add account_balance relationship to positions

Revision ID: add_account_balance_relationship
Revises: add_account_balance
Create Date: 2025-03-21 10:00:00.000000+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_account_balance_relationship'
down_revision = 'add_account_balance'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add account_balance_id column to positions table
    op.add_column('positions',
        sa.Column('account_balance_id', sa.Integer(), nullable=True)
    )
    
    # Add foreign key constraint
    op.create_foreign_key(
        'fk_positions_account_balance',
        'positions', 'account_balance',
        ['account_balance_id'], ['id']
    )
    
    # Update existing positions to use the first account balance
    op.execute("UPDATE positions SET account_balance_id = (SELECT id FROM account_balance LIMIT 1)")


def downgrade() -> None:
    # Drop foreign key constraint
    op.drop_constraint('fk_positions_account_balance', 'positions', type_='foreignkey')
    
    # Drop account_balance_id column
    op.drop_column('positions', 'account_balance_id') 