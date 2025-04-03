"""Add account_balance table

Revision ID: add_account_balance
Revises: add_closed_reason
Create Date: 2025-03-21 09:30:00.000000+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_account_balance'
down_revision = 'add_closed_reason'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create account_balance table
    op.create_table('account_balance',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('balance', sa.Float(), nullable=False, server_default='100.0'),
        sa.Column('last_updated', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Insert initial balance of $100
    op.execute("INSERT INTO account_balance (balance) VALUES (100.0)")


def downgrade() -> None:
    # Drop account_balance table
    op.drop_table('account_balance') 