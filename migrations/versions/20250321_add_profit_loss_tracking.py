"""Add profit/loss tracking to positions

Revision ID: add_profit_loss_tracking
Revises: add_account_balance_relationship
Create Date: 2025-03-21 11:00:00.000000+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_profit_loss_tracking'
down_revision = 'add_account_balance_relationship'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add profit/loss tracking columns to positions
    op.add_column('positions',
        sa.Column('profit_loss', sa.Float(), nullable=True)
    )
    op.add_column('positions',
        sa.Column('profit_loss_percentage', sa.Float(), nullable=True)
    )
    op.add_column('positions',
        sa.Column('lot_size', sa.Float(), nullable=True)
    )
    
    # Add balance tracking columns to account_balance
    op.add_column('account_balance',
        sa.Column('total_profit_loss', sa.Float(), nullable=True, server_default='0.0')
    )
    op.add_column('account_balance',
        sa.Column('total_trades', sa.Integer(), nullable=True, server_default='0')
    )
    op.add_column('account_balance',
        sa.Column('winning_trades', sa.Integer(), nullable=True, server_default='0')
    )
    op.add_column('account_balance',
        sa.Column('losing_trades', sa.Integer(), nullable=True, server_default='0')
    )
    op.add_column('account_balance',
        sa.Column('win_rate', sa.Float(), nullable=True, server_default='0.0')
    )


def downgrade() -> None:
    # Drop profit/loss tracking columns from positions
    op.drop_column('positions', 'profit_loss')
    op.drop_column('positions', 'profit_loss_percentage')
    op.drop_column('positions', 'lot_size')
    
    # Drop balance tracking columns from account_balance
    op.drop_column('account_balance', 'total_profit_loss')
    op.drop_column('account_balance', 'total_trades')
    op.drop_column('account_balance', 'winning_trades')
    op.drop_column('account_balance', 'losing_trades')
    op.drop_column('account_balance', 'win_rate') 