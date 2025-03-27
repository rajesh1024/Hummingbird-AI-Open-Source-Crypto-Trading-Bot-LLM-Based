"""Add closed_reason column to positions table

Revision ID: add_closed_reason
Revises: remove_market_structures
Create Date: 2025-03-21 08:00:00.000000+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_closed_reason'
down_revision = 'remove_market_structures'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add closed_reason column to positions table
    op.add_column('positions', sa.Column('closed_reason', sa.String(), nullable=True))


def downgrade() -> None:
    # Remove closed_reason column from positions table
    op.drop_column('positions', 'closed_reason') 