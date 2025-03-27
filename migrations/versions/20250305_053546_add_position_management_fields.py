"""Add position management fields

Revision ID: add_position_management_fields
Revises: 6b666da9f6ed
Create Date: 2025-03-05 05:35:46.623635+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_position_management_fields'
down_revision = '6b666da9f6ed'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns to positions table
    op.add_column('positions', sa.Column('risk_reward_ratio', sa.Float(), nullable=True, server_default='0.0'))
    op.add_column('positions', sa.Column('last_analysis_time', sa.DateTime(), nullable=True))
    op.add_column('positions', sa.Column('position_strength', sa.Float(), nullable=True, server_default='0.0'))
    op.add_column('positions', sa.Column('last_adjustment_reason', sa.String(), nullable=True))
    op.add_column('positions', sa.Column('adjustment_history', postgresql.JSON(), nullable=True, server_default='[]'))
    op.add_column('positions', sa.Column('model_confidence', sa.Float(), nullable=True, server_default='0.0'))
    op.add_column('positions', sa.Column('analysis_reasoning', sa.String(), nullable=True))


def downgrade() -> None:
    # Remove columns from positions table
    op.drop_column('positions', 'risk_reward_ratio')
    op.drop_column('positions', 'last_analysis_time')
    op.drop_column('positions', 'position_strength')
    op.drop_column('positions', 'last_adjustment_reason')
    op.drop_column('positions', 'adjustment_history')
    op.drop_column('positions', 'model_confidence')
    op.drop_column('positions', 'analysis_reasoning') 