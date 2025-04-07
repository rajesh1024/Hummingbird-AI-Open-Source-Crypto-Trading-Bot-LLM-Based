from datetime import datetime
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from src.data.models import Position, PositionStatus, PositionType, AccountBalance
import logging

class PositionManager:
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger(__name__)
        self.config = {}  # Will be set by the main application
        self.min_lot_size = 0.01  # Minimum lot size (1% of account)
        self.max_lot_size = 0.05  # Maximum lot size (5% of account)
        self.base_lot_size = 0.02  # Base lot size (2% of account)
    
    def create_position(self,
                       symbol: str,
                       position_type: PositionType,
                       entry_price: float,
                       stop_loss: float,
                       take_profit: float,
                       size: float,
                       timeframe: str,
                       confidence: float = 0.5) -> Optional[Position]:
        """Create a new trading position"""
        try:
            # Calculate and validate risk:reward ratio
            risk_reward_ratio = self._calculate_risk_reward_ratio(
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_type=position_type
            )
            
            # Get minimum risk:reward from config
            min_risk_reward = self.config.get('risk_management', {}).get('scalping', {}).get('min_risk_reward', 1.0)
            
            if risk_reward_ratio < min_risk_reward:
                self.logger.warning(f"Risk:reward ratio {risk_reward_ratio} is below minimum threshold of {min_risk_reward}")
                return None

            # Get account balance
            account_balance = self.db.query(AccountBalance).first()
            if not account_balance:
                self.logger.warning("No account balance found, using minimum lot size")
                lot_size = self.min_lot_size
            else:
                # Calculate lot size based on confidence and account balance
                lot_size = self.calculate_lot_size(confidence, account_balance.balance)
            
            position = Position(
                symbol=symbol,
                position_type=position_type,
                status=PositionStatus.PENDING,
                entry_price=entry_price,
                current_price=entry_price,  # Use entry price as current price initially
                stop_loss=stop_loss,
                take_profit=take_profit,
                size=lot_size,  # Use calculated lot size
                timeframe=timeframe,
                risk_reward_ratio=risk_reward_ratio,
                model_confidence=confidence,
                account_balance_id=account_balance.id if account_balance else None
            )
            self.db.add(position)
            self.db.commit()
            self.db.refresh(position)
            self.logger.info(f"Created new position with risk:reward ratio of {risk_reward_ratio} and lot size of {lot_size}")
            return position
            
        except Exception as e:
            self.logger.error(f"Error creating position: {str(e)}")
            self.db.rollback()
            return None
    
    def get_active_positions(self) -> List[Position]:
        """Get all active positions"""
        return self.db.query(Position).filter(
            Position.status.in_([PositionStatus.OPEN, PositionStatus.PENDING])
        ).all()
    
    def update_position_status(self,
                             position_id: int,
                             current_price: float) -> Optional[Position]:
        """Update position status based on current price"""
        try:
            position = self.db.query(Position).filter(Position.id == position_id).first()
            if not position:
                return None
            
            # Update current price and PnL
            position.current_price = current_price
            self.update_position_profit_loss(position)
            
            # Check for stop loss or take profit hits
            if position.position_type == PositionType.LONG:
                if current_price <= position.stop_loss:
                    # Close position
                    position.status = PositionStatus.CLOSED
                    position.closed_at = datetime.utcnow()
                    position.closed_reason = "SL"
                    self.logger.info(f"Closing LONG position {position_id} - Stop loss reached at {current_price}")
                    self._update_account_balance(position)
                    self.db.commit()
                    return position
                elif current_price >= position.take_profit:
                    # Close position
                    position.status = PositionStatus.CLOSED
                    position.closed_at = datetime.utcnow()
                    position.closed_reason = "TP"
                    self.logger.info(f"Closing LONG position {position_id} - Take profit reached at {current_price}")
                    self._update_account_balance(position)
                    self.db.commit()
                    return position
            else:  # SHORT position
                if current_price >= position.stop_loss:
                    # Close position
                    position.status = PositionStatus.CLOSED
                    position.closed_at = datetime.utcnow()
                    position.closed_reason = "SL"
                    self.logger.info(f"Closing SHORT position {position_id} - Stop loss reached at {current_price}")
                    self._update_account_balance(position)
                    self.db.commit()
                    return position
                elif current_price <= position.take_profit:
                    # Close position
                    position.status = PositionStatus.CLOSED
                    position.closed_at = datetime.utcnow()
                    position.closed_reason = "TP"
                    self.logger.info(f"Closing SHORT position {position_id} - Take profit reached at {current_price}")
                    self._update_account_balance(position)
                    self.db.commit()
                    return position
            
            # If position is still open, update its status
            if position.status == PositionStatus.PENDING:
                position.status = PositionStatus.OPEN
            
            self.db.commit()
            self.db.refresh(position)
            return position
            
        except Exception as e:
            self.logger.error(f"Error updating position status: {str(e)}")
            self.db.rollback()
            return None
    
    def close_position(self, position_id: int) -> bool:
        """Close a position"""
        position = self.db.query(Position).filter(Position.id == position_id).first()
        if not position:
            return False
        
        position.status = PositionStatus.CLOSED
        position.updated_at = datetime.utcnow()
        
        self.db.commit()
        return True
    
    def _calculate_pnl(self, position: Position) -> float:
        """Calculate position PnL"""
        min_risk_reward = self.config.get('risk_management', {}).get('scalping', {}).get('min_risk_reward', 1.0)
        _OpeningBal =  self.config.get('position',{}).get('min_balance',100)
        _TradeValue=30
        _PostionSize = 0.01
        
        if position.position_type == PositionType.LONG:
            return (position.current_price - position.entry_price) * position.size
        else:  # SHORT
            return (position.entry_price - position.current_price) * position.size
    
    def get_position_history(self, position_id: int) -> Dict:
        """Get complete position history"""
        return self.db.get_position_history(position_id)
    
    def adjust_stop_loss(self,
                        position_id: int,
                        new_stop_loss: float) -> Optional[Position]:
        """Adjust stop loss for a position"""
        try:
            position = self.db.query(Position).filter(Position.id == position_id).first()
            if not position:
                self.logger.error(f"Position {position_id} not found")
                return None
            
            # Validate new stop loss
            if position.position_type == PositionType.LONG:
                if new_stop_loss >= position.current_price:
                    self.logger.warning(f"Invalid stop loss for LONG position: {new_stop_loss} >= {position.current_price}")
                    return None
            else:  # SHORT
                if new_stop_loss <= position.current_price:
                    self.logger.warning(f"Invalid stop loss for SHORT position: {new_stop_loss} <= {position.current_price}")
                    return None
            
            # Update stop loss
            position.stop_loss = new_stop_loss
            position.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.logger.info(f"Successfully updated stop loss for position {position_id} to {new_stop_loss}")
            return position
            
        except Exception as e:
            self.logger.error(f"Error adjusting stop loss: {str(e)}")
            self.db.rollback()
            return None
    
    def adjust_take_profit(self,
                          position_id: int,
                          new_take_profit: float) -> Optional[Position]:
        """Adjust take profit for a position"""
        try:
            position = self.db.query(Position).filter(Position.id == position_id).first()
            if not position:
                self.logger.error(f"Position {position_id} not found")
                return None
            
            # Validate new take profit
            if position.position_type == PositionType.LONG:
                if new_take_profit <= position.current_price:
                    self.logger.warning(f"Invalid take profit for LONG position: {new_take_profit} <= {position.current_price}")
                    return None
            else:  # SHORT
                if new_take_profit >= position.current_price:
                    self.logger.warning(f"Invalid take profit for SHORT position: {new_take_profit} >= {position.current_price}")
                    return None
            
            # Update take profit
            position.take_profit = new_take_profit
            position.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.logger.info(f"Successfully updated take profit for position {position_id} to {new_take_profit}")
            return position
            
        except Exception as e:
            self.logger.error(f"Error adjusting take profit: {str(e)}")
            self.db.rollback()
            return None
    
    def update_position_with_analysis(self,
                                    position_id: int,
                                    current_price: float,
                                    signal: Dict = None,
                                    market_structure: Dict = None) -> Optional[Position]:
        """Update position with current price and check for SL/TP hits"""
        try:
            position = self.db.query(Position).filter_by(id=position_id).first()
            if not position:
                self.logger.error(f"Position {position_id} not found")
                return None

            # If position is already closed, don't update it
            if position.status == PositionStatus.CLOSED:
                self.logger.info(f"Position {position_id} is already closed, skipping update")
                return position

            # Update current price and PnL
            position.current_price = current_price
            position.pnl = self._calculate_pnl(position)

            # Check for stop loss or take profit hits first
            if position.position_type == PositionType.LONG:
                if current_price <= position.stop_loss:
                    # Close position
                    position.status = PositionStatus.CLOSED
                    position.closed_at = datetime.utcnow()
                    position.closed_reason = "SL"
                    self.logger.info(f"Closing LONG position {position_id} - Stop loss reached at {current_price}")
                    self.db.commit()
                    return position
                elif current_price >= position.take_profit:
                    # Close position
                    position.status = PositionStatus.CLOSED
                    position.closed_at = datetime.utcnow()
                    position.closed_reason = "TP"
                    self.logger.info(f"Closing LONG position {position_id} - Take profit reached at {current_price}")
                    self.db.commit()
                    return position
            else:  # SHORT position
                if current_price >= position.stop_loss:
                    # Close position
                    position.status = PositionStatus.CLOSED
                    position.closed_at = datetime.utcnow()
                    position.closed_reason = "SL"
                    self.logger.info(f"Closing SHORT position {position_id} - Stop loss reached at {current_price}")
                    self.db.commit()
                    return position
                elif current_price <= position.take_profit:
                    # Close position
                    position.status = PositionStatus.CLOSED
                    position.closed_at = datetime.utcnow()
                    position.closed_reason = "TP"
                    self.logger.info(f"Closing SHORT position {position_id} - Take profit reached at {current_price}")
                    self.db.commit()
                    return position

            # Only update analysis data if position is still open
            if position.status != PositionStatus.CLOSED:
                # Update position with signal data if provided
                if signal:
                    position.model_confidence = signal.get('confidence', 0.0)
                    position.analysis_reasoning = signal.get('reasoning', '')
                    position.last_analysis_time = datetime.utcnow()
                    
                    # Update position strength if market structure is provided
                    if market_structure:
                        position.position_strength = self._calculate_position_strength(position, market_structure)

                # If position is still pending, update its status to open
                if position.status == PositionStatus.PENDING:
                    position.status = PositionStatus.OPEN

                self.db.commit()
                self.logger.info(f"Successfully updated position {position_id} with analysis")

            return position

        except Exception as e:
            self.logger.error(f"Error updating position with analysis: {str(e)}")
            self.db.rollback()
            return None

    def _update_trailing_levels(self, position: Position, current_price: float) -> None:
        """Update trailing stop loss and take profit levels based on profitability"""
        try:
            pnl_percentage = abs((current_price - position.entry_price) / position.entry_price)
            
            # Get trailing stop distance from config (2% default)
            trailing_stop_distance = 0.02
            
            if position.position_type == PositionType.LONG:
                # Calculate new stop loss
                new_stop = current_price * (1 - trailing_stop_distance)
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    self.logger.info(f"Updated trailing stop loss to {new_stop}")
                
                # Calculate new take profit if close to target
                if current_price >= position.take_profit * 0.95:  # Within 5% of target
                    new_target = current_price * (1 + trailing_stop_distance)
                    if new_target > position.take_profit:
                        position.take_profit = new_target
                        self.logger.info(f"Updated trailing take profit to {new_target}")
            
            else:  # SHORT position
                # Calculate new stop loss
                new_stop = current_price * (1 + trailing_stop_distance)
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    self.logger.info(f"Updated trailing stop loss to {new_stop}")
                
                # Calculate new take profit if close to target
                if current_price <= position.take_profit * 1.05:  # Within 5% of target
                    new_target = current_price * (1 - trailing_stop_distance)
                    if new_target < position.take_profit:
                        position.take_profit = new_target
                        self.logger.info(f"Updated trailing take profit to {new_target}")
            
            # Record the adjustment in history
            adjustment = {
                'timestamp': datetime.utcnow(),
                'type': 'trailing_adjustment',
                'pnl_percentage': pnl_percentage,
                'new_stop_loss': position.stop_loss,
                'new_take_profit': position.take_profit
            }
            
            if not position.adjustment_history:
                position.adjustment_history = []
            position.adjustment_history.append(adjustment)
            
        except Exception as e:
            self.logger.error(f"Error updating trailing levels: {str(e)}")

    def _calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, take_profit: float, position_type: PositionType) -> float:
        """Calculate and validate risk:reward ratio"""
        try:
            if position_type == PositionType.LONG:
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:  # SHORT
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
            
            if risk <= 0:
                self.logger.warning("Invalid risk calculation - risk must be positive")
                return 0.0
                
            ratio = reward / risk
            # Standard format is risk:reward (e.g., 1:2 means risk 1 to make 2)
            return round(ratio, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating risk:reward ratio: {str(e)}")
            return 0.0

    def validate_position(self, entry_price: float, stop_loss: float, take_profit: float, position_type: PositionType) -> bool:
        """Validate position parameters"""
        try:
            # Calculate and validate risk:reward ratio
            risk_reward_ratio = self._calculate_risk_reward_ratio(
                entry_price, stop_loss, take_profit, position_type
            )
            
            # Get minimum risk:reward from config
            min_risk_reward = self.config.get('risk_management', {}).get('scalping', {}).get('min_risk_reward', 1.0)
            
            # Validate risk:reward ratio
            if risk_reward_ratio < min_risk_reward:
                self.logger.warning(f"Risk:reward ratio {risk_reward_ratio} is below minimum threshold of {min_risk_reward}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating position: {str(e)}")
            return False

    def _calculate_position_strength(self, position: dict, market_structure: Dict = None) -> float:
        """Calculate the strength of a position based on market structure and other factors"""
        try:
            strength = 0.0
            current_price = position['current_price']
            
            # Calculate risk:reward ratio
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            if position['position_type'] == "LONG":
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
                
            risk_reward_ratio = reward / risk if risk != 0 else 0
            
            # Base strength on risk:reward ratio (up to 0.4)
            min_rr = self.config.get('risk_management', {}).get('scalping', {}).get('min_risk_reward', 1.0)
            rr_strength = min(risk_reward_ratio / min_rr, 1.0) * 0.4
            strength += rr_strength
            
            # Add strength based on PnL (up to 0.3)
            pnl = position.get('pnl', 0)
            if pnl > 0:
                pnl_strength = min(abs(pnl) / 0.02, 1.0) * 0.3  # Max strength at 2% profit
                strength += pnl_strength
            
            # Add strength based on market structure if available (up to 0.3)
            if market_structure:
                # Check if price is near key levels
                for level in market_structure.get('key_levels', []):
                    level_price = level.get('price', 0)
                    if abs(current_price - level_price) / current_price < 0.001:  # Within 0.1%
                        strength += 0.1
                
                # Check if price is in a high probability zone
                for zone in market_structure.get('high_probability_zones', []):
                    zone_price = zone.get('price', 0)
                    if abs(current_price - zone_price) / current_price < 0.001:  # Within 0.1%
                        strength += 0.1
                
                # Check for SMC patterns
                if market_structure.get('smc_patterns'):
                    strength += 0.1
            
            # Cap total strength at 1.0
            return min(strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating position strength: {str(e)}")
            return 0.0

    def calculate_lot_size(self, confidence: float, account_balance: float) -> float:
        """Calculate position size based on confidence and account balance"""
        try:
            # Get account balance
            if not account_balance:
                self.logger.warning("No account balance provided, using minimum lot size")
                return self.min_lot_size

            # Calculate base position size (2% of account)
            base_size = account_balance * self.base_lot_size

            # Adjust size based on confidence (0.0 to 1.0)
            confidence_factor = max(0.0, min(1.0, confidence))
            adjusted_size = base_size * confidence_factor

            # Apply min/max limits
            min_size = account_balance * self.min_lot_size
            max_size = account_balance * self.max_lot_size
            final_size = max(min_size, min(adjusted_size, max_size))

            # Round to 2 decimal places
            return round(final_size, 2)

        except Exception as e:
            self.logger.error(f"Error calculating lot size: {str(e)}")
            return self.min_lot_size

    def update_position_profit_loss(self, position: Position) -> None:
        """Update position profit/loss and account balance"""
        try:
            if not position.account_balance:
                self.logger.warning(f"Position {position.id} has no associated account balance")
                return

            # Calculate profit/loss
            if position.position_type == PositionType.LONG:
                profit_loss = (position.current_price - position.entry_price) * position.size
            else:  # SHORT
                profit_loss = (position.entry_price - position.current_price) * position.size

            # Calculate percentage
            profit_loss_percentage = (profit_loss / (position.entry_price * position.size)) * 100

            # Update position
            position.profit_loss = profit_loss
            position.profit_loss_percentage = profit_loss_percentage

            # If position is closed, update account balance
            if position.status == PositionStatus.CLOSED:
                self._update_account_balance(position)

        except Exception as e:
            self.logger.error(f"Error updating position profit/loss: {str(e)}")

    def _update_account_balance(self, position: Position) -> None:
        """Update account balance when a position is closed"""
        try:
            account_balance = position.account_balance
            if not account_balance:
                return

            # Update total profit/loss
            account_balance.total_profit_loss += position.profit_loss

            # Update trade counts
            account_balance.total_trades += 1
            if position.profit_loss > 0:
                account_balance.winning_trades += 1
            elif position.profit_loss < 0:
                account_balance.losing_trades += 1

            # Update win rate
            if account_balance.total_trades > 0:
                account_balance.win_rate = (account_balance.winning_trades / account_balance.total_trades) * 100

            # Update current balance
            account_balance.balance += position.profit_loss
            account_balance.last_updated = datetime.utcnow()

            self.logger.info(f"Updated account balance for position {position.id}:")
            self.logger.info(f"Profit/Loss: ${position.profit_loss:.2f}")
            self.logger.info(f"New Balance: ${account_balance.balance:.2f}")
            self.logger.info(f"Win Rate: {account_balance.win_rate:.2f}%")

        except Exception as e:
            self.logger.error(f"Error updating account balance: {str(e)}") 