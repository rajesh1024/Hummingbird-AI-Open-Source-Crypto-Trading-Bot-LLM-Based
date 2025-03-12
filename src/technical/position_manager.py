from datetime import datetime
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from src.data.models import Position, PositionStatus, PositionType, ExitSignal, MarketStructureData
import logging

class PositionManager:
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger(__name__)
    
    def create_position(self,
                       symbol: str,
                       position_type: PositionType,
                       entry_price: float,
                       stop_loss: float,
                       take_profit: float,
                       size: float,
                       timeframe: str) -> Optional[Position]:
        """Create a new trading position"""
        try:
            position = Position(
                symbol=symbol,
                position_type=position_type,
                status=PositionStatus.PENDING,
                entry_price=entry_price,
                current_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                size=size,
                timeframe=timeframe
            )
            self.db.add(position)
            self.db.commit()
            self.db.refresh(position)
            return position
        except Exception as e:
            self.db.rollback()
            raise e
    
    def get_active_positions(self) -> List[Position]:
        """Get all active positions"""
        return self.db.query(Position).filter(
            Position.status.in_([PositionStatus.OPEN, PositionStatus.PENDING])
        ).all()
    
    def update_position_status(self,
                             position_id: int,
                             market_structure: MarketStructureData) -> Optional[Position]:
        """Update position status based on market structure"""
        position = self.db.query(Position).filter(Position.id == position_id).first()
        if not position:
            return None
        
        # Update current price and PnL
        position.current_price = market_structure.high  # Use high price for conservative PnL
        position.pnl = self._calculate_pnl(position)
        
        # Check for stop loss or take profit
        if position.pnl <= -position.stop_loss:
            position.status = PositionStatus.CLOSED
        elif position.pnl >= position.take_profit:
            position.status = PositionStatus.CLOSED
        
        self.db.commit()
        self.db.refresh(position)
        return position
    
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
                                  signal: Dict,
                                  market_structure: Optional[MarketStructureData] = None) -> Optional[Position]:
        """Update position with model's analysis and recommendations"""
        try:
            position = self.db.query(Position).filter(Position.id == position_id).first()
            if not position:
                self.logger.error(f"Position {position_id} not found")
                return None
            
            # Update position with model's analysis
            position.last_analysis_time = datetime.utcnow()
            position.model_confidence = signal.get('confidence', 0.0)
            position.analysis_reasoning = signal.get('reasoning', '')
            position.risk_reward_ratio = signal.get('position_management', {}).get('risk_reward_ratio', 0.0)
            
            # Store adjustment history
            adjustment = {
                'timestamp': datetime.utcnow().isoformat(),
                'action': signal.get('position_management', {}).get('action', 'MAINTAIN'),
                'stop_loss_adjustment': signal.get('position_management', {}).get('stop_loss_adjustment', ''),
                'take_profit_adjustment': signal.get('position_management', {}).get('take_profit_adjustment', ''),
                'confidence': signal.get('confidence', 0.0),
                'reasoning': signal.get('reasoning', '')
            }
            
            if position.adjustment_history is None:
                position.adjustment_history = []
            position.adjustment_history.append(adjustment)
            
            # Update last adjustment reason
            position.last_adjustment_reason = f"{adjustment['action']}: {adjustment['reasoning']}"
            
            # Calculate and update position strength
            if market_structure is not None:
                position.position_strength = self._calculate_position_strength(position, market_structure)
            else:
                # If no market structure data, use model confidence as base strength
                position.position_strength = position.model_confidence
            
            self.db.commit()
            self.logger.info(f"Successfully updated position {position_id} with analysis")
            return position
            
        except Exception as e:
            self.logger.error(f"Error updating position with analysis: {str(e)}")
            self.db.rollback()
            return None

    def _calculate_position_strength(self, position: Position, market_structure: MarketStructureData) -> float:
        """Calculate position strength based on market structure and model confidence"""
        try:
            # Base strength starts with model confidence
            strength = position.model_confidence
            
            # Adjust based on risk-reward ratio
            if position.risk_reward_ratio >= 2.0:
                strength *= 1.2
            elif position.risk_reward_ratio < 1.0:
                strength *= 0.8
            
            # Adjust based on market structure if available
            if hasattr(market_structure, 'structure_type'):
                if position.position_type == PositionType.LONG:
                    if market_structure.structure_type == MarketStructure.BULLISH:
                        strength *= 1.2
                    elif market_structure.structure_type == MarketStructure.BEARISH:
                        strength *= 0.8
                else:  # SHORT position
                    if market_structure.structure_type == MarketStructure.BEARISH:
                        strength *= 1.2
                    elif market_structure.structure_type == MarketStructure.BULLISH:
                        strength *= 0.8
            
            # Cap strength between 0 and 1
            return min(max(strength, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating position strength: {str(e)}")
            # Return model confidence as fallback
            return position.model_confidence

    def create_exit_signal(self,
                          position_id: int,
                          signal_type: str,
                          price: float,
                          reason: str,
                          confidence: float = 0.0,
                          model_analysis: str = None) -> Optional[ExitSignal]:
        """Create a new exit signal with model analysis"""
        try:
            exit_signal = ExitSignal(
                position_id=position_id,
                signal_type=signal_type,
                price=price,
                reason=reason,
                confidence=confidence,
                model_analysis=model_analysis
            )
            self.db.add(exit_signal)
            self.db.commit()
            return exit_signal
            
        except Exception as e:
            self.logger.error(f"Error creating exit signal: {str(e)}")
            self.db.rollback()
            return None 