def get_closed_positions(self, start_date=None, end_date=None):
        """Get closed positions within a date range"""
        query = """
            SELECT 
                p.symbol,
                p.position_type,
                p.entry_price,
                p.exit_price,
                p.realized_pnl,
                p.closed_reason,
                p.opened_at,
                p.closed_at
            FROM positions p
            WHERE p.status = 'closed'
        """
        params = []
        
        if start_date:
            query += " AND p.closed_at >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND p.closed_at < ?"
            params.append(end_date)
            
        query += " ORDER BY p.closed_at DESC"
        
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            positions = cursor.fetchall()
            return [{
                'symbol': pos[0],
                'position_type': pos[1],
                'entry_price': pos[2],
                'exit_price': pos[3],
                'realized_pnl': pos[4],
                'closed_reason': pos[5],
                'opened_at': pos[6],
                'closed_at': pos[7]
            } for pos in positions]
        except Exception as e:
            logger.error(f"Error getting closed positions: {str(e)}")
            return [] 