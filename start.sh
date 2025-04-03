#!/bin/bash
set -e

# Check if database exists
if ! psql -h postgres -U hummingbird -d hummingbird -c "\q" 2>/dev/null; then
    echo "Initializing database..."
    python /app/init_db.py
else
    echo "Database already exists, skipping initialization"
fi

cd /app/src/interface/dashboard
exec python server.py 