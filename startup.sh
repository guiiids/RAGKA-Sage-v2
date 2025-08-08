#!/usr/bin/env bash
# Start the SSH daemon
/usr/sbin/sshd

# Start Redis server
echo "Starting Redis server..."
redis-server /etc/redis/redis.conf
echo "Redis server started"

# Wait a moment for Redis to fully start
sleep 2

# Check if Redis is running
redis-cli ping > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "Redis is running and responding to pings"
else
    echo "WARNING: Redis may not be running properly"
fi

# Execute the main container command
exec "$@"
