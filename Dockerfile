FROM python:3.11-slim

WORKDIR /app

# Install Redis
RUN apt-get update && apt-get install -y redis-server

# Pre-copy only requirements to leverage layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

EXPOSE 8000
EXPOSE 2222


RUN apt-get update && apt-get install -y openssh-server \
    && mkdir /var/run/sshd
    
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
RUN echo 'Port 2222' >> /etc/ssh/sshd_config
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
RUN echo 'root:root' | chpasswd


COPY startup.sh /startup.sh
RUN sed -i 's/\r$//' /startup.sh && chmod +x /startup.sh

# Create Redis configuration directory
RUN mkdir -p /etc/redis
# Create a basic Redis configuration file
RUN echo "bind 127.0.0.1" > /etc/redis/redis.conf
RUN echo "port 6379" >> /etc/redis/redis.conf
RUN echo "daemonize yes" >> /etc/redis/redis.conf
RUN echo "supervised auto" >> /etc/redis/redis.conf
RUN echo "pidfile /var/run/redis/redis-server.pid" >> /etc/redis/redis.conf
RUN echo "dir /var/lib/redis" >> /etc/redis/redis.conf
RUN echo "loglevel notice" >> /etc/redis/redis.conf
RUN echo "logfile /var/log/redis/redis-server.log" >> /etc/redis/redis.conf
RUN mkdir -p /var/lib/redis /var/log/redis /var/run/redis
RUN chmod 777 /var/lib/redis /var/log/redis /var/run/redis

# Update startup script to start Redis before the application
CMD ["/startup.sh", "gunicorn", "main:app", "--bind", "0.0.0.0:8000"]
