FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    flask==3.0.0 \
    flask-socketio==5.3.6 \
    python-socketio==5.10.0 \
    docker==7.0.0 \
    eventlet==0.35.1

COPY app.py .
COPY templates/ templates/

EXPOSE 5000

CMD ["python", "app.py"]
