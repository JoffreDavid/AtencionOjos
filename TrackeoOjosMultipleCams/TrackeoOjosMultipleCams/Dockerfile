FROM python:3.10-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*


COPY app/ /app/

# Aumentar timeout y usar múltiples intentos
RUN pip install --no-cache-dir --timeout=1000 --retries=3 -r requirements.txt

CMD ["python", "capture.py"]
