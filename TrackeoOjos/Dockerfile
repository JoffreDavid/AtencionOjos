FROM python:3.10-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*


COPY app/ /app/


RUN pip install --no-cache-dir \
    opencv-python \
    mediapipe \
    flask \
    streamlit \
    pandas

CMD ["python", "capture.py"]
