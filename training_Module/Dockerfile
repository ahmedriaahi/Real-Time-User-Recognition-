# Use an official Python image
FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir tensorflow numpy opencv-python scikit-learn jupyter

# Set working directory
WORKDIR /app

# Copy training script
COPY train.py .

# Expose Jupyter Notebook port
EXPOSE 8888

# Run Jupyter on startup
CMD ["jupyter", "notebook", "--Notebook.app=''", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

