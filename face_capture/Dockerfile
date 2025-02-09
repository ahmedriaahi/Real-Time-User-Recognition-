FROM python:3.9

# Install dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter
RUN pip install notebook

# Set working directory
WORKDIR /app

# Copy the script
COPY face_capture.py .

# Expose Jupyter Notebook port
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--Notebook.app=''", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]


