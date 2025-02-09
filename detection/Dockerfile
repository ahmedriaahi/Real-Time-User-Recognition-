# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    opencv-python-headless \
    numpy \
    tensorflow \
    jupyter notebook
RUN apt update && apt install -y libgtk2.0-dev pkg-config
    

# Copy the detection script and model
COPY detect.py detect.py


# Expose Jupyter Notebook port
EXPOSE 8888

# Run Jupyter Notebook
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]

