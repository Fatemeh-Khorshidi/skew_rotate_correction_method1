# Use a Python base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy your main.py file into the container
ADD main.py .

# Install system packages, including libGL.so.1
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the contents of your local directory into the container
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Set environment variable to prevent tkinter from trying to open a display
ENV DISPLAY=:0

# Define the command to run your application
CMD ["python", "main.py"]
