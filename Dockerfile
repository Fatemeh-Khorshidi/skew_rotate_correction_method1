FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy your main.py file into the container
ADD app.py .

# Install system packages, including libGL.so.1
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the contents of your local directory into the container
COPY . /app

COPY ./samples/test /app/samples/test
COPY rotate_model.h5  app/rotate_model.h5

# Install Python dependencies
RUN pip install --default-timeout=10000 -r requirements.txt

# Set environment variable to prevent tkinter from trying to open a display
ENV DISPLAY=:0

# Define the command to run your application
CMD ["python", "app.py"]

