# Python image to use.
FROM python:3.12-slim
# Set the working directory to /app
WORKDIR /app
# copy the requirements file used for dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
# Copy the rest of the working directory contents into the container at /app
COPY . .
# Run app.py when the container launches
ENTRYPOINT ["mesop","main.py","--port=8080","--server.address=0.0.0.0"]
