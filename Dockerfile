# Use an official Python runtime as a parent image
FROM python:3.13.2-slim

RUN apt-get update && apt-get install libexpat1 -y

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the Hackathon Data code into the container
COPY Datasets_Hackathon ./Datasets_Hackathon
COPY files_translation ./files_translation

COPY config.py .
COPY app.py .

EXPOSE 8050

CMD ["python", "app.py"]
