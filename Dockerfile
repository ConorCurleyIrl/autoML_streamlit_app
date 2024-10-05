#Steps for Deployemnt of Streamlit App using Docker
#Create a Dockerfile: Write a Dockerfile to define how to build your Docker image.
#Create a requirements.txt File: List all your dependencies in a requirements.txt file.
#Build the Docker Image: Use docker build to create the Docker image.
#Run the Docker Container: Use docker run to start a container from the image.
#Deploy to a Cloud Service: Deploy your Docker container to a cloud service like Heroku.


# Use the official Python image from the Docker Hub
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit will run on - default streamlit port is 8501
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=localhost:"]

#further commands to build and run the docker image
#docker build -t easyml . 
#docker run -p 8501:8501 easyml