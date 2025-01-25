# Step 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . /app

# Step 4: Install any dependencies that your app requires
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Make port 5001 available to the outside world (if it's a web app)
EXPOSE 5001

# Step 6: Define the command to run your app
CMD ["python", "app.py"]