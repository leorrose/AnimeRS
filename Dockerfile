# set base image (host OS)
FROM python:3.9.1

# set the working directory in the container
WORKDIR /code

# copy files to the working directory
COPY . .

# install dependencies
RUN pip install -r requirements.txt

# command to run on container start
CMD [ "streamlit", "run", "./app/anime_app.py"]