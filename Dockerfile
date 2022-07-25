FROM python:3.8

WORKDIR /Python

COPY ./Python /Python

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt update -y
RUN apt upgrade -y
RUN apt-get install -y libgl1-mesa-dev
RUN apt install python3-dev python3-pip python3-setuptools -y
RUN python3 -m pip install --upgrade pip setuptools

EXPOSE 5000

CMD ["python", "server.py"]