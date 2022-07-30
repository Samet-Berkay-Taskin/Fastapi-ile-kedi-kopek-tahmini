FROM python:3.8-slim

COPY . /kedi-1
WORKDIR /kedi-1
RUN apt-get update -y && apt-get install -y libgomp1 libgl1-mesa-dev
RUN pip install -r requirements.txt
CMD [ "uvicorn", "kedi-1:app", "--host", "0.0.0.0", "--port", "80"]


EXPOSE 8888
