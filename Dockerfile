FROM ubuntu:20.04
RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev build-essential git
WORKDIR /app
ADD requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install -U numpy==1.19
RUN python3 -m spacy download en_core_web_lg
RUN python3 -m spacy download ro_core_news_lg
EXPOSE 6006
ENTRYPOINT [ "python3" ]
CMD [ "rb_api_server.py" ]
