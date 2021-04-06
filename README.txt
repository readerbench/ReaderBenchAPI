# ReaderBench Python API

This document presents the steps required to set up the project and presents its capabilities.


## Docker Setup

Assuming that the docker is available on the machine, after cloning the project navigate to its folder and run:
    > docker-compose up

This will begin building the container, and it will take a while. There might appear some incompatibility issues
between libraries, but the project should work regardless.
After the installation is finished a python server should start. If this point is reached, we can exit the 
container by pressing CTRL+C and restart it as a daemon with:
    > docker-compose up -d

To stop the container use:
    > docker-compose down


## API endpoints

The server runs on the 6006 port. In order to see the list of available endpoints the "rb_api_server.py" file can
be inspected.

### Sentiment endpoint

In order to access the sentiment endpoint a POST request has to be made to the following link
(server_address is the link where the container runs, for example "localhost"):
    server_address:6006/api/v1/sentiment

The body MUST be in a JSON format and contains three fields: text, lang and model.
    - text: the input text
    - lang: the language of the text (for Romanian is "RO")
    - model: the model to be used (for sentiment only the "base" model is available). This param is optional

An example of the JSON:
    {
        "text": "Iubesc acest produs. Îl folosesc în fiecare zi!",
        "lang": "RO",
        "model": "base"
    }

The response is similar to the one presented below:
    {
        "data": {
            "prediction": 0.9813715934753418
        },
        "errorMsg": "",
        "success": true
    }

The prediction represents the sentiment score of the text that can be between [0, 1]. 1 - as positive as a text it can be; 0 - as negative as a text it can be.

Note: The requests might take a while in the beginning due to the requirements of downloading the models and loading them into memory.
After the first call, the time should improve.
Also note that having a GPU with CUDA can drastically improve performance rather than using only CPU.
