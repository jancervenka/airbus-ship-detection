FROM python:3.5-alpine

WORKDIR /airbus-ship-detection
COPY . /airbus-ship-detection

RUN python setup.py install

EXPOSE 5000

CMD echo "test complete"
# CMD ./start.sh 
