FROM python:3.7

RUN wget http://download.redis.io/redis-stable.tar.gz
RUN tar xvzf redis-stable.tar.gz
WORKDIR /redis-stable
RUN make
RUN make install

WORKDIR /airbus-ship-detection
COPY . /airbus-ship-detection

RUN pip install -r requirements.txt
RUN python setup.py install

EXPOSE 5000
CMD redis-server --daemonize yes && \
    python -m asdc.main service --model /airbus-ship-detection/model_files/asdc.h5
