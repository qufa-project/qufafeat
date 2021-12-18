FROM ubuntu:20.04
LABEL maintainer cezanne@codemayo.com

RUN mkdir /qufafeat
WORKDIR /qufafeat
COPY requirements.txt heart_risk_5k.csv mkfeat_heart_risk_5k.json getimpt_heart_risk_5k.json \
    getimpt_heart_risk_5k_mkfeat.json featuretools-0.23.3.tar.gz /qufafeat/

RUN apt-get update && apt-get install -y --no-install-recommends \
    	    python3 python3-pip && \
    pip3 install -r requirements.txt && \
    (tar xzvf featuretools-0.23.3.tar.gz && cd featuretools-0.23.3 && python3 setup.py install) && \
    rm -rf /var/lib/apt/lists/* \
        requirements.txt featuretools-0.23.3 featuretools-0.23.3.tar.gz

CMD /bin/bash
