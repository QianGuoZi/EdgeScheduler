FROM python:3.8-slim
RUN apt update && apt install -y \
    iproute2 \
    net-tools \
    iperf3 \
    iputils-ping \
    nano \
    curl \
    pkg-config \
    libhdf5-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY dml_req.txt /home
WORKDIR /home
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow==2.13.0
RUN pip3 install -r dml_req.txt
CMD ["/bin/bash"]
