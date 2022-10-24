FROM quay.io/fenicsproject/stable:current
RUN apt-get update && apt-get upgrade && apt-get autoremove
COPY . /home/fenics/multirat
