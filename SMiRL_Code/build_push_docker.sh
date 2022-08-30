#!/bin/bash

docker build -f Dockerfile -t smirl:latest .
docker tag smirl:latest pozay/smirl:latest
docker push pozay/smirl:latest

