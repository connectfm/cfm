#!/bin/bash

docker build -t dbproc .
docker tag dbproc:latest 293365975941.dkr.ecr.us-east-2.amazonaws.com/dbproc:latest
docker push 293365975941.dkr.ecr.us-east-2.amazonaws.com/dbproc:latest