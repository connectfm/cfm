# Sources:
# https://www.hackernoon.com/how-to-deploy-aws-lambda-with-docker-containers-e51j3141
# https://docs.aws.amazon.com/lambda/latest/dg/python-image.html

# Sets the base image that includes the Lambda Runtime API
# Source: https://hub.docker.com/r/amazon/aws-lambda-python
FROM public.ecr.aws/lambda/python:3.8

# Ensure pip is up to date
RUN pip install --upgrade pip

# Reduces time for future builds via caching
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy required modules and packages
COPY recommendation/ ${LAMBDA_TASK_ROOT}

# Lambda handler
CMD ["app.handle"]