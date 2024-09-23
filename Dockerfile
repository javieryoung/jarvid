FROM public.ecr.aws/lambda/python:3.8

COPY app/ ./

RUN python -m pip install -r requirements.txt

CMD ["lambda_handler.lambda_handler"]
