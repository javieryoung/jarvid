FROM public.ecr.aws/lambda/python:3.8

COPY app/ ./

CMD ["lambda_handler.lambda_handler"]
