FROM public.ecr.aws/lambda/python:3.8

COPY lambda_handler.py ./

CMD ["lambda_handler.lambda_handler"]
