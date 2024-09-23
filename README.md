
# J.A.R.V.I.D.

### Login locally to AWS:
```
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 101344062716.dkr.ecr.us-east-2.amazonaws.com
```

### Building commands:
```
docker build -t jarvid-build:1.15 .
docker tag jarvid-build:1.15 101344062716.dkr.ecr.us-east-2.amazonaws.com/jarvid:1.15
docker push 101344062716.dkr.ecr.us-east-2.amazonaws.com/jarvid:1.15
```

### Update Lambda function:
```
aws lambda update-function-code \
    --function-name jarvid \
    --image-uri 101344062716.dkr.ecr.us-east-2.amazonaws.com/jarvid:1.15
```


### Deploy image locally:
```
docker run -p 5000:8080 101344062716.dkr.ecr.us-east-2.amazonaws.com/jarvid:1.15
