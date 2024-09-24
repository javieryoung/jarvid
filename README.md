
# J.A.R.V.I.D.

## Folder structure
The project is divided in two main folders: app and experimentation.

### app folder
Here we store the code uploaded to AWS Lambda. It performs the following steps:
- Receives Slack events
- Checks if is a request for assistance (using GPT-4o)
- Builds an embedding with the request, if needed
- Looks into the existing embeddings for the k nearest neighbors
- Connects to OpenAI API to generate a response
- Send the message back to the user

### experimentation folder
Here is the needed code to build the embeddings of the current employees. There is a main
jupyter notebook file in charge of doing that, step by step, that does the following:
- Processes a folder  where we store the CVs of the people that works in the company (using GPT-4o)
- Processes an excel file that has informatino about the people that works in the company and the partners they work for
- Merges those two dataframes into one
- Builds embeddings from that merge
- Normalizes the embeddings
- Stores the embeddings into a local file 

## Useful commands
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
