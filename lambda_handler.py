import json
import os
import urllib3

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_BOT_ID = os.getenv("SLACK_BOT_ID")
http = urllib3.PoolManager()

def lambda_handler(event, context):
    print(f"Event received: {json.dumps(event)}")

    body = json.loads(event['body'])
    print(f"Body: {json.dumps(body)}")

    if 'event' in body:
        channel = body['event']['channel']
        user = body['event']['user']
        message = body['event']['text']
        if user != SLACK_BOT_ID:
            send_message(channel, f"Hola <@{user}>! {message}")
        
    
    if 'challenge' in body:
        return {
            'statusCode': 200,
            'body': json.dumps({'challenge': body['challenge']})
        }
    
    return {
        'statusCode': 200,
        'body': json.dumps('OK')
    }

def send_message(channel, text):
    url = "https://slack.com/api/chat.postMessage"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {SLACK_BOT_TOKEN}'
    }
    message = {
        'channel': channel,
        'text': text
    }
    encoded_msg = json.dumps(message).encode('utf-8')
    response = http.request('POST', url, body=encoded_msg, headers=headers)
    print(f"Slack API response: {response.data}")
