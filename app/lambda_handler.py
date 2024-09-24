import json
import os
import urllib3
import boto3
from prompt_handler import prompt_handler

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('events')
messages_table = dynamodb.Table('messages')


SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_BOT_ID = os.getenv("SLACK_BOT_ID")
http = urllib3.PoolManager()

def lambda_handler(event, context):
    print(f"Event received: {json.dumps(event)}")

    body = json.loads(event['body'])
    print(f"Body: {json.dumps(body)}")

    if 'challenge' in body:
        return {
            'statusCode': 200,
            'body': json.dumps({'challenge': body['challenge']})
        }

    if 'event' in body:
        channel = body['event']['channel']
        user = body['event']['user']
        message = body['event']['text']
        message_id = body['event']['client_msg_id']

        if user != SLACK_BOT_ID:
            try:
                response = table.get_item(Key={'message_id': message_id})
            except Exception as e:
                print(e)
                return {
                    'statusCode': 500,
                    'body': json.dumps({'error': 'Failed to check message status'})
                }

            if 'Item' in response:
                print(f"Message {message_id} already processed.")
                return {
                    'statusCode': 200,
                    'body': json.dumps({'message': 'Message already processed.'})
                }
          
            table.put_item(Item={'message_id': message_id})
            conversation_history = get_conversation_history(user)
            response_message = prompt_handler(message, conversation_history)
            
            # store the messages in the table
            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": response_message})
            save_conversation_history(user, conversation_history)

            send_message(channel, response_message)
          

    return {
        'statusCode': 200,
        'body': json.dumps('OK')
    }


def get_conversation_history(user_id):
    try:
        response = messages_table.get_item(Key={'user_id': user_id})
        return response.get('Item', {}).get('conversation_history', [])
    except Exception as e:
        print(f"Error fetching conversation history: {e}")
        return []


def save_conversation_history(user_id, conversation_history):
    conversation_history = conversation_history[-20:]
    try:
        messages_table.put_item(
            Item={
                'user_id': user_id,
                'conversation_history': conversation_history
            }
        )
    except Exception as e:
        print(f"Error saving conversation history: {e}")


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
