import json
import os
import pytest
from unittest.mock import patch, MagicMock
from app.lambda_handler import lambda_handler, send_message

@pytest.fixture
def mock_table():
    with patch('app.lambda_handler.table') as mock:
        yield mock

@pytest.fixture
def mock_send_message():
    with patch('app.lambda_handler.send_message') as mock:
        yield mock

@pytest.fixture
def mock_prompt_handler():
    with patch('app.lambda_handler.prompt_handler') as mock:
        yield mock

def test_lambda_handler_challenge():
    event = {
        'body': json.dumps({'challenge': 'test_challenge'})
    }
    context = {}

    response = lambda_handler(event, context)

    assert response['statusCode'] == 200
    assert json.loads(response['body']) == {'challenge': 'test_challenge'}

def test_lambda_handler_event_processed(mock_table, mock_send_message, mock_prompt_handler):
    event = {
        'body': json.dumps({
            'event': {
                'channel': 'C123456',
                'user': 'U123456',
                'text': 'Hello!',
                'client_msg_id': 'msg123'
            }
        })
    }
    context = {}

    mock_table.get_item.return_value = {}
    mock_prompt_handler.return_value = 'Response from prompt_handler'

    response = lambda_handler(event, context)

    mock_table.put_item.assert_called_once_with(Item={'message_id': 'msg123'})
    mock_send_message.assert_called_once_with('C123456', 'Response from prompt_handler')
    assert response['statusCode'] == 200
    assert json.loads(response['body']) == 'OK'

def test_lambda_handler_message_already_processed(mock_table):
    event = {
        'body': json.dumps({
            'event': {
                'channel': 'C123456',
                'user': 'U123456',
                'text': 'Hello!',
                'client_msg_id': 'msg123'
            }
        })
    }
    context = {}

    mock_table.get_item.return_value = {'Item': {'message_id': 'msg123'}}

    response = lambda_handler(event, context)

    assert response['statusCode'] == 200
    assert json.loads(response['body']) == {'message': 'Message already processed.'}

def test_lambda_handler_error_on_get_item(mock_table):
    event = {
        'body': json.dumps({
            'event': {
                'channel': 'C123456',
                'user': 'U123456',
                'text': 'Hello!',
                'client_msg_id': 'msg123'
            }
        })
    }
    context = {}

    mock_table.get_item.side_effect = Exception("DB Error")

    response = lambda_handler(event, context)

    assert response['statusCode'] == 500
    assert json.loads(response['body'])["error"] == 'Failed to check message status'

def test_send_message():
    with patch('app.lambda_handler.urllib3.PoolManager') as mock_pool:
        mock_http = MagicMock()
        mock_pool.return_value = mock_http

        channel = 'C123456'
        text = 'Hello, Slack!'
        send_message(channel, text)

        expected_url = "https://slack.com/api/chat.postMessage"
        expected_headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("SLACK_BOT_TOKEN")}'
        }
        expected_body = json.dumps({'channel': channel, 'text': text}).encode('utf-8')

        mock_http.request.assert_called_once_with('POST', expected_url, body=expected_body, headers=expected_headers)

if __name__ == '__main__':
    pytest.main()
