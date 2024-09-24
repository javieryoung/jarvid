import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from app.prompt_handler import generate_response, gpt_completion, generate_user_embeddings, find_similar_experts, create_profile_prompt, classify_user_intent_with_gpt, build_general_prompt

# Mocking the OpenAI Client and FAISS components

@pytest.fixture
def mock_openai_client():
    with patch('app.prompt_handler.client') as mock_client:
        yield mock_client

@pytest.fixture
def mock_faiss():
    with patch('app.prompt_handler.faiss') as mock_faiss:
        yield mock_faiss

def test_generate_response_general_inquiry(mock_openai_client):
    mock_openai_client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="This is a general inquiry response"))])
    
    response = generate_response("What is Python?")
    assert response == "This is a general inquiry response"

def test_generate_response_search_expert(mock_openai_client, mock_faiss):
    mock_openai_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="search expert"))]),  # For intent classification
        MagicMock(choices=[MagicMock(message=MagicMock(content="This is an expert response"))])  # For expert search
    ]

    # Simulate expert search logic
    with patch('app.prompt_handler.handle_expert_search', return_value="Expert search system prompt"):
        response = generate_response("I need an expert on AI.")
        assert response == "This is an expert response"


def test_create_profile_prompt():
    profile_data = pd.DataFrame({
        'Name': ['John Doe'],
        'Partner': ['TechCorp'],
        'Industry': ['Software'],
        'Technologies': ['Python, AI'],
        'Processed_CV': ['10 years of AI experience']
    })

    profile_string = create_profile_prompt(profile_data.iloc[0])
    assert "John Doe" in profile_string
    assert "TechCorp" in profile_string
    assert "10 years of AI experience" in profile_string

def test_classify_user_intent_with_gpt(mock_openai_client):
    mock_openai_client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="search expert"))])

    intent = classify_user_intent_with_gpt("I need help with AI")
    assert intent == "search expert"

def test_build_general_prompt():
    prompt = build_general_prompt("What is Python?")
    assert "What is Python?" in prompt
    assert "You are a helpful assistant" in prompt
