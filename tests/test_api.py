import os
import pytest
from fastapi.testclient import TestClient

# Ensure test configuration before importing application
os.environ["API_KEY"] = "test-secret-key"
os.environ["MOCK_MODEL"] = "true"
os.environ["MODEL_NAME"] = "qwen3-4b"

from src.config import settings
# Force settings update for test runtime
settings.API_KEY = "test-secret-key"
settings.MOCK_MODEL = True
settings.MODEL_NAME = "qwen3-4b"

from src.server import app
from src.model import model_manager

# Ensure model manager is loaded in mock mode
model_manager._is_mock = True
model_manager._is_loaded = True

client = TestClient(app)

AUTH_HEADER = {"Authorization": "Bearer test-secret-key"}


def test_root_endpoint():
    """Test GET / returns server status and metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "yasamarium/llmserver"
    assert data["model"] == "qwen3-4b"
    assert data["status"] == "online"
    assert "uptime_seconds" in data
    assert "endpoints" in data


def test_health_endpoint():
    """Test GET /health returns exactly the required JSON format."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "qwen3-4b",
    }


def test_auth_missing_header():
    """Test that requests without Authorization header return 401."""
    payload = {
        "model": "qwen3-4b",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 401
    err = response.json()
    assert "error" in err
    assert err["error"]["code"] == "invalid_api_key"


def test_auth_invalid_token():
    """Test that requests with an incorrect Bearer token return 401."""
    payload = {
        "model": "qwen3-4b",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401
    err = response.json()
    assert "error" in err
    assert err["error"]["code"] == "invalid_api_key"


def test_chat_completions_basic():
    """Test POST /v1/chat/completions with valid auth and basic message."""
    payload = {
        "model": "qwen3-4b",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADER)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "qwen3-4b"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0
    assert "usage" in data
    assert data["usage"]["total_tokens"] > 0


def test_chat_completions_conversation_history():
    """Test POST /v1/chat/completions with multi-turn conversation and system prompt."""
    payload = {
        "model": "qwen3-4b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2 + 2 equals 4."},
            {"role": "user", "content": "Multiply it by 2."},
        ],
        "temperature": 0.5,
        "max_tokens": 128,
    }
    response = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADER)
    assert response.status_code == 200
    data = response.json()
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_chat_completions_streaming():
    """Test POST /v1/chat/completions streaming responses (text/event-stream)."""
    payload = {
        "model": "qwen3-4b",
        "messages": [{"role": "user", "content": "Hello stream"}],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADER)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    text = response.text
    assert "data: " in text
    assert "data: [DONE]" in text


def test_text_completions():
    """Test standard POST /v1/completions endpoint."""
    payload = {
        "model": "qwen3-4b",
        "prompt": "Once upon a time",
        "max_tokens": 64,
        "temperature": 0.7,
    }
    response = client.post("/v1/completions", json=payload, headers=AUTH_HEADER)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"]) == 1
    assert "text" in data["choices"][0]


def test_invalid_json_body():
    """Test that malformed JSON requests return 400."""
    response = client.post(
        "/v1/chat/completions",
        content="not-valid-json",
        headers={"Content-Type": "application/json", "Authorization": "Bearer test-secret-key"},
    )
    assert response.status_code == 400
    err = response.json()
    assert "error" in err


def test_missing_messages():
    """Test that requests without messages field return 400."""
    payload = {"model": "qwen3-4b"}
    response = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADER)
    assert response.status_code == 400
    err = response.json()
    assert "error" in err


def test_invalid_role():
    """Test that unsupported message roles return 400."""
    payload = {
        "model": "qwen3-4b",
        "messages": [{"role": "superman", "content": "Hello"}],
    }
    response = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADER)
    assert response.status_code == 400


def test_list_models():
    """Test GET /v1/models returns OpenAI format model listing."""
    response = client.get("/v1/models", headers=AUTH_HEADER)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert any(m["id"] == "qwen3-4b" for m in data["data"])
