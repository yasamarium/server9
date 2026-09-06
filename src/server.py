import json
import time
import logging
import asyncio
from typing import List, Optional, Union, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator

from src.config import settings
from src.auth import verify_api_key, AuthenticationError
from src.model import model_manager

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llmserver")

SERVER_START_TIME = time.time()


# -----------------------------------------------------------------------------
# Lifespan Handler (Startup & Graceful Shutdown)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up yasamarium/llmserver...")
    logger.info(f"Target model: {settings.MODEL_NAME}")
    logger.info(f"Context size: {settings.CONTEXT_SIZE} | CPU Threads: {settings.THREADS}")
    
    # Initialize / Load model
    try:
        model_manager.load_model()
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        # Let the server still start so /health can report status or wait for retry

    yield

    logger.info("Gracefully shutting down yasamarium/llmserver...")


# -----------------------------------------------------------------------------
# FastAPI App Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title="yasamarium/llmserver",
    description="OpenAI-compatible HTTP API running Qwen3 4B on GitHub Actions / CPU",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for external access (e.g. Vercel deployed frontend yasamarium/llm)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request Middleware for Clean Logging
# -----------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    # Never log Authorization header or body content to maintain strict privacy
    logger.info(f"Incoming request: {request.method} {request.url.path} from {client_ip}")

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Completed {request.method} {request.url.path} - Status {response.status_code} ({duration_ms:.1f}ms)"
        )
        return response
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Error handling {request.method} {request.url.path} after {duration_ms:.1f}ms: {exc}")
        raise


# -----------------------------------------------------------------------------
# Exception Handlers
# -----------------------------------------------------------------------------
@app.exception_handler(AuthenticationError)
async def auth_error_handler(request: Request, exc: AuthenticationError):
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error for {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": {
                "message": f"Malformed or invalid request data: {exc.errors()[0].get('msg', 'Validation error')}",
                "type": "invalid_request_error",
                "param": str(exc.errors()[0].get("loc", [])),
                "code": "bad_request",
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Ensure detail is formatted as OpenAI error format if not already
    detail = exc.detail
    if not isinstance(detail, dict) or "error" not in detail:
        detail = {
            "error": {
                "message": str(detail),
                "type": "invalid_request_error" if exc.status_code < 500 else "server_error",
                "code": exc.status_code,
            }
        }
    return JSONResponse(status_code=exc.status_code, content=detail)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": f"Internal inference server error: {str(exc)}",
                "type": "server_error",
                "param": None,
                "code": "internal_error",
            }
        },
    )


# -----------------------------------------------------------------------------
# Schemas (OpenAI-compatible)
# -----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender: system, user, or assistant")
    content: str = Field(..., description="Message content")

    @field_validator("role")
    def validate_role(cls, v):
        allowed = {"system", "user", "assistant", "tool"}
        if v.lower() not in allowed:
            raise ValueError(f"Role must be one of: {allowed}")
        return v.lower()


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=settings.MODEL_NAME, description="Model identifier")
    messages: List[ChatMessage] = Field(..., min_length=1, description="Conversation messages")
    temperature: Optional[float] = Field(default=settings.TEMPERATURE, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=settings.MAX_TOKENS, ge=1, le=8192)
    stream: Optional[bool] = Field(default=False, description="Whether to stream response chunks")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stop: Optional[Union[str, List[str]]] = None


class CompletionRequest(BaseModel):
    model: Optional[str] = Field(default=settings.MODEL_NAME)
    prompt: Union[str, List[str]] = Field(..., description="Text prompt for completion")
    temperature: Optional[float] = Field(default=settings.TEMPERATURE, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=settings.MAX_TOKENS, ge=1, le=8192)
    stream: Optional[bool] = Field(default=False)
    top_p: Optional[float] = Field(default=1.0)
    stop: Optional[Union[str, List[str]]] = None


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/", summary="Basic Server Information")
async def root():
    """Return basic server metadata, status, and uptime."""
    return {
        "name": "yasamarium/llmserver",
        "version": "1.0.0",
        "status": "online",
        "model": settings.MODEL_NAME,
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 2),
        "model_loaded": model_manager.is_loaded,
        "endpoints": [
            "GET /health",
            "GET /v1/models",
            "POST /v1/chat/completions",
            "POST /v1/completions",
        ],
    }


@app.get("/health", summary="Health Check")
async def health():
    """
    Health check endpoint returning exact format required:
    {
      "status": "ok",
      "model": "qwen3-4b"
    }
    """
    return {
        "status": "ok",
        "model": settings.MODEL_NAME,
    }


@app.get("/v1/models", summary="List Models")
async def list_models(dependencies: None = Depends(verify_api_key)):
    """OpenAI-compatible models listing."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.MODEL_NAME,
                "object": "model",
                "created": int(SERVER_START_TIME),
                "owned_by": "yasamarium",
            }
        ],
    }


@app.post("/v1/chat/completions", summary="Create Chat Completion")
async def chat_completions(
    request: ChatCompletionRequest,
    dependencies: None = Depends(verify_api_key),
):
    """
    OpenAI-compatible chat completion endpoint.
    Supports system, user, assistant messages, conversation history, and streaming.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "message": "Model is not yet loaded or is currently restarting.",
                    "type": "server_error",
                    "code": "model_not_ready",
                }
            },
        )

    logger.info(
        f"Inference start: Chat completion with {len(request.messages)} messages, "
        f"max_tokens={request.max_tokens}, stream={request.stream}"
    )

    formatted_messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if not request.stream:
        start_t = time.time()
        # Offload blocking CPU inference to threadpool so event loop remains responsive
        result = await asyncio.to_thread(
            model_manager.chat_completion,
            messages=formatted_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
            stop=request.stop,
        )
        latency = (time.time() - start_t) * 1000
        logger.info(f"Inference complete in {latency:.1f}ms")
        return result

    # Streaming mode via SSE (Server-Sent Events)
    async def sse_stream():
        def generate_chunks():
            return model_manager.chat_completion(
                messages=formatted_messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                stop=request.stop,
            )

        # Retrieve the generator in thread
        chunks = await asyncio.to_thread(generate_chunks)

        for chunk in chunks:
            payload = json.dumps(chunk)
            yield f"data: {payload}\n\n"
            # Small yield to event loop
            await asyncio.sleep(0.001)

        yield "data: [DONE]\n\n"
        logger.info("Streaming inference complete")

    return StreamingResponse(sse_stream(), media_type="text/event-stream")


@app.post("/v1/completions", summary="Create Completion")
async def completions(
    request: CompletionRequest,
    dependencies: None = Depends(verify_api_key),
):
    """
    OpenAI-compatible text completion endpoint.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "message": "Model is not yet loaded or is currently restarting.",
                    "type": "server_error",
                    "code": "model_not_ready",
                }
            },
        )

    prompt_text = request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)

    logger.info(
        f"Inference start: Text completion, prompt_len={len(prompt_text)}, "
        f"max_tokens={request.max_tokens}, stream={request.stream}"
    )

    if not request.stream:
        start_t = time.time()
        result = await asyncio.to_thread(
            model_manager.completion,
            prompt=prompt_text,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
            stop=request.stop,
        )
        latency = (time.time() - start_t) * 1000
        logger.info(f"Inference complete in {latency:.1f}ms")
        return result

    # Streaming mode
    async def sse_stream():
        def generate_chunks():
            return model_manager.completion(
                prompt=prompt_text,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                stop=request.stop,
            )

        chunks = await asyncio.to_thread(generate_chunks)

        for chunk in chunks:
            payload = json.dumps(chunk)
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.001)

        yield "data: [DONE]\n\n"
        logger.info("Streaming completion complete")

    return StreamingResponse(sse_stream(), media_type="text/event-stream")
