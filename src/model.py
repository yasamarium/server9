import os
import time
import uuid
import logging
from typing import List, Dict, Any, Generator, Optional
from src.config import settings

logger = logging.getLogger("llmserver.model")

class ModelManager:
    """
    Manages loading and inference for GGUF models using llama.cpp / llama-cpp-python.
    Optimized for CPU execution on standard environments.
    """

    def __init__(self):
        self._model = None
        self._is_loaded = False
        self._is_mock = settings.MOCK_MODEL

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_model(self) -> None:
        """Load the GGUF model into memory."""
        if self._is_mock:
            logger.info("MOCK_MODEL enabled: Initializing mock inference engine.")
            self._is_loaded = True
            return

        model_path = settings.MODEL_PATH
        if not os.path.exists(model_path):
            logger.warning(
                f"Model file not found at {model_path}. "
                "Will attempt mock fallback if MOCK_MODEL is allowed, otherwise inference will be unavailable."
            )
            # Check if llama_cpp is even installed
            try:
                import llama_cpp
            except ImportError:
                logger.warning("llama-cpp-python is not installed. Operating in mock fallback mode.")
                self._is_mock = True
                self._is_loaded = True
                return
            return

        try:
            from llama_cpp import Llama

            logger.info(
                f"Loading GGUF model from {model_path} with "
                f"threads={settings.THREADS}, context_size={settings.CONTEXT_SIZE}, batch_size={settings.BATCH_SIZE}..."
            )
            start_t = time.time()
            self._model = Llama(
                model_path=model_path,
                n_ctx=settings.CONTEXT_SIZE,
                n_threads=settings.THREADS,
                n_batch=settings.BATCH_SIZE,
                n_gpu_layers=0,  # Pure CPU execution
                verbose=False,
            )
            self._is_loaded = True
            logger.info(f"Model successfully loaded in {time.time() - start_t:.2f}s.")
        except ImportError:
            logger.warning("llama-cpp-python not found. Falling back to mock engine for testing.")
            self._is_mock = True
            self._is_loaded = True
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self._is_loaded = False
            raise

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """
        Execute OpenAI-compatible chat completion.
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded or unavailable.")

        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        if self._is_mock:
            return self._mock_chat_completion(req_id, created_time, messages, stream=stream)

        # Native llama-cpp-python chat completion
        return self._model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

    def completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """
        Execute OpenAI-compatible standard text completion.
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded or unavailable.")

        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        if self._is_mock:
            return self._mock_completion(req_id, created_time, prompt, stream=stream)

        return self._model(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Mock / Testing Helpers
    # -------------------------------------------------------------------------
    def _mock_chat_completion(
        self, req_id: str, created: int, messages: List[Dict[str, str]], stream: bool = False
    ) -> Any:
        last_msg = messages[-1]["content"] if messages else ""
        mock_response = f"Simulated response to: {last_msg}"

        if not stream:
            return {
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": settings.MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": mock_response},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(last_msg.split()) + 5,
                    "completion_tokens": len(mock_response.split()),
                    "total_tokens": len(last_msg.split()) + len(mock_response.split()) + 5,
                },
            }

        def mock_stream_generator() -> Generator[Dict[str, Any], None, None]:
            words = mock_response.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": settings.MODEL_NAME,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
            yield {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": settings.MODEL_NAME,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

        return mock_stream_generator()

    def _mock_completion(
        self, req_id: str, created: int, prompt: str, stream: bool = False
    ) -> Any:
        mock_response = f"Simulated completion for: {prompt[:30]}..."
        if not stream:
            return {
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": settings.MODEL_NAME,
                "choices": [
                    {
                        "text": mock_response,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(mock_response.split()),
                    "total_tokens": len(prompt.split()) + len(mock_response.split()),
                },
            }

        def mock_stream_generator() -> Generator[Dict[str, Any], None, None]:
            words = mock_response.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield {
                    "id": req_id,
                    "object": "text_completion",
                    "created": created,
                    "model": settings.MODEL_NAME,
                    "choices": [
                        {
                            "text": chunk,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
            yield {
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": settings.MODEL_NAME,
                "choices": [
                    {"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}
                ],
            }

        return mock_stream_generator()


# Global singleton instance
model_manager = ModelManager()
