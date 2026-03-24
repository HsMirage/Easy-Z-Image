# -*- coding: utf-8 -*-
"""OpenAI 兼容图片生成 API 服务"""
import base64
import io
import json
import signal
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

# Load .env early.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from config import (
    DEFAULT_HEIGHT,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    OPENAI_COMPAT_HOST,
    OPENAI_COMPAT_PORT,
    MODEL_ID,
    OPENAI_COMPAT_API_KEY,
    OPENAI_COMPAT_DEFAULT_RESPONSE_FORMAT,
    OPENAI_COMPAT_MAX_IMAGES_PER_REQUEST,
    OPENAI_COMPAT_MODEL_NAME,
    OPENAI_COMPAT_PUBLIC_BASE_URL,
    OPENAI_COMPAT_SAVE_ROOT,
    WORKER_UNLOAD_MODEL_AFTER_JOB,
    WORKER_UNLOAD_MODEL_ON_ERROR,
    PRELOAD_MODEL_ON_START,
)
from generator import GenerationParams, get_generation_lock, get_generator


class OpenAICompatError(Exception):
    """带状态码的 API 错误"""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        code: Optional[str] = None,
        param: Optional[str] = None,
        error_type: str = "invalid_request_error",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.param = param
        self.error_type = error_type


class CompatHTTPServer(ThreadingHTTPServer):
    """线程化 HTTP 服务"""

    daemon_threads = True
    allow_reuse_address = True


class OpenAICompatService:
    """复用现有生成器，对外暴露 OpenAI 兼容图片接口"""

    def __init__(self):
        self.generator = get_generator()
        self.storage_root = OPENAI_COMPAT_SAVE_ROOT
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.runtime_lock = get_generation_lock()
        self.started_at = int(time.time())
        self.current_request_id: Optional[str] = None

    def authenticate(self, headers) -> None:
        """验证 Bearer Token 或 X-API-Key。未配置时放行。"""
        if not OPENAI_COMPAT_API_KEY:
            return

        auth_header = headers.get("Authorization", "")
        api_key_header = headers.get("X-API-Key", "")
        bearer_token = ""
        if auth_header.lower().startswith("bearer "):
            bearer_token = auth_header[7:].strip()

        if bearer_token == OPENAI_COMPAT_API_KEY or api_key_header == OPENAI_COMPAT_API_KEY:
            return

        raise OpenAICompatError(
            HTTPStatus.UNAUTHORIZED,
            "Invalid API key",
            code="invalid_api_key",
            error_type="authentication_error",
        )

    def list_models(self) -> dict:
        """返回模型列表"""
        created = int(datetime.now(timezone.utc).timestamp())
        model_ids = []
        for model_id in (OPENAI_COMPAT_MODEL_NAME, MODEL_ID):
            if model_id not in model_ids:
                model_ids.append(model_id)

        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "z-image-worker",
                }
                for model_id in model_ids
            ],
        }

    def health(self) -> dict:
        """返回服务状态"""
        placement = self.generator.placement
        return {
            "object": "health",
            "status": "busy" if self.runtime_lock.locked() else "idle",
            "model": OPENAI_COMPAT_MODEL_NAME,
            "backing_model": MODEL_ID,
            "generator_loaded": self.generator.loaded,
            "placement_mode": placement.mode,
            "execution_device": placement.execution_device,
            "dtype": placement.dtype_name,
            "uptime_seconds": max(int(time.time()) - self.started_at, 0),
        }

    def generate_images(self, payload: dict, base_url: str) -> dict:
        """处理 OpenAI 兼容图片生成请求"""
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise OpenAICompatError(HTTPStatus.BAD_REQUEST, "Field `prompt` is required", param="prompt")

        requested_model = str(payload.get("model") or OPENAI_COMPAT_MODEL_NAME).strip()
        size = str(payload.get("size") or "").strip()
        width, height = self._parse_dimensions(size=size, width=payload.get("width"), height=payload.get("height"))
        steps = self._parse_positive_int(payload.get("steps"), default=DEFAULT_STEPS, field_name="steps")
        n = self._parse_positive_int(payload.get("n"), default=1, field_name="n")
        if n > OPENAI_COMPAT_MAX_IMAGES_PER_REQUEST:
            raise OpenAICompatError(
                HTTPStatus.BAD_REQUEST,
                f"Field `n` exceeds the configured limit ({OPENAI_COMPAT_MAX_IMAGES_PER_REQUEST})",
                param="n",
            )

        seed = self._parse_optional_int(payload.get("seed"), field_name="seed")
        response_format = self._normalize_response_format(payload.get("response_format"))
        user = payload.get("user")
        created = int(time.time())

        data = []
        with self.runtime_lock:
            request_id = uuid.uuid4().hex
            self.current_request_id = request_id
            try:
                for index in range(n):
                    effective_seed = -1 if seed is None else seed + index
                    params = GenerationParams(
                        prompt=prompt,
                        width=width,
                        height=height,
                        steps=steps,
                        seed=effective_seed,
                    )
                    image, metadata = self.generator.generate(params)
                    relative_path, image_bytes = self._save_image(
                        image=image,
                        prompt=prompt,
                        model=requested_model,
                        user=user,
                        metadata=metadata,
                    )

                    item = {"revised_prompt": prompt}
                    if response_format == "b64_json":
                        item["b64_json"] = base64.b64encode(image_bytes).decode("ascii")
                    else:
                        item["url"] = f"{base_url}/v1/images/files/{relative_path}"
                    data.append(item)
            except Exception:
                if WORKER_UNLOAD_MODEL_ON_ERROR:
                    print("[API] Unloading model after error to release VRAM")
                    self.generator.unload_model()
                raise
            finally:
                self.current_request_id = None
                if WORKER_UNLOAD_MODEL_AFTER_JOB and self.generator.loaded:
                    print("[API] Unloading model after request")
                    self.generator.unload_model()
                else:
                    if self.generator.loaded:
                        print("[API] Keeping model resident on GPU(s)")
                    self.generator.release_memory()

        return {
            "object": "list",
            "created": created,
            "data": data,
        }

    def generate_chat_completion(self, payload: dict, base_url: str) -> dict:
        """兼容部分客户端错误地通过 chat/completions 发起图片生成。"""
        image_payload = self._chat_payload_to_image_payload(payload)
        image_result = self.generate_images(image_payload, base_url)
        requested_model = str(payload.get("model") or OPENAI_COMPAT_MODEL_NAME).strip()

        content_parts = []
        for item in image_result["data"]:
            if item.get("url"):
                content_parts.append(f"![generated image]({item['url']})")
            elif item.get("b64_json"):
                content_parts.append(f"![generated image](data:image/png;base64,{item['b64_json']})")

        content = "\n".join(content_parts).strip()
        if not content:
            content = "Image generation completed."

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": image_result["created"],
            "model": requested_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "images": image_result["data"],
        }

    def load_image_bytes(self, relative_path: str) -> bytes:
        """读取已生成的图片，供 URL 响应回传"""
        clean_path = unquote(relative_path).lstrip("/")
        if not clean_path:
            raise OpenAICompatError(HTTPStatus.NOT_FOUND, "Image not found", code="not_found")

        image_path = (self.storage_root / clean_path).resolve()
        storage_root = self.storage_root.resolve()
        if storage_root not in image_path.parents and image_path != storage_root:
            raise OpenAICompatError(HTTPStatus.NOT_FOUND, "Image not found", code="not_found")
        if not image_path.is_file() or image_path.suffix.lower() != ".png":
            raise OpenAICompatError(HTTPStatus.NOT_FOUND, "Image not found", code="not_found")

        return image_path.read_bytes()

    @staticmethod
    def _parse_positive_int(value, *, default: int, field_name: str) -> int:
        if value is None or value == "":
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise OpenAICompatError(
                HTTPStatus.BAD_REQUEST,
                f"Field `{field_name}` must be an integer",
                param=field_name,
            ) from exc
        if parsed <= 0:
            raise OpenAICompatError(
                HTTPStatus.BAD_REQUEST,
                f"Field `{field_name}` must be greater than 0",
                param=field_name,
            )
        return parsed

    @staticmethod
    def _parse_optional_int(value, *, field_name: str) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise OpenAICompatError(
                HTTPStatus.BAD_REQUEST,
                f"Field `{field_name}` must be an integer",
                param=field_name,
            ) from exc

    @staticmethod
    def _parse_dimensions(*, size: str, width, height) -> tuple[int, int]:
        if size and size.lower() != "auto":
            normalized = size.lower().replace("*", "x")
            parts = normalized.split("x")
            if len(parts) != 2:
                raise OpenAICompatError(
                    HTTPStatus.BAD_REQUEST,
                    "Field `size` must look like `1024x1024`",
                    param="size",
                )
            try:
                return int(parts[0]), int(parts[1])
            except ValueError as exc:
                raise OpenAICompatError(
                    HTTPStatus.BAD_REQUEST,
                    "Field `size` must look like `1024x1024`",
                    param="size",
                ) from exc

        parsed_width = OpenAICompatService._parse_positive_int(width, default=DEFAULT_WIDTH, field_name="width")
        parsed_height = OpenAICompatService._parse_positive_int(height, default=DEFAULT_HEIGHT, field_name="height")
        return parsed_width, parsed_height

    @staticmethod
    def _normalize_response_format(value) -> str:
        fmt_value = value
        if isinstance(value, dict):
            # 兼容部分客户端将 response_format 包装成对象，例如 {"type": "url"}。
            for key in ("type", "value", "format"):
                candidate = value.get(key)
                if candidate is not None:
                    fmt_value = candidate
                    break
            else:
                fmt_value = None

        fmt = str(fmt_value or OPENAI_COMPAT_DEFAULT_RESPONSE_FORMAT).strip().lower()
        if fmt in {"b64", "base64"}:
            fmt = "b64_json"
        if fmt not in {"url", "b64_json"}:
            raise OpenAICompatError(
                HTTPStatus.BAD_REQUEST,
                "Field `response_format` must be `url` or `b64_json`",
                param="response_format",
            )
        return fmt

    @staticmethod
    def _chat_payload_to_image_payload(payload: dict) -> dict:
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            prompt = OpenAICompatService._extract_prompt_from_messages(payload.get("messages"))
        if not prompt:
            raise OpenAICompatError(
                HTTPStatus.BAD_REQUEST,
                "Unable to extract an image prompt from chat messages",
                param="messages",
            )

        # Default to b64_json for chat completions: the caller may not be able
        # to fetch image URLs (e.g. Cherry Studio behind NAT / localhost).
        response_format = payload.get("response_format") or "b64_json"
        image_payload = {
            "prompt": prompt,
            "model": payload.get("model"),
            "size": payload.get("size"),
            "width": payload.get("width"),
            "height": payload.get("height"),
            "steps": payload.get("steps"),
            "n": payload.get("n"),
            "seed": payload.get("seed"),
            "response_format": response_format,
            "user": payload.get("user"),
        }
        return {key: value for key, value in image_payload.items() if value is not None}

    @staticmethod
    def _extract_prompt_from_messages(messages) -> str:
        if not isinstance(messages, list):
            return ""

        text_parts = []
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "").strip().lower() not in {"user", ""}:
                continue

            content = message.get("content")
            if isinstance(content, str):
                text = content.strip()
                if text:
                    return text
                continue

            if not isinstance(content, list):
                continue

            text_parts.clear()
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "").strip().lower()
                if item_type in {"text", "input_text"}:
                    text = str(item.get("text") or "").strip()
                    if text:
                        text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)

        return ""

    def _save_image(self, *, image, prompt: str, model: str, user, metadata: dict) -> tuple[str, bytes]:
        """保存 PNG 及元数据"""
        date_path = datetime.now().strftime("%Y-%m-%d")
        file_id = uuid.uuid4().hex
        target_dir = self.storage_root / date_path
        target_dir.mkdir(parents=True, exist_ok=True)

        image_bytes = self._image_to_png_bytes(image)
        image_path = target_dir / f"{file_id}.png"
        metadata_path = target_dir / f"{file_id}.json"

        image_path.write_bytes(image_bytes)
        metadata_path.write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "model": model,
                    "user": user,
                    "saved_at": datetime.now().isoformat(),
                    "metadata": metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return f"{date_path}/{image_path.name}", image_bytes

    @staticmethod
    def _image_to_png_bytes(image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


class OpenAICompatRequestHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理"""

    server_version = "ZImageOpenAICompat/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[HTTP] {self.address_string()} - {fmt % args}")

    @property
    def service(self) -> OpenAICompatService:
        return self.server.service  # type: ignore[attr-defined]

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        try:
            path = self._normalized_path()
            if path in {"/", ""}:
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "object": "info",
                        "service": "z-image-openai-compatible-api",
                        "images_generation_endpoint": "/v1/images/generations",
                        "models_endpoint": "/v1/models",
                        "health_endpoint": "/health",
                        "model": OPENAI_COMPAT_MODEL_NAME,
                    },
                )
                return

            if path in {"/health", "/v1/health"}:
                self._send_json(HTTPStatus.OK, self.service.health())
                return

            if path in {"/models", "/v1/models"}:
                self.service.authenticate(self.headers)
                self._send_json(HTTPStatus.OK, self.service.list_models())
                return

            file_prefixes = ("/images/files/", "/v1/images/files/")
            if any(path.startswith(prefix) for prefix in file_prefixes):
                relative_path = path.split("/images/files/", 1)[1]
                image_bytes = self.service.load_image_bytes(relative_path)
                self._send_bytes(HTTPStatus.OK, image_bytes, "image/png")
                return

            raise OpenAICompatError(HTTPStatus.NOT_FOUND, "Not found", code="not_found")
        except OpenAICompatError as exc:
            self._send_openai_error(exc)
        except Exception as exc:
            traceback.print_exc()
            self._send_openai_error(
                OpenAICompatError(HTTPStatus.INTERNAL_SERVER_ERROR, f"Internal error: {exc}", code="server_error")
            )

    def do_POST(self):
        try:
            path = self._normalized_path()
            if path in {"/chat/completions", "/v1/chat/completions"}:
                self.service.authenticate(self.headers)
                payload = self._read_json_body()
                base_url = self._public_base_url()
                result = self.service.generate_chat_completion(payload, base_url)
                self._send_json(HTTPStatus.OK, result)
                return

            if path in {"/images/generations", "/v1/images/generations"}:
                self.service.authenticate(self.headers)
                payload = self._read_json_body()
                base_url = self._public_base_url()
                result = self.service.generate_images(payload, base_url)
                self._send_json(HTTPStatus.OK, result)
                return

            raise OpenAICompatError(HTTPStatus.NOT_FOUND, "Not found", code="not_found")
        except OpenAICompatError as exc:
            self._send_openai_error(exc)
        except Exception as exc:
            traceback.print_exc()
            self._send_openai_error(
                OpenAICompatError(HTTPStatus.INTERNAL_SERVER_ERROR, f"Internal error: {exc}", code="server_error")
            )

    def _normalized_path(self) -> str:
        path = urlparse(self.path).path.rstrip("/") or "/"
        while path.startswith("/v1/v1/"):
            path = path.replace("/v1/v1/", "/v1/", 1)
        if path == "/v1/v1":
            path = "/v1"
        return path

    def _read_json_body(self) -> dict:
        content_length = self.headers.get("Content-Length", "").strip()
        transfer_encoding = self.headers.get("Transfer-Encoding", "").strip().lower()
        chunked = "chunked" in transfer_encoding

        if not content_length and not chunked:
            # No Content-Length and no chunked encoding: cannot determine body length.
            # Raise a clear error so clients can debug the issue.
            raise OpenAICompatError(
                HTTPStatus.LENGTH_REQUIRED,
                "Content-Length header is required",
                code="invalid_request",
            )

        if chunked:
            # Read chunked transfer-encoded body
            chunks = []
            while True:
                size_line = self.rfile.readline().strip()
                try:
                    chunk_size = int(size_line, 16)
                except ValueError:
                    break
                if chunk_size == 0:
                    break
                chunks.append(self.rfile.read(chunk_size))
                self.rfile.read(2)  # CRLF after chunk
            raw_body = b"".join(chunks)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise OpenAICompatError(
                    HTTPStatus.BAD_REQUEST,
                    "Request body must be valid JSON",
                    code="invalid_request",
                ) from exc
            if not isinstance(payload, dict):
                raise OpenAICompatError(
                    HTTPStatus.BAD_REQUEST,
                    "Request body must be a JSON object",
                    code="invalid_request",
                )
            return payload

        try:
            length = int(content_length)
        except ValueError as exc:
            raise OpenAICompatError(
                HTTPStatus.BAD_REQUEST,
                "Invalid Content-Length header",
                code="invalid_request",
            ) from exc

        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenAICompatError(
                HTTPStatus.BAD_REQUEST,
                "Request body must be valid JSON",
                code="invalid_request",
            ) from exc

        if not isinstance(payload, dict):
            raise OpenAICompatError(
                HTTPStatus.BAD_REQUEST,
                "Request body must be a JSON object",
                code="invalid_request",
            )
        return payload

    def _public_base_url(self) -> str:
        if OPENAI_COMPAT_PUBLIC_BASE_URL:
            return OPENAI_COMPAT_PUBLIC_BASE_URL

        host = self.headers.get("Host", "").strip()
        if not host:
            host = f"localhost:{OPENAI_COMPAT_PORT}"
        scheme = self.headers.get("X-Forwarded-Proto", "http").strip() or "http"
        return f"{scheme}://{host}".rstrip("/")

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status_code: int, payload: bytes, content_type: str) -> None:
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_openai_error(self, exc: OpenAICompatError) -> None:
        self._send_json(
            exc.status_code,
            {
                "error": {
                    "message": exc.message,
                    "type": exc.error_type,
                    "param": exc.param,
                    "code": exc.code,
                }
            },
        )

    def _send_cors_headers(self) -> None:
        origin = self.headers.get("Origin", "").strip()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        else:
            self.send_header("Access-Control-Allow-Origin", "*")

        request_headers = self.headers.get("Access-Control-Request-Headers", "").strip()
        if request_headers:
            self.send_header("Access-Control-Allow-Headers", request_headers)
            self.send_header("Vary", "Access-Control-Request-Headers")
        else:
            self.send_header(
                "Access-Control-Allow-Headers",
                "Authorization, Content-Type, X-API-Key, OpenAI-Organization, OpenAI-Project",
            )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Max-Age", "86400")
        if self.headers.get("Access-Control-Request-Private-Network", "").strip().lower() == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")


def create_server(
    *,
    host: str = OPENAI_COMPAT_HOST,
    port: int = OPENAI_COMPAT_PORT,
    service: Optional[OpenAICompatService] = None,
) -> tuple[OpenAICompatService, CompatHTTPServer]:
    """创建可独立运行或内嵌到 Worker 中的 OpenAI 兼容服务实例。"""
    service = service or OpenAICompatService()
    server = CompatHTTPServer((host, port), OpenAICompatRequestHandler)
    server.service = service  # type: ignore[attr-defined]
    return service, server


def print_startup_banner(*, host: str = OPENAI_COMPAT_HOST, port: int = OPENAI_COMPAT_PORT) -> None:
    """打印统一的 OpenAI 兼容 API 启动信息。"""
    print("=" * 60)
    print("  Z-Image OpenAI-Compatible Image API")
    print(f"  Listen: http://{host}:{port}")
    print(f"  Model:  {OPENAI_COMPAT_MODEL_NAME}")
    print(f"  Auth:   {'enabled' if OPENAI_COMPAT_API_KEY else 'disabled'}")
    print(
        "  Unload after request: "
        f"{'enabled' if WORKER_UNLOAD_MODEL_AFTER_JOB else 'disabled'}"
    )
    print("=" * 60)
    print("[API] Endpoints:")
    print("[API]   GET  /v1/models")
    print("[API]   GET  /health")
    print("[API]   POST /v1/images/generations")


def main() -> None:
    """启动 OpenAI 兼容服务"""
    service, server = create_server()

    print_startup_banner()

    if PRELOAD_MODEL_ON_START:
        print("[API] Preloading model onto GPU(s) before serving...")
        lock = service.runtime_lock
        with lock:
            service.generator.load_model()
        print(
            f"[API] Preload complete: mode={service.generator.placement.mode}, "
            f"dtype={service.generator.placement.dtype_name}, "
            f"execution={service.generator.placement.execution_device}"
        )
    else:
        print("[API] Model loading is deferred until the first request")

    def shutdown_handler(signum, frame):
        print(f"\n[Signal] Received signal {signum}, shutting down API server...")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        server.serve_forever()
    finally:
        server.server_close()
        print("[API] Server stopped")


if __name__ == "__main__":
    main()
