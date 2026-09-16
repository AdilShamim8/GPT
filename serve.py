#!/usr/bin/env python
"""
Production REST API Server for nano-gpt-prod.
Zero external server dependencies (pure Python http.server) with OpenAI API compatibility.
"""

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import torch
from gpt.config.presets import get_preset
from gpt.inference.generator import TextGenerator
from gpt.model.gpt import GPT
from gpt.model.serialization import load_checkpoint
from gpt.serve.chat import handle_chat_completion
from gpt.serve.completions import handle_completion
from gpt.serve.registry import ModelRegistry
from gpt.serve.streaming import stream_chat_completion_sse
from gpt.serve.types import ChatCompletionRequest, CompletionRequest
from gpt.tokenizer.factory import get_tokenizer

logger = logging.getLogger("gpt.serve")


class GPTServerHandler(BaseHTTPRequestHandler):
    generator: TextGenerator = None
    registry: ModelRegistry = None
    web_dir: Path = None

    def _set_cors_headers(self, content_type: str = "application/json"):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Type", content_type)

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/healthz" or self.path == "/health":
            self.send_response(200)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "service": "nano-gpt-prod"}).encode("utf-8"))

        elif self.path == "/v1/models":
            self.send_response(200)
            self._set_cors_headers()
            self.end_headers()
            data = {"object": "list", "data": self.registry.list_cards()}
            self.wfile.write(json.dumps(data).encode("utf-8"))

        # Serve static web interface if requested
        elif self.web_dir and (self.path == "/" or self.path.startswith("/web")):
            file_path = self.web_dir / "index.html" if self.path in {"/", "/web", "/web/"} else self.web_dir / self.path.replace("/web/", "")
            if file_path.exists() and file_path.is_file():
                self.send_response(200)
                content_type = "text/html"
                if file_path.suffix == ".css":
                    content_type = "text/css"
                elif file_path.suffix == ".js":
                    content_type = "application/javascript"
                self._set_cors_headers(content_type)
                self.end_headers()
                self.wfile.write(file_path.read_bytes())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        payload = json.loads(body) if body else {}

        if self.path == "/v1/completions":
            req = CompletionRequest(
                prompt=payload.get("prompt", ""),
                model=payload.get("model", "nano-gpt"),
                temperature=float(payload.get("temperature", 0.8)),
                top_p=float(payload.get("top_p", 0.95)),
                max_tokens=int(payload.get("max_tokens", 100)),
                repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
            )
            resp = handle_completion(self.generator, req)
            self.send_response(200)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif self.path == "/v1/chat/completions":
            req = ChatCompletionRequest(
                messages=payload.get("messages", []),
                model=payload.get("model", "nano-gpt"),
                temperature=float(payload.get("temperature", 0.8)),
                top_p=float(payload.get("top_p", 0.95)),
                max_tokens=int(payload.get("max_tokens", 100)),
                stream=bool(payload.get("stream", False)),
                repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
            )

            if req.stream:
                # SSE streaming mode
                self.send_response(200)
                self._set_cors_headers("text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                for chunk in stream_chat_completion_sse(self.generator, req):
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            else:
                resp = handle_chat_completion(self.generator, req)
                self.send_response(200)
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(host: str = "0.0.0.0", port: int = 8000, web: bool = True):
    server = HTTPServer((host, port), GPTServerHandler)
    print(f"nano-gpt-prod server listening on http://{host}:{port}")
    print(f"  Health check: http://{host}:{port}/healthz")
    print(f"  API Models:   http://{host}:{port}/v1/models")
    print(f"  API Chat:     POST http://{host}:{port}/v1/chat/completions")
    if web:
        print(f"  Web UI:       http://localhost:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Run nano-gpt-prod OpenAI API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--preset", type=str, default="nano_shakespeare")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--web", action="store_true", default=True, help="Serve Web UI")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"

    if args.checkpoint and Path(args.checkpoint).exists():
        model, cfg, _, _ = load_checkpoint(args.checkpoint, device=device)
    else:
        cfg = get_preset(args.preset)
        model = GPT(cfg)

    # Tokenizer
    if args.tokenizer and Path(args.tokenizer).exists():
        tokenizer = get_tokenizer(args.tokenizer)
    else:
        corpus = Path("input.txt").read_text(encoding="utf-8") if Path("input.txt").exists() else "abcdefghijklmnopqrstuvwxyz "
        tokenizer = get_tokenizer("char", text=corpus)

    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)
    registry = ModelRegistry()
    registry.register(args.preset, model)

    GPTServerHandler.generator = generator
    GPTServerHandler.registry = registry
    GPTServerHandler.web_dir = Path(__file__).parent / "gpt" / "web"

    run_server(host=args.host, port=args.port, web=args.web)


if __name__ == "__main__":
    main()
