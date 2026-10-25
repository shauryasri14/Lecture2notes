"""Exercises the REAL OllamaClient + CLI against a local fake Ollama HTTP server
(no model needed). Verifies wiring and exit codes, not model quality."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from conftest import FIXTURE, fake_response
from l2n.cli import main


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"models": [{"name": "gemma3:4b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send(200, {"response": fake_response(body["prompt"], body.get("format") == "json"), "done": True})


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_cli_from_transcript_end_to_end(server, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_HOST", server)
    monkeypatch.chdir(tmp_path)
    code = main(["--from-transcript", str(FIXTURE), "--output-dir", str(tmp_path / "o"),
                 "--chunk-size", "50", "--chunk-overlap", "5"])
    assert code == 0, capsys.readouterr()
    out = next((tmp_path / "o").iterdir())
    assert "next Friday" in (out / "notes.md").read_text()
    assert len(json.loads((out / "action_items.json").read_text())["action_items"]) == 1


def test_cli_reports_ollama_down(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens
    monkeypatch.chdir(tmp_path)
    assert main(["--from-transcript", str(FIXTURE), "--output-dir", str(tmp_path / "o")]) == 1
    assert "Cannot reach Ollama" in capsys.readouterr().err


def test_cli_argument_validation(capsys):
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(["a.mp3", "--watch", "inbox"])
    assert main(["--from-transcript", str(FIXTURE), "--chunk-size", "100", "--chunk-overlap", "100"]) == 1
    assert main(["--from-transcript", str(FIXTURE), "--lecture-date", "yesterday"]) == 1
    assert main(["--help"] if False else ["missing.mp3", "--output-dir", "/tmp/l2n_x"]) in (1,)
