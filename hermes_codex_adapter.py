#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


LISTEN_HOST = os.environ.get("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "18644"))
UPSTREAM_BASE = os.environ.get("UPSTREAM_BASE", "http://127.0.0.1:18642").rstrip("/")

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
}


def rewrite_developer_roles(value):
    if isinstance(value, dict):
        rewritten = {}
        for key, item in value.items():
            if key == "role" and item == "developer":
                rewritten[key] = "system"
            else:
                rewritten[key] = rewrite_developer_roles(item)
        return rewritten
    if isinstance(value, list):
        return [rewrite_developer_roles(item) for item in value]
    return value


def normalize_response_input(payload):
    if not isinstance(payload, dict):
        return payload

    if payload.get("instructions"):
        instructions = payload.get("instructions")
        existing = payload.get("input")
        system_message = {"role": "system", "content": instructions}
        if isinstance(existing, list):
            payload["input"] = [system_message] + existing
        elif existing:
            payload["input"] = [system_message, {"role": "user", "content": existing}]
        else:
            payload["input"] = [system_message]
        payload.pop("instructions", None)

    if isinstance(payload.get("input"), str):
        payload["input"] = [{"role": "user", "content": payload["input"]}]

    if isinstance(payload.get("input"), list):
        payload["input"] = [normalize_message(item) for item in payload["input"]]

    return payload


def normalize_message(item):
    if not isinstance(item, dict):
        return {"role": "user", "content": str(item)}

    if item.get("type") == "message" and "role" in item:
        item = dict(item)
        item.pop("type", None)

    content = item.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text is None:
                    text = part.get("content")
                if text is not None:
                    parts.append(str(text))
            elif part is not None:
                parts.append(str(part))
        item = dict(item)
        item["content"] = "\n".join(parts)

    return item


@dataclass
class SseEvent:
    event: str | None
    data: dict | None
    raw_data: str | None


class AdapterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stdout.write("%s - %s\n" % (self.address_string(), fmt % args))
        sys.stdout.flush()

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def do_PUT(self):
        self.forward()

    def do_PATCH(self):
        self.forward()

    def do_DELETE(self):
        self.forward()

    def forward(self):
        body = self._read_body()
        headers = self._forward_headers()

        content_type = self.headers.get("Content-Type", "")
        if body and "application/json" in content_type.lower():
            try:
                payload = json.loads(body.decode("utf-8"))
                payload = rewrite_developer_roles(payload)
                if self.path.split("?", 1)[0] == "/v1/responses":
                    payload = normalize_response_input(payload)
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                headers["Content-Type"] = "application/json"
            except Exception as exc:
                self.send_json_error(400, "invalid_json", str(exc))
                return

        if body is not None:
            headers["Content-Length"] = str(len(body))

        upstream_url = self._upstream_url()
        request = urllib.request.Request(
            upstream_url,
            data=body,
            headers=headers,
            method=self.command,
        )

        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                if self._should_transform_sse(response):
                    self._relay_responses_sse(response.status, response.headers, response)
                else:
                    self._relay_response(response.status, response.headers, response)
        except urllib.error.HTTPError as exc:
            self._relay_response(exc.code, exc.headers, exc)
        except Exception as exc:
            self.send_json_error(502, "upstream_error", str(exc))

    def _read_body(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return None
        return self.rfile.read(length)

    def _forward_headers(self):
        headers = {}
        for key, value in self.headers.items():
            if key.lower() not in HOP_BY_HOP_HEADERS:
                headers[key] = value
        return headers

    def _upstream_url(self):
        split = urlsplit(self.path)
        path = split.path or "/"
        if split.query:
            path += "?" + split.query
        return UPSTREAM_BASE + path

    def _relay_response(self, status, headers, response):
        self.send_response(status)
        has_length = False
        for key, value in headers.items():
            lower = key.lower()
            if lower in HOP_BY_HOP_HEADERS:
                continue
            if lower == "content-length":
                has_length = True
            self.send_header(key, value)
        if not has_length:
            self.send_header("Connection", "close")
        self.end_headers()

        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()

    def _should_transform_sse(self, response):
        content_type = response.headers.get("Content-Type", "")
        return self.path.split("?", 1)[0] == "/v1/responses" and "text/event-stream" in content_type.lower()

    def _relay_responses_sse(self, status, headers, response):
        self.send_response(status)
        for key, value in headers.items():
            lower = key.lower()
            if lower in HOP_BY_HOP_HEADERS or lower == "content-length":
                continue
            self.send_header(key, value)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        sequence = 0
        active_item_id = None
        active_text = []
        content_part_open = False

        for event in iter_sse_events(response):
            for outgoing in expand_responses_event(event, active_item_id, active_text, content_part_open):
                event_name, payload, state = outgoing
                if state.get("active_item_id") is not None:
                    active_item_id = state["active_item_id"]
                if state.get("append_delta") is not None:
                    active_text.append(state["append_delta"])
                if "content_part_open" in state:
                    content_part_open = state["content_part_open"]

                if isinstance(payload, dict):
                    payload["sequence_number"] = sequence
                    sequence += 1
                    self._write_sse(event_name, payload)
                elif event.raw_data is not None:
                    self.wfile.write((f"event: {event_name}\n" if event_name else "").encode("utf-8"))
                    self.wfile.write(f"data: {event.raw_data}\n\n".encode("utf-8"))
                    self.wfile.flush()

    def _write_sse(self, event_name, payload):
        self.wfile.write(f"event: {event_name}\n".encode("utf-8"))
        self.wfile.write(b"data: ")
        self.wfile.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        self.wfile.write(b"\n\n")
        self.wfile.flush()

    def send_json_error(self, status, code, message):
        body = json.dumps({"error": {"code": code, "message": message}}, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def iter_sse_events(response):
    event_name = None
    data_lines = []
    for raw_line in response:
        line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
        if line == "":
            if event_name or data_lines:
                raw_data = "\n".join(data_lines)
                data = None
                if raw_data:
                    try:
                        data = json.loads(raw_data)
                    except Exception:
                        pass
                yield SseEvent(event_name, data, raw_data)
            event_name = None
            data_lines = []
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())


def expand_responses_event(event, active_item_id, active_text, content_part_open):
    if not isinstance(event.data, dict):
        yield (event.event, event.data, {})
        return

    event_name = event.event or event.data.get("type")
    data = dict(event.data)
    data.pop("sequence_number", None)

    if event_name == "response.output_item.added":
        item_id = data.get("item", {}).get("id")
        yield (event_name, data, {"active_item_id": item_id})
        if item_id:
            part = {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": data.get("output_index", 0),
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            }
            yield ("response.content_part.added", part, {"content_part_open": True})
        return

    if event_name == "response.output_text.delta":
        if active_item_id and "item_id" not in data:
            data["item_id"] = active_item_id
        yield (event_name, data, {"append_delta": data.get("delta", "")})
        return

    if event_name == "response.output_text.done":
        if active_item_id and "item_id" not in data:
            data["item_id"] = active_item_id
        if "text" not in data:
            data["text"] = "".join(active_text)
        yield (event_name, data, {})
        if content_part_open:
            part = {
                "type": "response.content_part.done",
                "item_id": data.get("item_id") or active_item_id,
                "output_index": data.get("output_index", 0),
                "content_index": data.get("content_index", 0),
                "part": {"type": "output_text", "text": data.get("text", "".join(active_text))},
            }
            yield ("response.content_part.done", part, {"content_part_open": False})
        return

    yield (event_name, data, {})


def main():
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), AdapterHandler)
    print(f"hermes-codex-adapter listening on {LISTEN_HOST}:{LISTEN_PORT}, upstream={UPSTREAM_BASE}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
