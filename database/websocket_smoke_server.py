#!/usr/bin/env python3
"""Small WebSocket endpoint for local smoke tests of the sensor bridge."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import socket
import socketserver
import struct
import threading
from dataclasses import dataclass, field


WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


@dataclass
class ReceivedMessages:
    values: list[dict] = field(default_factory=list)
    event: threading.Event = field(default_factory=threading.Event)

    def append(self, payload: dict) -> None:
        self.values.append(payload)
        self.event.set()


def websocket_accept(key: str) -> str:
    digest = hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def read_http_headers(conn: socket.socket) -> dict[str, str]:
    raw = b""
    while b"\r\n\r\n" not in raw:
        chunk = conn.recv(4096)
        if not chunk:
            break
        raw += chunk
        if len(raw) > 65536:
            raise ValueError("HTTP header too large")

    lines = raw.decode("iso-8859-1").split("\r\n")
    headers = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    return headers


def read_exact(conn: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_ws_text(conn: socket.socket) -> str | None:
    first, second = read_exact(conn, 2)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F

    if opcode == 0x8:
        return None
    if opcode != 0x1:
        raise ValueError(f"unsupported opcode: {opcode}")

    if length == 126:
        length = struct.unpack("!H", read_exact(conn, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", read_exact(conn, 8))[0]

    mask = read_exact(conn, 4) if masked else b""
    payload = read_exact(conn, length)
    if masked:
        payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    return payload.decode("utf-8")


def write_ws_text(conn: socket.socket, text: str) -> None:
    payload = text.encode("utf-8")
    header = bytearray([0x81])
    if len(payload) < 126:
        header.append(len(payload))
    elif len(payload) <= 0xFFFF:
        header.append(126)
        header.extend(struct.pack("!H", len(payload)))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", len(payload)))
    conn.sendall(bytes(header) + payload)


def make_masked_client_text_frame(text: str) -> bytes:
    payload = text.encode("utf-8")
    mask = b"\x01\x02\x03\x04"
    header = bytearray([0x81])
    if len(payload) < 126:
        header.append(0x80 | len(payload))
    elif len(payload) <= 0xFFFF:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", len(payload)))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", len(payload)))
    header.extend(mask)
    header.extend(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    return bytes(header)


class SmokeWebSocketHandler(socketserver.BaseRequestHandler):
    messages: ReceivedMessages

    def handle(self) -> None:
        headers = read_http_headers(self.request)
        key = headers.get("sec-websocket-key")
        if not key:
            self.request.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            return

        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {websocket_accept(key)}\r\n"
            "\r\n"
        )
        self.request.sendall(response.encode("ascii"))

        while True:
            try:
                text = read_ws_text(self.request)
            except ConnectionError:
                return
            if text is None:
                return
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {"raw": text}
            self.messages.append(payload)
            write_ws_text(self.request, json.dumps({"ok": True}, ensure_ascii=False))


def run_server(host: str, port: int, messages: ReceivedMessages) -> socketserver.ThreadingTCPServer:
    class Handler(SmokeWebSocketHandler):
        pass

    Handler.messages = messages
    server = socketserver.ThreadingTCPServer((host, port), Handler)
    server.daemon_threads = True
    server.allow_reuse_address = True
    return server


def self_test() -> int:
    messages = ReceivedMessages()
    server = run_server("127.0.0.1", 0, messages)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        from database.send import SAMPLE_SENSOR_LINE, parse_sensor_line
    except ModuleNotFoundError:
        from send import SAMPLE_SENSOR_LINE, parse_sensor_line

    payload = parse_sensor_line(SAMPLE_SENSOR_LINE)
    assert payload is not None

    with socket.create_connection((host, port), timeout=3.0) as conn:
        key = base64.b64encode(b"0123456789abcdef").decode("ascii")
        request = (
            "GET /CIS_Server/eggstream HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        conn.sendall(request.encode("ascii"))
        response = conn.recv(4096)
        if b"101 Switching Protocols" not in response:
            return 1

        conn.sendall(make_masked_client_text_frame(json.dumps(payload)))
        read_ws_text(conn)

    ok = messages.event.wait(timeout=2.0) and messages.values and messages.values[0]["type"] == "robot_update"
    server.shutdown()
    server.server_close()
    print("websocket_smoke_server self-test:", "OK" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    messages = ReceivedMessages()
    server = run_server(args.host, args.port, messages)
    print(f"WebSocket smoke server listening on ws://{args.host}:{args.port}/CIS_Server/eggstream")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
