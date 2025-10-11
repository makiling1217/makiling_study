# main.py  — FastAPI only / LINE bot (image-safe)
import os, hmac, hashlib, base64, json, ast, math, re, logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse, Response
import httpx

# ====== 基本設定 ======
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_CONTENT_URL = "https://api-data.line.me/v2/bot/message/{messageId}/content"  # ← 重要：api-data


# ====== 共通ユーティリティ ======
async def reply_message(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"replyToken": reply_token, "messages": messages}
    async with httpx.AsyncClient(timeout=20) as ac:
        r = await ac.post(LINE_REPLY_URL, headers=headers, json=payload)
        logging.info(f'HTTP Request: POST {LINE_REPLY_URL} "{r.http_version} {r.status_code} {r.reason_phrase}"')
        r.raise_for_status()


async def get_line_image_bytes(message_id: str) -> bytes:
    # 公式どおり api-data.line.me から取得（api.line.me だと 404）
    url = LINE_CONTENT_URL.format(messageId=message_id)
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    async with httpx.AsyncClient(timeout=20) as ac:
        r = await ac.get(url, headers=headers)
        logging.error(f'GET {url} "{r.status_code}"' if r.status_code >= 400 else f'GET {url} "200"')
        r.raise_for_status()
        return r.content


def verify_signature(channel_secret: str, body: bytes, signature: str) -> bool:
    mac = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature or "")


# ====== 安全計算ユーティリティ（calc: ... 用） ======
ALLOWED_FUNCS = {
    "sin": lambda x: math.sin(math.radians(x)),
    "cos": lambda x: math.cos(math.radians(x)),
    "tan": lambda x: math.tan(math.radians(x)),
    "asin": lambda x: math.degrees(math.asin(x)),
    "acos": lambda x: math.degrees(math.acos(x)),
    "atan": lambda x: math.degrees(math.atan(x)),
    "sqrt": math.sqrt,
    "log": math.log,      # 自然対数
    "log10": math.log10,
    "abs": abs,
}
ALLOWED_NAMES = {"pi": math.pi, "e": math.e}

class SafeEval(ast.NodeVisitor):
    def visit(self, node):  # type: ignore[override]
        if isinstance(node, ast.Expression): return self.visit(node.body)
        if isinstance(node, ast.Num): return node.n
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
        if isinstance(node, ast.BinOp):
            l, r = self.visit(node.left), self.visit(node.right)
            if isinstance(node.op, ast.Add): return l + r
            if isinstance(node.op, ast.Sub): return l - r
            if isinstance(node.op, ast.Mult): return l * r
            if isinstance(node.op, ast.Div): return l / r
            if isinstance(node.op, ast.Pow): return l ** r
            if isinstance(node.op, ast.Mod): return l % r
            raise ValueError("operator not allowed")
        if isinstance(node, ast.UnaryOp):
            v = self.visit(node.operand)
            if isinstance(node.op, ast.UAdd): return +v
            if isinstan
