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
            if isinstance(node.op, ast.USub): return -v
            raise ValueError("unary op not allowed")
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_NAMES: return ALLOWED_NAMES[node.id]
            raise ValueError("name not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name): raise ValueError("call not allowed")
            fname = node.func.id
            if fname not in ALLOWED_FUNCS: raise ValueError(f"func {fname} not allowed")
            args = [self.visit(a) for a in node.args]
            return ALLOWED_FUNCS[fname](*args)
        raise ValueError("node not allowed")

def safe_calc(expr: str) -> float:
    # 例: sin(30)+3^2, sqrt(2), log10(100)
    expr = expr.replace("^", "**")
    tree = ast.parse(expr, mode="eval")
    return SafeEval().visit(tree)

def cg50_keyseq(expr: str) -> str:
    seq = expr
    seq = re.sub(r"\s+", "", seq)
    seq = seq.replace("^", "**")  # 統一
    seq = (seq.replace("sin", "[SIN]").replace("cos", "[COS]").replace("tan", "[TAN]")
               .replace("asin", "[SHIFT][SIN]^-1").replace("acos", "[SHIFT][COS]^-1").replace("atan", "[SHIFT][TAN]^-1")
               .replace("sqrt", "[√]").replace("log10", "[LOG] 10 , ").replace("log", "[LN]")
               .replace("**", "^").replace("*", "×").replace("/", "÷"))
    return "角度:Deg を確認 → 入力: " + seq + " → [EXE]"


# ====== ルーティング ======
@app.get("/")
async def root():
    return {"ok": True, "message": "LINE bot (FastAPI) running"}

@app.get("/botinfo")
async def botinfo():
    # 自己診断：トークンが正しいか（200で一致、401は不一致）
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    async with httpx.AsyncClient(timeout=10) as ac:
        r = await ac.get("https://api.line.me/v2/bot/info", headers=headers)
    return Response(r.text, media_type="application/json", status_code=r.status_code)

@app.post("/webhook")
async def webhook(request: Request, x_line_signature: Optional[str] = Header(default=None)):
    body_bytes = await request.body()
    # 署名検証（必要に応じて一時オフにして切り分け可）
    if LINE_CHANNEL_SECRET and not verify_signature(LINE_CHANNEL_SECRET, body_bytes, x_line_signature or ""):
        logging.error("Signature verify failed")
        return JSONResponse({"message": "signature error"}, status_code=400)

    logging.info('POST /webhook "HTTP/1.1 200 OK"')
    body = json.loads(body_bytes.decode("utf-8"))
    events = body.get("events", [])

    for event in events:
        etype = event.get("type")
        if etype != "message":
            continue

        reply_token = event.get("replyToken")
        m = event.get("message", {})
        msg_type = m.get("type")

        logging.info(f'message.id = {m.get("id")} type={msg_type}')

        try:
            if msg_type == "text":
                text = (m.get("text") or "").strip()
                if text.lower() == "ping":
                    await reply_message(reply_token, [{"type": "text", "text": "pong ✅"}])

                elif text.lower().startswith("calc:"):
                    expr = text[5:].strip()
                    try:
                        val = safe_calc(expr)
                        seq = cg50_keyseq(expr)
                        msg = f"計算OK ✅\n式: {expr}\n結果: {val}\n\nfx-CG50操作ガイド:\n{seq}"
                    except Exception as e:
                        msg = f"式の解析に失敗しました ❌\n入力例: calc: sin(30)+3^2\n詳細: {e}"
                    await reply_message(reply_token, [{"type": "text", "text": msg}])

                else:
                    await reply_message(reply_token, [{
                        "type": "text",
                        "text": "受信しました。\n計算は `calc: ...` 形式で送ってね。\n例: `calc: sin(30)+3^2`",
                    }])

            elif msg_type == "image":
                # contentProvider が external の場合は外部URL直取り
                cp = m.get("contentProvider") or {}
                if cp.get("type") == "external" and cp.get("originalContentUrl"):
                    async with httpx.AsyncClient(timeout=20) as ac:
                        r = await ac.get(cp["originalContentUrl"])
                        r.raise_for_status()
                        img_bytes = r.content
                    logging.info("Downloaded external image OK")
                else:
                    img_bytes = await get_line_image_bytes(m.get("id"))
                    logging.info(f"Downloaded image bytes: {len(img_bytes)}")

                # ★誤答防止のため、現状はテキスト誘導のみ（OCR/解法は検算付きで後日ON）
                guide = (
                    "📷 画像を受け取りました！\n"
                    "誤答防止のため、今は画像の自動解法を停止しています。\n"
                    "まずはテキストで式を送ってください。\n"
                    "例:  calc: sin(30)+3^2"
                )
                await reply_message(reply_token, [{"type": "text", "text": guide}])

            else:
                await reply_message(reply_token, [{"type": "text", "text": f"未対応メッセージタイプ: {msg_type}"}])

        except httpx.HTTPStatusError as he:
            await reply_message(reply_token, [{"type": "text", "text": f"HTTPエラー: {he.response.status_code}"}])
            logging.exception("HTTPStatusError")
        except Exception:
            await reply_message(reply_token, [{"type": "text", "text": "内部エラーが発生しました。"}])
            logging.exception("Unhandled error")

    return JSONResponse({"status": "ok"})
