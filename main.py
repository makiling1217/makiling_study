import base64, hashlib, hmac, json, os, logging
from typing import Any, Dict, List
from fastapi import FastAPI, Request, HTTPException
import httpx

app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
HEADERS_JSON = {
    "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

def verify_signature(body_bytes: bytes, signature: str) -> bool:
    if not LINE_CHANNEL_SECRET:
        logging.error("ENV LINE_CHANNEL_SECRET is empty")
        return False
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body_bytes, hashlib.sha256).digest()
    expect = base64.b64encode(mac).decode("utf-8")
    ok = hmac.compare_digest(expect, signature)
    if not ok:
        logging.error("Signature NG (expect=%s, got=%s)", expect, signature)
    return ok

async def line_reply(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    async with httpx.AsyncClient(timeout=10) as ac:
        r = await ac.post(LINE_REPLY_URL, headers=HEADERS_JSON, json={
            "replyToken": reply_token,
            "messages": messages[:5]  # LINE仕様：最大5件
        })
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            logging.exception("LINE reply error %s %s", r.status_code, r.text)
            raise e

@app.get("/")
def health():
    return {"ok": True}

@app.post("/webhook")
async def webhook(request: Request):
    # ① 署名検証
    signature = request.headers.get("x-line-signature", "")
    body_bytes = await request.body()
    if not verify_signature(body_bytes, signature):
        raise HTTPException(status_code=400, detail="Bad signature")

    body = json.loads(body_bytes.decode("utf-8"))
    events = body.get("events", [])
    logging.info("events=%s", json.dumps(events, ensure_ascii=False))

    # ② 各イベント処理
    for ev in events:
        etype = ev.get("type")
        if etype != "message":
            # 既読など他イベントはスキップ（必要なら追加）
            continue

        msg = ev.get("message", {})
        mtype = msg.get("type")
        reply_token = ev.get("replyToken")

        try:
            if mtype == "text":
                user_text = msg.get("text", "")
                await line_reply(reply_token, [{
                    "type": "text",
                    "text": f"受け取りました：{user_text}"
                }])

            elif mtype == "image":
                # 画像は内容を取得せず “受信したよ” と返す（まずは無反応回避）
                await line_reply(reply_token, [{
                    "type": "text",
                    "text": "画像を受信しました📷（解析は未対応です。テキストで問題文を送ってもOK）"
                }])

            else:
                await line_reply(reply_token, [{
                    "type": "text",
                    "text": f"{mtype} メッセージはまだ未対応です。テキストか画像を送ってください。"
                }])

        except Exception:
            # 例外が出ても“何か返す”ようにしてデバッグ継続
            logging.exception("handler error")
            try:
                await line_reply(reply_token, [{"type":"text","text":"内部エラー：ログを確認します🙇"}])
            except Exception:
                pass

    # ③ LINE 仕様：とにかく 200 を返す
    return "OK"
