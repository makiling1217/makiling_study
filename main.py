import os, json, base64, hmac, hashlib, logging, re
from typing import Any, Dict, List
from fastapi import FastAPI, Request, HTTPException
import httpx

# ====== 基本設定 ======
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_CONTENT_URL = "https://api.line.me/v2/bot/message/{messageId}/content"

HEADERS_JSON = {
    "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    "Content-Type": "application/json",
}

# ====== 署名検証 ======
def verify_signature(body: bytes, signature: str) -> bool:
    if not LINE_CHANNEL_SECRET:
        logging.error("LINE_CHANNEL_SECRET is empty")
        return False
    mac = hmac.new(LINE_CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    expect = base64.b64encode(mac).decode()
    ok = hmac.compare_digest(expect, signature or "")
    if not ok:
        logging.error("Signature NG")
    return ok

# ====== LINE 返信 ======
async def line_reply(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    async with httpx.AsyncClient(timeout=20) as ac:
        r = await ac.post(LINE_REPLY_URL, headers=HEADERS_JSON, json={
            "replyToken": reply_token, "messages": messages[:5]
        })
        r.raise_for_status()
        
# ===== LINE 画像/動画/音声を取得（bytesを返す） =====
import os, httpx

LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]

async def download_line_content(message_id: str) -> bytes:
    """message_id からコンテンツ(画像等)をダウンロード"""
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"  # ← api-data が正解
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()            # 404/401 はここで例外になります
        return r.content

# ====== 画像バイト取得 ======
async def get_line_image_bytes(message_id: str) -> bytes:
    url = LINE_CONTENT_URL.format(messageId=message_id)
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.get(url, headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"})
        r.raise_for_status()
        return r.content

# ====== fx-CG50 手順テンプレ（問題に応じて返す） ======
def cg50_steps_for_text(text: str) -> str:
    t = text.replace(" ", "")
    # 1) 放物線 y=-x^2+4ax+b 系
    if ("y=-x^2+4ax+b" in t) or ("放物線" in t and "4ax+b" in t):
        return (
            "【fx-CG50 操作】\n"
            "1) MENU → GRAPH → EXE（FUNC 画面）\n"
            "2) Y1 を ON（カーソルを Y1 行→F1[SELECT] で左の＝が濃く）\n"
            "3) Y1 に次を入力 → EXE\n"
            "   [(-)] → [X,θ,T] → [x²] → [+] → 4 → [×] → [ALPHA][log](A) → [×] → [X,θ,T] → [+] → [ALPHA][ln](B)\n"
            "   ※ [ALPHA][log]＝A，[ALPHA][ln]＝B\n"
            "4) A, B を入れる：MENU → RUN-MAT\n"
            "   例: A=0.5 は 0 . 5 → [SHIFT][RCL](STO▶) → [ALPHA][log](A) → EXE\n"
            "       B=4   は 4 → [SHIFT][RCL] → [ALPHA][ln](B) → EXE\n"
            "5) MENU → GRAPH → F6[DRAW]\n"
            "6) 頂点は SHIFT+F5[G-Solv] → MAX（下向き放物線）で読み取る\n"
        )
    # 2) 勝率 p=1/3 の勝敗確率（Aチーム3勝0敗/5試合3勝2敗 など）
    if ("勝率" in t or "確率" in t) and ("1/3" in t or "１/３" in t or "1÷3" in t):
        return (
            "【fx-CG50 操作（RUN-MATのみ）】\n"
            "a) 3戦全勝： (1÷3) を括弧で →  ( 1 ÷ 3 ) → [SHIFT][^] → 3 → EXE\n"
            "b) 5戦3勝2敗： 5C3×(1/3)^3×(2/3)^2 を計算\n"
            "   5 × 4 ÷ 2 ÷ 1 → EXE で 10 を得ても良い（=5C3）\n"
            "   10 × ( 1 ÷ 3 ) [SHIFT][^] 3 × ( 2 ÷ 3 ) [SHIFT][^] 2 → EXE\n"
        )
    # デフォルト
    return (
        "【fx-CG50 基本】\n"
        "・関数：MENU→GRAPH、方程式：MENU→EQUA、数値計算：MENU→RUN-MAT\n"
        "・グラフは式を Y1 に入れて EXE → F6[DRAW]、解読は SHIFT+F5[G-Solv]\n"
        "・係数を文字 A,B にして RUN-MAT で A=… を STO▶ 代入 → 再描画\n"
    )

# ====== 画像を GPT で読んで解く ======
async def solve_from_image(img_bytes: bytes) -> str:
    if not OPENAI_API_KEY:
        return "（サーバ設定：OPENAI_API_KEY 未設定）"

    b64 = base64.b64encode(img_bytes).decode()
    # OpenAI Chat Completions（Vision）
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "system",
            "content": (
                "あなたは日本の大学数学・高校数学・中学数学の世界一の解説アシスタントです。"
                "A4サイズまでの試験プリント画像もしくはテキスト画像が来ます。最大2問まで。"
                "各問について必ず次の形式で細かく丁寧に返してください：\n"
                "【問題】(OCRした日本語)\n"
                "【答え】(数値や式。分数は既約が必須)\n"
                "【考え方】(20行以内)\n"
                "【電卓手順】fx-CG50用のキー列を角括弧で具体的に（例：[(-)] [X,θ,T] [x²] ...、[SHIFT][RCL] など。EXEを入れる場所も）\n"
            )
        },{
            "role":"user",
            "content":[
                {"type":"text","text":"この画像の数学問題を読み取り、上の形式で日本語で出力してください。"},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        "temperature": 0.2,
        "max_tokens": 1200
    }

    async with httpx.AsyncClient(timeout=60) as ac:
        r = await ac.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json=payload
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]

    # 足りない場合に補助の定型手順をつける（よく出る2タイプの判定）
    extra = cg50_steps_for_text(text)
    return text + "\n\n" + extra
from fastapi import FastAPI, Request
import httpx

app = FastAPI()

async def line_reply(reply_token: str, messages: list[dict]):
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {"replyToken": reply_token, "messages": messages}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()

@app.post("/webhook")
async def webhook(req: Request):
    body = await req.json()
    for ev in body.get("events", []):
        if ev.get("type") == "message":
            msg = ev["message"]

            # --- 画像メッセージ ---
            if msg["type"] == "image":
                print("message.id =", msg["id"])  # ← ログに出す（後でcurlで使える）

                cp = msg.get("contentProvider", {"type": "line"})
                if cp["type"] == "line":
                    data = await download_line_content(msg["id"])
                else:
                    # 外部URLの画像（contentProvider: external）の場合
                    async with httpx.AsyncClient(timeout=30) as client:
                        r = await client.get(cp["originalContentUrl"])
                        r.raise_for_status()
                        data = r.content

                # ここで OCR/数式読取→解答→電卓手順を生成する処理に渡す
                # 今は到達確認としてサイズを返す
                await line_reply(
                    ev["replyToken"],
                    [{"type": "text",
                      "text": f"画像OK: {len(data)} bytes 取得。message.id={msg['id']}"}]
                )
                continue

            # --- それ以外（テキスト等）は既存処理 ---
            # await line_reply(...)

    return {"ok": True}

# ====== ルーティング ======
@app.get("/")
def hello(): return {"ok": True}

@app.post("/webhook")
async def webhook(request: Request):
    body_bytes = await request.body()
    if not verify_signature(body_bytes, request.headers.get("x-line-signature", "")):
        raise HTTPException(status_code=400, detail="Bad signature")

    body = json.loads(body_bytes.decode("utf-8"))
    events = body.get("events", [])
    logging.info("events=%s", json.dumps(events, ensure_ascii=False))

    for ev in events:
        if ev.get("type") != "message":
            continue

        m = ev.get("message", {})
        mtype = m.get("type")
        reply_token = ev.get("replyToken")

        try:
            if mtype == "image":
                # 1) 画像取得 → 2) GPT でOCR+解答 → 3) 返信
                img = await get_line_image_bytes(m.get("id"))
                answer = await solve_from_image(img)
                await line_reply(reply_token, [{"type":"text","text": answer[:4900]}])

            elif mtype == "text":
                txt = m.get("text", "")
                # 画像なしでも、問題テキストならそのまま補助の手順を付けて返す
                extra = cg50_steps_for_text(txt)
                await line_reply(reply_token, [{"type":"text","text": f"受信：{txt}\n\n{extra}"}])

            else:
                await line_reply(reply_token, [{"type":"text","text": f"{mtype} には未対応です。"}])

        except Exception as e:
            logging.exception("handler error")
            try:
                await line_reply(reply_token, [{"type":"text","text": f"内部エラー：{e}"}])
            except Exception:
                pass
                

    return "OK"

# main.py
from fastapi import FastAPI, Request, BackgroundTasks, Response

app = FastAPI()

@app.get("/")
def health():
    return {"ok": True}

def handle_events(body_bytes: bytes, signature: str):
    # ここで署名検証・返信など重い処理を実施
    # ...（既存のロジックを丸ごと移動）...
    pass

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    sig  = request.headers.get("x-line-signature", "")
    # すぐにバックグラウンドで処理
    background_tasks.add_task(handle_events, body, sig)
    # LINEには**即**200を返す
    return Response(status_code=200)
# main.py
import os
import logging
from fastapi import FastAPI, Request, BackgroundTasks, Response

# 必要なら LINE SDK などの import
# from linebot import LineBotApi, WebhookHandler
# from linebot.exceptions import InvalidSignatureError

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# 任意：ヘルスチェック
@app.get("/")
def health():
    return {"ok": True}

# ここに “重い処理/外部呼び出し” を全部集約
def process_line_events(body_bytes: bytes, signature: str):
    try:
        body_text = body_bytes.decode("utf-8")

        # === あなたの既存ロジックをここに移す ===
        # 例:
        # channel_secret = os.environ["LINE_CHANNEL_SECRET"]
        # channel_token  = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
        # handler = WebhookHandler(channel_secret)
        # line_bot_api = LineBotApi(channel_token)
        # handler.handle(body_text, signature)   # ここでイベント分岐・返信など
        # =======================================

        logger.info("LINE events processed.")
    except Exception as e:
        logger.exception(f"process_line_events error: {e}")

# Webhook: 即 200 を返して起動時間・外部待ちを回避
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    signature = request.headers.get("x-line-signature", "")
    # 実処理はバックグラウンドへ
    background_tasks.add_task(process_line_events, body, signature)
    # LINE へは**すぐ** 200 を返す（検証・本番ともタイムアウト回避）
    return Response(status_code=200)



