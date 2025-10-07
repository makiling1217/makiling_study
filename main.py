# main.py
from fastapi import FastAPI, Request
import os, json, base64, asyncio
import httpx
from typing import List

app = FastAPI()

LINE_REPLY_URL   = "https://api.line.me/v2/bot/message/reply"
LINE_CONTENT_URL = "https://api-data.line.me/v2/bot/message/{messageId}/content"

# ---- 小さなユーティリティ -------------------------------------------------

def chunk_text(s: str, limit: int = 1800) -> List[str]:
    """LINEのテキスト上限対策（ざっくり分割）"""
    out = []
    while s:
        out.append(s[:limit])
        s = s[limit:]
    return out or ["(empty)"]

async def line_reply(reply_token: str, text: str) -> None:
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not token:
        print("WARN: LINE_CHANNEL_ACCESS_TOKEN is missing; skip reply")
        return
    messages = [{"type": "text", "text": part} for part in chunk_text(text, 1800)]
    payload  = {"replyToken": reply_token, "messages": messages}
    headers  = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(LINE_REPLY_URL, headers=headers, json=payload)
        print("LINE reply status:", r.status_code, r.text)

async def fetch_line_image_bytes(message_id: str) -> bytes:
    """LINEの画像バイナリを取得（コンテンツAPI）"""
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not token:
        raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN is missing")
    url = LINE_CONTENT_URL.format(messageId=message_id)
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.content

async def solve_casio_steps_from_image(img_bytes: bytes) -> str:
    """
    OpenAI (Vision) で画像の問題を読み取り、
    CASIO fx-CG50（日本版）でのキー操作手順を日本語で返す。
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return ("⚠ OpenAIのAPIキーが未設定です（OPENAI_API_KEY）。\n"
                "Render の『Environment → Add Environment Variable』から設定してください。")

    b64 = base64.b64encode(img_bytes).decode("utf-8")

    # 指示（日本語・fx-CG50特化）
    system_prompt = (
        "あなたはCASIO fx-CG50（日本版）に詳しいチューターです。"
        "与えられた問題画像を読み取り、電卓で解くための最短のキー操作手順を日本語で出力します。"
        "出力フォーマットは厳守：\n"
        "1) 問題の要約（1行）\n"
        "2) キー操作（箇条書き。キーは[SHIFT] [ALPHA] [MENU] [OPTN] [EXE] [×] [÷] [^] など角括弧で書く。"
        "メニュー遷移は → で表現。数式は電卓入力そのまま。)\n"
        "3) 計算の結果（可能なら数値）\n"
        "4) 補足（注意点や別解があれば1-2行）\n"
        "説明は簡潔に。数式の丸めは指示がなければ有効数字3～4桁程度。"
    )

    user_text = (
        "画像の問題を読み取り、CASIO fx-CG50 での具体的なキー操作だけを丁寧に教えて。"
        "可能なら最終結果も計算して。"
    )

    # Chat Completions（gpt-4o-mini）に画像を渡す
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0.2,
        "max_tokens": 900,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "auto"}}
            ]}
        ],
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            print("OpenAI error:", resp.status_code, resp.text)
            return f"⚠ OpenAIエラー {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return "⚠ 解析結果の取得に失敗しました。画像を少し明るく/鮮明にして再送してみてください。"

# ---- ルーティング ---------------------------------------------------------

@app.get("/")
def root():
    # Renderのヘルスチェック用
    return {"status": "ok"}

@app.post("/webhook")
async def webhook(req: Request):
    """
    LINE Verify 対策：空や非JSONでも 200 を返す。
    実イベント時はテキスト/画像を処理して返信。
    """
    try:
        raw = await req.body()
        body = json.loads(raw) if raw else {}
    except Exception:
        body = {}

    print("WEBHOOK:", body)

    events = body.get("events", [])
    if not events:
        # Verify や空POSTの場合
        return {"ok": True}

    # 1イベントずつ処理
    tasks = [handle_event(ev) for ev in events]
    await asyncio.gather(*tasks)
    return {"ok": True}

# ---- イベント個別処理 -----------------------------------------------------

async def handle_event(event: dict) -> None:
    etype = event.get("type")
    if etype == "message":
        await handle_message_event(event)
    elif etype in ("follow", "memberJoined"):
        # 友だち追加など
        token = event.get("replyToken")
        if token:
            msg = ("友だち追加ありがとう！\n"
                   "このアカウントでは、**問題の写真**を送ると "
                   "CASIO fx-CG50 のキー操作手順を返信します。\n"
                   "テキストでも質問OK。")
            await line_reply(token, msg)

async def handle_message_event(event: dict) -> None:
    msg = event.get("message", {})
    mtype = msg.get("type")
    reply_token = event.get("replyToken")

    # テキスト：案内 or エコー
    if mtype == "text":
        text = msg.get("text", "").strip()
        if text in ("help", "ヘルプ", "使い方"):
            guide = (
                "📸 画像解析モード\n"
                "問題の写真を送ると、CASIO fx-CG50（日本版）で解くためのキー操作を返信します。\n"
                "・文字がはっきり写るよう明るく撮影\n"
                "・計算過程や最終結果も返します（できる範囲で）\n\n"
                "📝 テキストもOK：『sin(30°) は？』など。"
            )
            await line_reply(reply_token, guide)
        else:
            # シンプルにエコー + 案内一行
            await line_reply(reply_token, f"あなた：{text}\n（画像を送るとfx-CG50のキー操作を返すよ）")
        return

    # 画像：OpenAIで解析 → 手順を返信
    if mtype == "image":
        try:
            image_id = msg.get("id")
            img_bytes = await fetch_line_image_bytes(image_id)
            answer = await solve_casio_steps_from_image(img_bytes)
            await line_reply(reply_token, answer)
        except Exception as e:
            print("handle image error:", repr(e))
            await line_reply(
                reply_token,
                "⚠ 画像の取得/解析に失敗しました。もう一度、文字がくっきり写るように撮って送ってください。"
            )
        return

    # 未対応タイプ
    await line_reply(reply_token, "このメッセージタイプにはまだ対応していません。テキストか画像で送ってください。")
