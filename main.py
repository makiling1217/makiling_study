from fastapi import FastAPI, Request
import os, httpx

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    print("WEBHOOK:", body)  # 受けたイベントを確認

    try:
        events = body.get("events", [])
        if events:
            event = events[0]
            if event.get("type") == "message" and event.get("message", {}).get("type") == "text":
                reply_token = event.get("replyToken")
                user_text = event["message"]["text"]
                await line_reply(reply_token, f"あなた: {user_text}")
    except Exception as e:
        print("handler error:", e)

    return {"ok": True}

async def line_reply(reply_token: str, text: str):
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    if not token:
        print("ERROR: 環境変数 LINE_CHANNEL_ACCESS_TOKEN が未設定")
        return
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"replyToken": reply_token, "messages": [{"type": "text", "text": text}]}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, headers=headers, json=payload)
            print("LINE reply status:", r.status_code, r.text)
    except Exception as e:
        print("httpx error:", e)
