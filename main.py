from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
import os, httpx, json

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True}

# 画像配信用（Render上で https://<your>.onrender.com/image/fx-guide で取得）
@app.get("/image/fx-guide")
def image_fx_guide():
    # リポジトリに static/fx_guide.png を置くこと（PNG/JPGどちらでもOK。10MB以下）
    path = "static/fx_guide.png"
    if not os.path.exists(path):
        # 画像が未配置でも落ちないように 404 ではなく軽いプレースホルダ説明を返す
        # 実運用では用意しておくのが前提
        return {"error": "Put your image at static/fx_guide.png in the repo."}
    return FileResponse(path, media_type="image/png")

@app.post("/webhook")
async def webhook(req: Request):
    """
    LINE Verify 対策：空/非JSONでも 200 を返す。
    テキスト: エコー。キーワード(写真/fx/キー配置)を含むと画像つき返信。
    """
    # ── リクエストボディの安全パース ──
    try:
        raw = await req.body()
        body = json.loads(raw) if raw else {}
    except Exception:
        body = {}

    print("WEBHOOK:", body)

    events = body.get("events", [])
    if not events:
        return {"ok": True}

    ev = events[0]
    if ev.get("type") != "message":
        return {"ok": True}

    msg = ev.get("message", {})
    reply_token = ev.get("replyToken")
    if not reply_token:
        return {"ok": True}

    # LINEチャネルトークン（Renderの Environment Variables に設定済みの想定）
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not token:
        print("WARN: no LINE_CHANNEL_ACCESS_TOKEN; skip reply")
        return {"ok": True}

    # 返信ヘッダ
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # ベースURL（Renderのドメインを自動取得）
    base_url = str(req.base_url).rstrip("/")

    # --- メッセージ種別ごとに返信 ---
    messages = []

    if msg.get("type") == "text":
        user_text = (msg.get("text") or "").strip()
        # まずはエコー
        messages.append({"type": "text", "text": f"あなた：{user_text}"})

        # キーワードで画像も付ける
        if any(k in user_text for k in ["写真", "fx", "キー配置", "画像", "ガイド"]):
            img_url = f"{base_url}/image/fx-guide"
            messages.append({
                "type": "image",
                "originalContentUrl": img_url,
                "previewImageUrl": img_url
            })

    elif msg.get("type") == "image":
        # 画像が送られてきたときは受領メッセージ＋ガイド画像を返す
        messages = [
            {"type": "text", "text": "画像を受け取りました。参考画像を送りますね。"},
            {
                "type": "image",
                "originalContentUrl": f"{base_url}/image/fx-guide",
                "previewImageUrl": f"{base_url}/image/fx-guide"
            }
        ]
    else:
        messages = [{"type": "text", "text": "対応していないメッセージ種別です。"}]

    # 送信
    payload = {"replyToken": reply_token, "messages": messages}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                "https://api.line.me/v2/bot/message/reply",
                headers=headers,
                json=payload
            )
        print("LINE reply status:", r.status_code, r.text)
    except Exception as e:
        print("ERROR on reply:", e)

    return {"ok": True}
