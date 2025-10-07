from fastapi import FastAPI, Request
import os, json, httpx, base64, ast, operator, re, random

app = FastAPI()

# ====== 返信共通 ======
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"

async def line_reply(reply_token: str, text: str):
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    if not token:
        print("ERROR: LINE_CHANNEL_ACCESS_TOKEN not set")
        return
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"replyToken": reply_token, "messages": [{"type": "text", "text": text}]}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(LINE_REPLY_URL, headers=headers, json=payload)
        print("LINE reply status:", r.status_code, r.text)

# ====== fx-CG50N 向け ビジョン要約 ======
SYSTEM_PROMPT = (
    "あなたは CASIO fx-CG50N の操作ガイドです。"
    "ユーザーが送った“数学の問題画像”を読み取り、"
    "fx-CG50Nで解くためのキー操作を **日本語** で、番号付きの短い手順で出力してください。"
    "キーは角括弧で表記: [SHIFT], [ALPHA], [OPTN], [MENU], [EXE], [AC/ON], [x^2], [√], [^], [×], [÷], [−], [+], [=], [DEL] など。"
    "必要ならメニュー遷移も明記（例: [MENU]→[RUN-MAT]）。"
    "式の入力例や注意点があれば最後に1〜2行で補足。"
)

async def solve_from_image_jp(image_bytes: bytes) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "⚠️ 画像の解析キーが未設定です。\n"
            "Render > Environment > Environment Variables > Add で "
            "`OPENAI_API_KEY` をセットしてから再デプロイしてください。"
        )
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # Chat Completions (Vision)
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text",
                 "text": "次の画像の数学の問題を読み取り、fx-CG50Nで解くキー操作を手順で教えて。"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]},
        ],
        "max_tokens": 900,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, json=body)
        if r.status_code != 200:
            print("OpenAI error:", r.status_code, r.text)
            return "ごめんね、画像の解析に失敗しました。もう一度、明るくピントの合った写真で送ってみてね。"
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

# ====== おまけ（テキスト時のエコー＆ミニ機能）======
_ALLOWED_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
                ast.Mult: operator.mul, ast.Div: operator.truediv,
                ast.UAdd: operator.pos, ast.USub: operator.neg}
def _eval_ast(n):
    if isinstance(n, ast.Num): return n.n
    if hasattr(ast, "Constant") and isinstance(n, ast.Constant):
        if isinstance(n.value, (int, float)): return n.value
    if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(n.op)](_eval_ast(n.operand))
    if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(n.op)](_eval_ast(n.left), _eval_ast(n.right))
    raise ValueError

def safe_calc(expr:str):
    if not re.fullmatch(r"[0-9\.\+\-\*/\(\)\s]+", expr):
        raise ValueError
    return _eval_ast(ast.parse(expr, mode="eval").body)

# ====== ルート ======
@app.get("/")
def root():
    return {"status": "ok"}

# ====== Webhook ======
@app.post("/webhook")
async def webhook(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}
    print("WEBHOOK:", body)

    events = body.get("events", [])
    if not events:
        return {"ok": True}

    ev = events[0]
    rtoken = ev.get("replyToken")
    m = ev.get("message", {})
    mtype = m.get("type")

    # 1) 画像が来たら → 画像を取得 → 解析 → 手順を返信
    if ev.get("type") == "message" and mtype == "image":
        msg_id = m.get("id")
        token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
        if not token:
            await line_reply(rtoken, "内部設定エラー：LINEトークン未設定")
            return {"ok": True}

        # LINEの画像データ取得（api-data.line.me）
        img_headers = {"Authorization": f"Bearer {token}"}
        url = f"https://api-data.line.me/v2/bot/message/{msg_id}/content"
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(url, headers=img_headers)
            if r.status_code != 200:
                print("LINE content error:", r.status_code, r.text)
                await line_reply(rtoken, "画像を取得できませんでした。もう一度送ってね。")
                return {"ok": True}
            image_bytes = r.content

        answer = await solve_from_image_jp(image_bytes)
        await line_reply(rtoken, answer)
        return {"ok": True}

    # 2) テキストはオマケのコマンド or エコー
    if ev.get("type") == "message" and mtype == "text":
        text = m.get("text", "").strip()
        if text.lower().startswith("/help") or text == "ヘルプ":
            await line_reply(rtoken,
                "📷 写真を送ると、fx-CG50N での解き方（キー操作手順）を返信します。\n"
                "オマケ: /calc 1+2*3, /dice 2d6")
            return {"ok": True}
        if text.lower().startswith("/calc"):
            expr = text[5:].strip()
            try:
                v = safe_calc(expr)
                await line_reply(rtoken, f"{expr} = {v:.10g}")
            except Exception:
                await line_reply(rtoken, "式は + - * / () と数字だけで書いてね。")
            return {"ok": True}
        # 既定：エコー
        await line_reply(rtoken, f"あなた；{text}")
        return {"ok": True}

    # その他（postback等）
    await line_reply(rtoken, "受け付けました。写真を送ると操作手順を返せます。")
    return {"ok": True}
