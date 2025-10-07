from fastapi import FastAPI, Request
import os, json, httpx, re

app = FastAPI()

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
CHANNEL_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

@app.get("/")
def root():
    return {"ok": True}

# ---------- LINE 返信 ----------
async def line_reply(reply_token: str, text: str):
    if not CHANNEL_TOKEN:
        print("WARN: no LINE_CHANNEL_ACCESS_TOKEN; skip reply")
        return
    headers = {
        "Authorization": f"Bearer {CHANNEL_TOKEN}",
        "Content-Type": "application/json"
    }
    body = {"replyToken": reply_token, "messages": [{"type": "text", "text": text[:4900]}]}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(LINE_REPLY_URL, headers=headers, json=body)
        print("LINE reply status:", r.status_code, r.text)

# ---------- “二次方程式ステップ” 生成 ----------
ABC_RE = re.compile(
    r"a\s*=\s*([\-−]?\s*[\d\.]+).*?b\s*=\s*([\-−]?\s*[\d\.]+).*?c\s*=\s*([\-−]?\s*[\d\.]+)",
    re.I | re.S
)

def normalize_minus(s: str) -> str:
    return s.replace("−", "-").replace("ー", "-").replace("―", "-")

def parse_abc(text: str):
    t = normalize_minus(text)
    m = ABC_RE.search(t)
    if m: return (m.group(1), m.group(2), m.group(3))
    def one(label):
        mm = re.search(rf"{label}\s*=\s*([\-−]?\s*[\d\.]+)", t, re.I)
        return mm.group(1) if mm else None
    return (one("a"), one("b"), one("c"))

def numbered(lines):
    return "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1))

def build_fx_cg50_quadratic(a=None, b=None, c=None) -> str:
    steps = []
    steps.append("Main Menu で「EQUATION（EQUA）」を開く")
    steps.append("下のソフトキーで「Poly（多項式）」→ Degree を『2』にする")
    steps.append(f"a を入力{f'（{a}）' if a else ''} → [EXE]")
    steps.append(f"b を入力{f'（{b}）' if b else ''} → [EXE]")
    steps.append(f"c を入力{f'（{c}）' if c else ''} → [EXE]")
    steps.append("解 x₁, x₂ が表示される（[F6] で表示切替／小数⇔分数が出る場合あり）")

    tips = []
    tips.append("負の数は **[(-)] → 数字 → [EXE]**（白い [−] は“引き算”用）")
    tips.append("分数は [a b/c] または ÷、√ は [√]")
    tips.append("[EXIT] で一つ戻る、[AC/ON] で入力をクリア")

    msg = []
    msg.append("【fx-CG50：二次方程式 aX² + bX + c = 0】")
    msg.append(numbered(steps))
    msg.append("")
    msg.append("（入力のコツ）")
    msg.append(numbered(tips))
    if any(x and "-" in normalize_minus(x) for x in (a, b, c)):
        msg.append("\n※ 今回は負の係数があるので、必ず [(-)] から入力してください。")
    if a and b and c:
        msg.append(f"\n入力チェック：a={a}, b={b}, c={c}")
    return "\n".join(msg)

def want_quadratic(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ["二次", "ax^2", "a x^2", "equa", "equation", "poly", "多項式"])

# ---------- Webhook ----------
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
    if ev.get("type") == "message" and ev.get("message", {}).get("type") == "text":
        text = ev["message"]["text"]
        reply_token = ev["replyToken"]

        if want_quadratic(text):
            a, b, c = parse_abc(text)
            guide = build_fx_cg50_quadratic(a, b, c)
            await line_reply(reply_token, guide)
            return {"ok": True"}

        hint = (
            "メモ：負の数は [(-)]、引き算は [−]。\n"
            "例）「二次 a=1 b=-3 c=2」と送ると、番号つき手順で返信します。"
        )
        await line_reply(reply_token, f"あなた：{text}\n\n{hint}")
        return {"ok": True}

    if ev.get("type") == "message":
        await line_reply(
            ev["replyToken"],
            "画像を受信しました。係数が分かる場合は「二次 a=… b=… c=…」と送ると、番号つき手順で案内します。"
        )
        return {"ok": True}

    return {"ok": True}
