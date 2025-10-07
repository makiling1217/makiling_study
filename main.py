from fastapi import FastAPI, Request
import os, json, httpx, re

app = FastAPI()

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
CHANNEL_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
REF_IMAGE_URL = os.getenv("REF_IMAGE_URL")  # 任意。参考画像URLを入れると画像も一緒に送ります

@app.get("/")
def root():
    return {"ok": True}

# ===== 共通ヘルパ =====
async def reply_messages(reply_token: str, messages: list):
    if not CHANNEL_TOKEN:
        print("WARN: no LINE_CHANNEL_ACCESS_TOKEN; skip reply")
        return
    headers = {
        "Authorization": f"Bearer {CHANNEL_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"replyToken": reply_token, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(LINE_REPLY_URL, headers=headers, json=payload)
        print("LINE reply status:", r.status_code, r.text)

def numbered(lines: list) -> str:
    return "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1))

def normalize_minus(s: str) -> str:
    return s.replace("−","-").replace("ー","-").replace("―","-")

# ===== fx-CG50 用ガイド =====
def ops_cheatsheet() -> str:
    """「操作方法」用：キーの位置と押し方のチートシート（番号つき）"""
    basics = [
        "決定/次へ：右下の青い [EXE]",
        "1つ戻る：真ん中列の [EXIT]",
        "入力を全部消す：青い [AC/ON]（電源兼用）",
        "1文字消す：[DEL]",
        "カーソル移動：丸い十字キー（↑↓←→）",
    ]
    numbers = [
        "負の数：**[(-)] → 数字 → [EXE]**（白い[−]は“引き算”なので使わない）",
        "小数点：[•]（ピリオド）",
        "分数： [a b/c] で 3/5 のように入力（表示の切替は [S⇔D]）",
        "平方根： **[SHIFT] + [x²]**（キー上の黄色い「√」）",
        "べき乗： [x^y] で 2^3 など",
    ]
    quadratic = [
        "Main Menu で **EQUATION** を開く（EQUA と略さず『EQUATION』）",
        "画面下のソフトキーで **Poly** → Degree を『2』にする",
        "a を入力 → [EXE]（例：a=-3 は **[(-)] 3 [EXE]**）",
        "b を入力 → [EXE]",
        "c を入力 → [EXE]",
        "解 x₁, x₂ が表示される。必要なら [F6] で表示切替（分数/小数など）",
    ]
    tip = [
        "間違えたら [DEL]、全部やり直すなら [AC/ON]。",
        "[EXIT] は“1つ戻る”。押しすぎたらもう一度 [EXE] で戻れる画面も多い。",
    ]

    msg  = "【fx-CG50 操作チートシート】\n"
    msg += "\n＜基本＞\n"   + numbered(basics)
    msg += "\n\n＜数の入力＞\n" + numbered(numbers)
    msg += "\n\n＜二次方程式（aX²+bX+c=0）＞\n" + numbered(quadratic)
    msg += "\n\n＜コツ＞\n" + numbered(tip)
    return msg

def guide_fx_cg50_quadratic(a=None, b=None, c=None) -> str:
    def has(v): return v is not None and str(v).strip() != ""
    steps = [
        "Main Menu で EQUATION を開く",
        "Poly → Degree=2 を選ぶ",
        f"a を入力{f'（{a}）' if has(a) else ''} → [EXE]",
        f"b を入力{f'（{b}）' if has(b) else ''} → [EXE]",
        f"c を入力{f'（{c}）' if has(c) else ''} → [EXE]",
        "解 x₁, x₂ が出る（[F6] で表示切替）",
    ]
    tips = [
        "負の数は **[(-)] → 数字 → [EXE]**（白い[−]は引き算）",
        "分数は [a b/c]、平方根は **[SHIFT]+[x²]** の「√」",
        "[EXIT] で戻る／[AC/ON] で全消去",
    ]
    msg = "【fx-CG50：aX² + bX + c = 0 の解き方】\n" + numbered(steps) + "\n\n（入力のコツ）\n" + numbered(tips)
    if has(a) and has(b) and has(c):
        msg += f"\n\n入力チェック：a={a}, b={b}, c={c}"
    return msg

ABC_RE = re.compile(r"a\s*=\s*([\-−]?\s*[\d\.]+).*?b\s*=\s*([\-−]?\s*[\d\.]+).*?c\s*=\s*([\-−]?\s*[\d\.]+)", re.I|re.S)
def parse_abc(text: str):
    t = normalize_minus(text)
    m = ABC_RE.search(t)
    if m: return (m.group(1), m.group(2), m.group(3))
    return (None, None, None)

def looks_quadratic(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ["二次", "ax^2", "a x^2", "poly", "equa", "equation"])

def looks_ops(text: str) -> bool:
    t = text.strip()
    keys = ["操作方法", "操作", "使い方", "ヘルプ", "help"]
    return any(k in t for k in keys)

# ===== Webhook =====
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
    if ev.get("type") != "message":
        return {"ok": True}

    m = ev.get("message", {})
    reply_token = ev.get("replyToken")

    # 画像だけ送られてきたら：チートシート＋（あれば）参考画像
    if m.get("type") == "image":
        messages = [{"type":"text","text": ops_cheatsheet()}]
        if REF_IMAGE_URL:
            messages.append({"type":"image",
                             "originalContentUrl": REF_IMAGE_URL,
                             "previewImageUrl": REF_IMAGE_URL})
        await reply_messages(reply_token, messages)
        return {"ok": True}

    # テキスト
    if m.get("type") == "text":
        text = m.get("text","")

        # 「操作方法」トリガ
        if looks_ops(text):
            messages = [{"type":"text","text": ops_cheatsheet()}]
            if REF_IMAGE_URL:
                messages.append({"type":"image",
                                 "originalContentUrl": REF_IMAGE_URL,
                                 "previewImageUrl": REF_IMAGE_URL})
            await reply_messages(reply_token, messages)
            return {"ok": True}

        # 二次方程式トリガ
        if looks_quadratic(text):
            a,b,c = parse_abc(text)
            msg = guide_fx_cg50_quadratic(a,b,c)
            messages = [{"type":"text","text": msg}]
            if REF_IMAGE_URL:
                messages.append({"type":"image",
                                 "originalContentUrl": REF_IMAGE_URL,
                                 "previewImageUrl": REF_IMAGE_URL})
            await reply_messages(reply_token, messages)
            return {"ok": True}

        # それ以外
        messages = [{"type":"text",
                     "text":"番号つきで案内します。\n・『操作方法』→ キー操作のチートシート\n・『二次 a=1 b=-3 c=2』→ aX²+bX+c=0 の解き方\n画像だけ送ってもOKです。"}]
        await reply_messages(reply_token, messages)
        return {"ok": True}

    return {"ok": True}
