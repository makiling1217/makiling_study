from fastapi import FastAPI, Request
import os, json, httpx, re

app = FastAPI()

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
CHANNEL_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
REF_IMAGE_URL = os.getenv("REF_IMAGE_URL")  # 任意：参考画像の https 直リンク

@app.get("/")
def root():
    return {"ok": True}

# ===== 共通ヘルパ =====
async def reply_messages(reply_token: str, messages: list):
    if not CHANNEL_TOKEN:
        print("WARN: no LINE_CHANNEL_ACCESS_TOKEN; skip reply")
        return
    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
    payload = {"replyToken": reply_token, "messages": messages[:5]}  # LINEは最大5件
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(LINE_REPLY_URL, headers=headers, json=payload)
        print("LINE reply status:", r.status_code, r.text)

def numbered(lines: list) -> str:
    return "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1))

# ===== fx-CG50：二次方程式の手順 =====
def guide_fx_cg50_quadratic(a=None, b=None, c=None) -> str:
    def has(v): return v is not None and str(v).strip() != ""
    steps = [
        "Main Menu で「EQUATION」のアイコンを開く（※EQUAと略さない表示）",
        "画面下のソフトキーで「Poly（多項式）」→ Degree を『2』にする",
        f"a を入力{f'（{a}）' if has(a) else ''} → [EXE]",
        f"b を入力{f'（{b}）' if has(b) else ''} → [EXE]",
        f"c を入力{f'（{c}）' if has(c) else ''} → [EXE]",
        "解 x₁, x₂ が表示される（[F6] などで表示切替／分数↔小数の切替が出ることがあります）",
    ]
    tips = [
        "負の数は **青い [(-)] → 数字 → [EXE]**（白い [−] は“引き算”なのでNG）",
        "分数は **[a b/c]** または **÷** を使って 3÷4 のように入力可",
        "小数は **[.]**、平方根は **[√]**、累乗は **[^]**（または x^2 等）",
        "カーソル移動は十字キー、[DEL] で1字消去、[AC/ON] で行全消去、[EXIT] で1画面戻る",
    ]
    msg = "【fx-CG50：aX² + bX + c = 0 の解き方】\n" + numbered(steps) + "\n\n（入力のコツ）\n" + numbered(tips)
    if has(a) and has(b) and has(c):
        msg += f"\n\n入力チェック：a={a}, b={b}, c={c}"
    return msg

# ===== fx-CG50：よく使う操作（「操作方法」コマンド用）=====
def help_fx_cg50_cheatsheet() -> str:
    blocks = []

    blocks.append("【fx-CG50：よく使う操作キー（番号つき）】")
    blocks.append(numbered([
        "モード起動：Main Menu → **EQUATION** アイコン",
        "二次方程式：画面下のソフトキー → **Poly** → **Degree=2** を選ぶ",
        "確定：**[EXE]**、1つ戻る：**[EXIT]**、全消去：**[AC/ON]**、1文字削除：**[DEL]**",
        "カーソル移動：十字キー（◀▲▼▶）／画面下の **[F1]〜[F6]** はソフトキー",
        "負の数：**青い [(-)] → 数字 → [EXE]**（白い [−] は“引き算”）",
        "分数：**[a b/c]** または **÷** を使って 3÷4 などと入力",
        "小数：**[.]** を使って 0.25 などと入力",
        "平方根：**[√]**、二乗：**[x²]**、一般の累乗：**[^]** → 指数 → [EXE]",
        "括弧：**[(] [)]** でグループ化（例：(2+3)×4）",
        "結果画面：x₁/x₂ の切替や **分数↔小数（S⇔D）** が表示される場合は画面下の **Fキー** で操作",
    ]))

    blocks.append("\n【クイック例】\n" + numbered([
        "EQUATION → Poly → Degree=2",
        "a=-3 を入力：**[(-)] 3 [EXE]**",
        "b=1.5 を入力：**1 [.] 5 [EXE]**",
        "c=3/4 を入力：**3 ÷ 4 [EXE]**（または **3 [a b/c] 4 [EXE]**）",
    ]))

    blocks.append("\n※画像を送ってもこの一覧は返ります。必要なら『二次 a=1 b=-3 c=2』のように係数付きで送ると、その値で手順を出します。")

    return "\n".join(blocks)

# ===== テキスト解析 =====
ABC_RE = re.compile(r"a\s*=\s*([\-−]?\s*[\d\.]+).*?b\s*=\s*([\-−]?\s*[\d\.]+).*?c\s*=\s*([\-−]?\s*[\d\.]+)", re.I | re.S)
def normalize_minus(s: str) -> str:
    return s.replace("−","-").replace("ー","-").replace("―","-")
def parse_abc(text: str):
    t = normalize_minus(text)
    m = ABC_RE.search(t)
    if m: return (m.group(1), m.group(2), m.group(3))
    return (None, None, None)

def looks_quadratic(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ["二次", "ax^2", "a x^2", "poly", "equa", "equation"])

def is_help_command(text: str) -> bool:
    t = text.strip().lower()
    return any(k in t for k in ["操作方法", "使い方", "ヘルプ", "help"])

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
    reply_token = ev["replyToken"]

    # ---- テキスト ----
    if m.get("type") == "text":
        text = m.get("text", "")

        # ①「操作方法」コマンド → チートシート返信（+任意画像）
        if is_help_command(text):
            msgs = [{"type":"text","text": help_fx_cg50_cheatsheet()}]
            if REF_IMAGE_URL:
                msgs.append({"type":"image","originalContentUrl": REF_IMAGE_URL,"previewImageUrl": REF_IMAGE_URL})
            await reply_messages(reply_token, msgs)
            return {"ok": True}

        # ② 二次方程式ガイド
        if looks_quadratic(text):
            a,b,c = parse_abc(text)
            msg = guide_fx_cg50_quadratic(a,b,c)
            msgs = [{"type":"text","text": msg}]
            if REF_IMAGE_URL:
                msgs.append({"type":"image","originalContentUrl": REF_IMAGE_URL,"previewImageUrl": REF_IMAGE_URL})
            await reply_messages(reply_token, msgs)
            return {"ok": True}

        # ③ その他のテキスト
        await reply_messages(reply_token, [{
            "type":"text",
            "text":"番号つきで案内します。\n- 画像だけ送ってもOK\n- 『操作方法』でキーの一覧\n- 『二次 a=1 b=-3 c=2』で係数入り手順"
        }])
        return {"ok": True}

    # ---- 画像：常にチートシート（+任意画像）を返す ----
    if m.get("type") == "image":
        msgs = [{"type":"text","text": help_fx_cg50_cheatsheet()}]
        if REF_IMAGE_URL:
            msgs.append({"type":"image","originalContentUrl": REF_IMAGE_URL,"previewImageUrl": REF_IMAGE_URL})
        await reply_messages(reply_token, msgs)
        return {"ok": True}

    return {"ok": True}
