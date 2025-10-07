# main.py
from fastapi import FastAPI, Request
import os, httpx, json, re, math

app = FastAPI()

# ---------------- 共通ユーティリティ ----------------
def fmt_num(x: float) -> str:
    s = f"{x:.10g}".rstrip("0").rstrip(".")
    return s if s else "0"

def reply_text(reply_token: str, text: str):
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not token:
        print("ERROR: LINE_CHANNEL_ACCESS_TOKEN 未設定")
        return
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {"replyToken": reply_token, "messages": [{"type": "text", "text": text}]}
    r = httpx.post("https://api.line.me/v2/bot/message/reply",
                   headers=headers, content=json.dumps(body))
    print("LINE reply status:", r.status_code, r.text)

# ---------------- 操作手順（番号付きチートシート） ----------------
CHEATSHEET = (
    "【fx-CG50：二次方程式 aX²+bX+c=0 の操作手順】\n"
    "1. [MENU] → アイコン **EQUATION** を選ぶ\n"
    "2. [F2] Quadratic（aX²+bX+c=0）を選ぶ\n"
    "3. a を入力 → [EXE]\n"
    "4. b を入力 → [EXE]\n"
    "5. c を入力 → [EXE]\n"
    "6. 解が出たら [▲][▼] で切替、[EXE] で確定\n"
    "7. 小ワザ：負号は白い「(−)」キー／引き算は灰色「−」。戻るのは [EXIT]\n"
    "（例：テキストなら「二次 a=1 b=-3 c=2」や「二次 1,-3,2」でもOK）"
)

def send_cheatsheet(reply_token: str):
    reply_text(reply_token, CHEATSHEET)

# ---------------- 係数の抽出と計算 ----------------
NUM = r"[+-]?\s*(?:\d+(?:\.\d+)?|\.\d+)"
re_abc = re.compile(r"二次.*?a\s*=\s*("+NUM+").*?b\s*=\s*("+NUM+").*?c\s*=\s*("+NUM+")", re.IGNORECASE)
re_csv = re.compile(r"二次\s+("+NUM+")\s*[,，]\s*("+NUM+")\s*[,，]\s*("+NUM+")")

def parse_coeffs(text: str):
    t = text.replace("　", " ")
    m = re_abc.search(t)
    if m:
        return tuple(float(x.replace(" ", "")) for x in m.groups())
    m = re_csv.search(t)
    if m:
        return tuple(float(x.replace(" ", "")) for x in m.groups())
    return None

def solve_quadratic(a: float, b: float, c: float) -> str:
    # 1次の特例
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return (
                "【一次（特例）】a=0 かつ b=0\n"
                f"a={fmt_num(a)}, b={fmt_num(b)}, c={fmt_num(c)}\n"
                "→ c=0 なら無数の解／c≠0 なら解なし\n\n"
                "＜操作手順（一次）＞\n"
                "1.[MENU]→EQUATION\n2.[F1] Linear\n3.b 入力→[EXE]\n4.c 入力→[EXE]"
            )
        x = -c/b
        return (
            "【一次（特例）】bx+c=0 を解きます\n"
            f"b={fmt_num(b)}, c={fmt_num(c)}\n"
            f"→ 解：x = -c/b = {fmt_num(x)}\n\n"
            "＜操作手順（一次）＞\n"
            "1.[MENU]→EQUATION\n2.[F1] Linear\n3.b 入力→[EXE]\n4.c 入力→[EXE]"
        )

    D = b*b - 4*a*c
    lines = []
    lines.append("【二次方程式 aX²+bX+c=0 の計算】")
    lines.append(f"a={fmt_num(a)}, b={fmt_num(b)}, c={fmt_num(c)}")
    lines.append(f"1) 判別式 D=b²-4ac = {fmt_num(D)}")

    if D > 0:
        sqrtD = math.sqrt(D)
        x1 = (-b + sqrtD) / (2*a)
        x2 = (-b - sqrtD) / (2*a)
        lines.append("2) D>0 → 異なる実数解")
        lines.append(f"   x₁ = {fmt_num(x1)}")
        lines.append(f"   x₂ = {fmt_num(x2)}")
    elif abs(D) <= 1e-12:
        x = (-b) / (2*a)
        lines.append("2) D=0 → 重解")
        lines.append(f"   x = {fmt_num(x)}")
    else:
        sqrtD = math.sqrt(-D)
        real = (-b) / (2*a)
        imag = sqrtD / (2*a)
        lines.append("2) D<0 → 実数解なし（複素数解）")
        lines.append(f"   x = {fmt_num(real)} ± {fmt_num(imag)} i")

    lines.append("")
    lines.append("＜電卓の操作（番号付き）＞")
    lines.append("1.[MENU]→EQUATION")
    lines.append("2.[F2] Quadratic（aX²+bX+c=0）")
    lines.append("3.a 入力→[EXE]")
    lines.append("4.b 入力→[EXE]")
    lines.append("5.c 入力→[EXE]")
    lines.append("6.[▲][▼] で解を確認、[EXE] で確定")
    lines.append("7. 負号は白い「(−)」、引き算は灰色「−」。[EXIT] で戻る")
    return "\n".join(lines)

# ---------------- ルート ----------------
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/webhook")
async def webhook(request: Request):
    # Verify対策：壊れた/空JSONでも200
    try:
        raw = await request.body()
        body = json.loads(raw) if raw else {}
    except Exception:
        body = {}
    print("WEBHOOK:", body)

    events = body.get("events", [])
    if not events:
        return {"ok": True}

    ev = events[0]
    reply_token = ev.get("replyToken", "")
    if ev.get("type") != "message":
        return {"ok": True}

    msg = ev.get("message", {})
    mtype = msg.get("type")

    # 1) 画像→そのまま手順を返信
    if mtype == "image":
        send_cheatsheet(reply_token)
        return {"ok": True}

    # 2) テキスト→「操作方法」 or 係数 or ヘルプ
    if mtype == "text":
        text = (msg.get("text") or "").strip()

        if text in ("操作方法", "ヘルプ", "使い方"):
            send_cheatsheet(reply_token)
            return {"ok": True}

        coeffs = parse_coeffs(text)
        if coeffs:
            a, b, c = coeffs
            reply_text(reply_token, solve_quadratic(a, b, c))
            return {"ok": True}

        # 既定の使い方ガイド
        reply_text(
            reply_token,
            "使い方：\n"
            "・写真を送る → 番号つきの操作手順を返信\n"
            "・係数で計算 → 例）二次 a=1 b=-3 c=2  または 例）二次 1,-3,2\n"
            "・操作方法を見る → 「操作方法」"
        )
        return {"ok": True}

    return {"ok": True}
