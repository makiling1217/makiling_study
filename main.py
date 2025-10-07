# main.py  —  fx-CG50 × LINE Bot（画像&テキスト両対応）

from fastapi import FastAPI, Request
import os, json, re, math, cmath, base64, io
import httpx
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Dict, Any

LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = FastAPI()

# ---------------- LINE reply ----------------
async def line_reply(reply_token: str, texts: List[str]):
    if not LINE_TOKEN: return
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    payload = {"replyToken": reply_token,
               "messages": [{"type":"text","text":t[:4900]} for t in texts]}
    async with httpx.AsyncClient(timeout=20) as cli:
        r = await cli.post(url, headers=headers, json=payload)
        print("LINE reply status:", r.status_code, r.text[:200])

HELP_TEXT = (
    "使い方：\n"
    "1) 問題の写真を送る → 解析して『式＋答え＋番号付き手順』（最大2問）\n"
    "2) 文字で係数：\n"
    "   例) 二次 1 -3 2 / 二次 1,-3,2 / 二次1,-3,2 / 二次 1.-3.2 / 二次 a=1 b=-3 c=2\n"
    "3) キー操作の一覧：『操作方法』\n"
)

GENERAL_OPERATIONS = (
    "【fx-CG50 キー操作の総合ガイド】\n"
    "1. 負の数入力：‘−3’は［(−)］キー（x10^x左）。引き算[−]とは別。\n"
    "2. 分数/小数切替：[SHIFT]→[S⇔D]\n"
    "3. 角度設定：[SHIFT]→[SETUP]→Angle（Deg/Rad）\n"
    "4. 複素数を許可：[SHIFT]→[SETUP]→Complex: a+bi\n"
    "5. EQUATION種別：画面下の[F1]〜[F6]ラベルに従う（例：F2 POLY / F3 SIML / F5 COEF / F6 SOLV）\n"
    "6. クリア：[AC/ON]\n"
)

def steps_quadratic(a: float, b: float, c: float) -> str:
    return (
        "【fx-CG50 二次方程式（EQUATION）】\n"
        "1. [MENU] → 『EQUA (Equation)』 → [EXE]\n"
        "2. [F2] POLY → 次数『2』を選択\n"
        f"3. 係数入力： a={a} → [EXE] → b={b} → [EXE] → c={c} → [EXE]\n"
        "   ※負の数は［(−)］キー。引き算の[−]と別。\n"
        "4. [▼/▲]で x₁, x₂ を確認。分数/小数は[SHIFT]→[S⇔D]\n"
        "5. 複素解は[SHIFT]→[SETUP]→Complex: a+bi をON\n"
        "6. 下段キー参考：F1戻る / F2 POLY / F3 SIML / F4 DEG選択 / F5 COEF / F6 SOLV\n"
    )

def steps_linear(a: float, b: float) -> str:
    return (
        "【fx-CG50 一次 ax+b=0（EQUATION）】\n"
        "1. [MENU]→EQUA→[EXE] / 2. [F2] POLY → 次数『1』\n"
        f"3. a={a} → [EXE] → b={b} → [EXE]  ※負の数は［(−)］\n"
        "4. 表示された解を確認（[S⇔D]で表記切替）\n"
    )

# ---------------- solver ----------------
def solve_quadratic(a: float, b: float, c: float) -> Dict[str, Any]:
    if abs(a) < 1e-12:
        return solve_linear(b, c) | {"as_linear": True}
    D = b*b - 4*a*c
    kind = "D>0（2実数解）" if D > 1e-12 else ("D=0（重解）" if abs(D) <= 1e-12 else "D<0（虚数解）")
    sqrtD = cmath.sqrt(D)
    r1 = (-b + sqrtD) / (2*a)
    r2 = (-b - sqrtD) / (2*a)
    def fmt(z): return f"{z.real:.10g}" if abs(z.imag)<1e-12 else f"{z.real:.10g} + {z.imag:.10g}i"
    return {"equation": f"{a}x^2 + {b}x + {c} = 0", "kind": kind, "roots": [fmt(r1), fmt(r2)]}

def solve_linear(a: float, b: float) -> Dict[str, Any]:
    if abs(a) < 1e-12:
        return {"equation": f"{a}x + {b} = 0", "kind": "解なし（a=0）", "roots": []}
    return {"equation": f"{a}x + {b} = 0", "kind": "一次方程式", "roots": [f"{(-b/a):.10g}"]}

# ---------------- text command parser ----------------
NUM = r"[-+]?\d+(?:\.\d+)?"

def parse_quadratic_command(text: str):
    s = text.replace("，", ",").replace("　", " ").strip()
    if not s.startswith("二次"): return None

    # a=1 b=-3 c=2
    m = re.search(r"a\s*=\s*("+NUM+").*?b\s*=\s*("+NUM+").*?c\s*=\s*("+NUM+")", s, re.I)
    if m: return tuple(float(x) for x in m.groups())

    # スペース/カンマ/中点（・, ･, ．）区切り
    m = re.search(r"二次[\s]*(.+)", s)
    if m:
        tail = m.group(1)
        tail = tail.replace("・", ".").replace("･", ".").replace("．", ".").replace("/", " ")
        # ピリオドが区切りの可能性に対応： "1.-3.2" → "1 . -3 . 2"
        tail = re.sub(r"(?<=\d)\.(?=[+-]?\d\b)", " . ", tail)  # 区切り用.
        # 3つの数を順に拾う
        cand = re.findall(NUM, tail)
        if len(cand) >= 3:
            try:
                a, b, c = float(cand[0]), float(cand[1]), float(cand[2])
                return (a, b, c)
            except: pass

    # 特殊：完全に「整数.整数.整数」だけのケース（小数点としてではなく区切り）
    m = re.search(r"二次\s*([+-]?\d+)\.([+-]?\d+)\.([+-]?\d+)\b", s)
    if m: return tuple(float(x) for x in m.groups())
    return None

# ---------------- equation text parser ----------------
def parse_equation_line(line: str):
    t = line.replace("−","-").replace("×","x").replace("X","x").replace(" ", "")
    # quadratic: ax^2 + bx + c = 0（a,b,c省略可）
    qm = re.search(r"^([+-]?\d*\.?\d*)x\^?2([+-]\d*\.?\d*)x([+-]\d*\.?\d*)=0$", t)
    if qm:
        def cv(s):
            if s in ("","+"): return 1.0
            if s=="-": return -1.0
            return float(s)
        a = cv(qm.group(1)); b = float(qm.group(2)); c = float(qm.group(3))
        return {"type":"quadratic","a":a,"b":b,"c":c}
    # linear: ax + b = 0
    lm = re.search(r"^([+-]?\d*\.?\d*)x([+-]\d*\.?\d*)=0$", t)
    if lm:
        def cv(s): return 1.0 if s in ("","+") else (-1.0 if s=="-" else float(s))
        a = cv(lm.group(1)); b = float(lm.group(2))
        return {"type":"linear","a":a,"b":b}
    return None

# ---------------- image handling ----------------
async def fetch_line_image(message_id: str) -> bytes:
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    async with httpx.AsyncClient(timeout=30) as cli:
        r = await cli.get(url, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"content get failed: {r.status_code}")
        return r.content

def preprocess(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    w, h = img.size
    scale = 2048 / max(w, h)
    if scale > 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.7)
    img = img.filter(ImageFilter.SHARPEN)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

async def vision_to_equation_lines(img_bytes: bytes) -> List[str]:
    if not OPENAI_API_KEY: return []
    b64 = base64.b64encode(img_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    prompt = (
      "画像内の一次/二次方程式を最大2つ、各1行で厳密に書き出してください。"
      "例: 'x^2-3x+2=0' , '2x+5=0'。余計な語句や説明は一切書かない。"
      "xは必ず小文字x、二次は必ず x^2 を使う。=0 の右側には何も書かない。最大2行。"
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role":"system","content":"You output only equation lines."},
            {"role":"user","content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":data_url}}
            ]},
        ],
        "temperature": 0.1,
        "max_tokens": 200,
    }
    async with httpx.AsyncClient(timeout=60) as cli:
        r = await cli.post("https://api.openai.com/v1/chat/completions",
                           headers=headers, json=payload)
        if r.status_code != 200:
            print("OpenAI status:", r.status_code, r.text[:200]); return []
        content = r.json()["choices"][0]["message"]["content"]
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        # ゴミを弾く（=0 を含むものだけ）
        lines = [ln for ln in lines if "=0" in ln]
        return lines[:2]

# ---------------- FastAPI ----------------
@app.get("/")
def root(): return {"ok": True}

@app.post("/webhook")
async def webhook(req: Request):
    try:
        body = await req.json()
    except: body = {}
    print("WEBHOOK:", body)

    for ev in body.get("events", []):
        rtoken = ev.get("replyToken")
        if ev.get("type") != "message": continue
        msg = ev.get("message", {})
        mtype = msg.get("type")

        # 画像
        if mtype == "image":
            try:
                await line_reply(rtoken, ["解析中…（最大2問）"])
                raw = await fetch_line_image(msg["id"])
                proc = preprocess(raw)
                lines = await vision_to_equation_lines(proc)
                if not lines:
                    await line_reply(rtoken, ["式を特定できませんでした。紙面を大きく、ピントを合わせて再撮影してみてください。"])
                    continue
                probs = []
                for ln in lines:
                    p = parse_equation_line(ln)
                    if p: probs.append(p)
                if not probs:
                    await line_reply(rtoken, ["式の解析に失敗しました。もう一度お試しください。"])
                    continue

                out = []
                for i,p in enumerate(probs,1):
                    if p["type"]=="quadratic":
                        sol = solve_quadratic(p["a"], p["b"], p["c"])
                        out += [f"【問題{i}/{len(probs)}】{sol['equation']}",
                                f"種別: {sol['kind']}\n解: {', '.join(sol['roots'])}",
                                steps_quadratic(p["a"], p["b"], p["c"])]
                    else:
                        sol = solve_linear(p["a"], p["b"])
                        out += [f"【問題{i}/{len(probs)}】{sol['equation']}",
                                f"種別: {sol['kind']}\n解: {', '.join(sol['roots']) if sol['roots'] else 'なし'}",
                                steps_linear(p["a"], p["b"])]
                await line_reply(rtoken, out)

            except Exception as e:
                print("image flow error:", repr(e))
                await line_reply(rtoken, ["画像解析で内部エラーが発生しました。撮り直すか、係数を『二次 1 -3 2』の形で送ってください。"])

        # テキスト
        elif mtype == "text":
            text = (msg.get("text") or "").strip()
            if text in ("操作方法","ヘルプ","help"):
                await line_reply(rtoken, [GENERAL_OPERATIONS]); continue

            abc = parse_quadratic_command(text)
            if abc:
                a,b,c = abc
                sol = solve_quadratic(a,b,c)
                await line_reply(rtoken, [
                    f"【式】{sol['equation']}\n【種別】{sol['kind']}\n【解】"+(", ".join(sol["roots"]) or "なし"),
                    steps_quadratic(a,b,c)
                ])
                continue
            await line_reply(rtoken, [HELP_TEXT])

    return {"ok": True}
