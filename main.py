# main.py — fx-CG50 × LINE Bot（写真/文字どちらもOK・強化版）
from fastapi import FastAPI, Request
import os, json, re, math, cmath, base64, io
import httpx
from httpx import Timeout
from typing import List, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = FastAPI()

# ---------------- LINE reply ----------------
async def line_reply(reply_token: str, texts: List[str]):
    if not LINE_TOKEN:
        print("WARN: LINE token missing"); return
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    payload = {
        "replyToken": reply_token,
        "messages": [{"type":"text","text": t[:4900]} for t in texts]
    }
    async with httpx.AsyncClient(timeout=Timeout(10, read=45, write=20, pool=10)) as cli:
        r = await cli.post(url, headers=headers, json=payload)
        print("LINE reply status:", r.status_code, r.text[:200])

HELP_TEXT = (
    "使い方：\n"
    "1) 問題の写真を送る → 『式＋答え＋番号付き手順』（最大2問）\n"
    "2) 文字で係数：\n"
    "   例) 二次 1 -3 2 / 二次 1,-3,2 / 二次1,-3,2 / 二次 1.-3.2 / 二次 a=1 b=-3 c=2\n"
    "3) キー操作の一覧：『操作方法』\n"
)

GENERAL_OPERATIONS = (
    "【fx-CG50 キー操作の総合ガイド】\n"
    "1. 負の数：‘−3’は［(−)］キー（x10^x左）。引き算[−]とは別。\n"
    "2. 分数/小数切替：[SHIFT]→[S⇔D]\n"
    "3. 角度設定：[SHIFT]→[SETUP]→Angle（Deg/Rad）\n"
    "4. 複素数可：[SHIFT]→[SETUP]→Complex: a+bi\n"
    "5. EQUATION下段：F1戻る / F2 POLY / F3 SIML / F4 次数切替 / F5 COEF / F6 SOLV\n"
    "6. クリア：[AC/ON]\n"
)

def steps_quadratic(a: float, b: float, c: float) -> str:
    return (
        "【fx-CG50 二次方程式（EQUATION）】\n"
        "1. [MENU] →『EQUA (Equation)』→ [EXE]\n"
        "2. [F2] POLY → 次数『2』\n"
        f"3. 係数入力： a={a}→[EXE] → b={b}→[EXE] → c={c}→[EXE]\n"
        "   ※負の数は［(−)］。引き算[−]と別。\n"
        "4. [▼/▲]で x₁, x₂ を確認（[SHIFT]→[S⇔D]で表記切替）\n"
        "5. 複素解は[SHIFT]→[SETUP]→Complex: a+bi をON\n"
        "6. F1戻る / F2 POLY / F3 SIML / F4 次数 / F5 COEF / F6 SOLV\n"
    )

def steps_linear(a: float, b: float) -> str:
    return (
        "【fx-CG50 一次 ax+b=0】\n"
        "1. [MENU]→EQUA→[EXE]\n"
        "2. [F2] POLY → 次数『1』\n"
        f"3. a={a}→[EXE] → b={b}→[EXE]（負の数は［(−)］）\n"
        "4. 解を確認（[S⇔D]で切替）\n"
    )

# ---------------- solver ----------------
def solve_quadratic(a: float, b: float, c: float) -> Dict[str, Any]:
    if abs(a) < 1e-12:
        return solve_linear(b, c) | {"as_linear": True}
    D = b*b - 4*a*c
    kind = "D>0（2実数解）" if D > 1e-12 else ("D=0（重解）" if abs(D) <= 1e-12 else "D<0（虚数解）")
    r1 = (-b + cmath.sqrt(D)) / (2*a)
    r2 = (-b - cmath.sqrt(D)) / (2*a)
    def fmt(z): return f"{z.real:.10g}" if abs(z.imag)<1e-12 else f"{z.real:.10g} + {z.imag:.10g}i"
    return {"equation": f"{a}x^2 + {b}x + {c} = 0", "kind": kind, "roots": [fmt(r1), fmt(r2)]}

def solve_linear(a: float, b: float) -> Dict[str, Any]:
    if abs(a) < 1e-12:
        return {"equation": f"{a}x + {b} = 0", "kind": "解なし（a=0）", "roots": []}
    return {"equation": f"{a}x + {b} = 0", "kind": "一次方程式", "roots": [f"{(-b/a):.10g}"]}

# ---------------- parsers ----------------
NUM = r"[-+]?\d+(?:\.\d+)?"

def parse_quadratic_command(text: str):
    s = text.replace("，", ",").replace("　", " ").strip()
    if not s.startswith("二次"): return None
    tail = s[2:].strip()

    # a=1 b=-3 c=2
    m = re.search(r"a\s*=\s*("+NUM+").*?b\s*=\s*("+NUM+").*?c\s*=\s*("+NUM+")", tail, re.I)
    if m: return tuple(float(x) for x in m.groups())

    # まず「整数.整数.整数」を“区切りドット”として扱う（例: 1.-3.2）
    m = re.match(r"^\s*([+-]?\d+)\.\s*([+-]?\d+)\.\s*([+-]?\d+)\s*$", tail)
    if m: return tuple(float(x) for x in m.groups())

    # カンマ/スペース/スラッシュ/中点をスペースに統一
    norm = tail
    for ch in [",","、","/","／","・","･",";","；"]:
        norm = norm.replace(ch, " ")
    # ここで通常の数取り
    nums = re.findall(NUM, norm)
    if len(nums) >= 3:
        a,b,c = float(nums[0]), float(nums[1]), float(nums[2])
        return (a,b,c)

    return None

def parse_equation_line(line: str):
    t = line.replace("−","-").replace("×","x").replace("X","x").replace(" ", "")
    m = re.match(r"^([+-]?\d*\.?\d*)x\^?2([+-]\d*\.?\d*)x([+-]\d*\.?\d*)=0$", t)
    if m:
        def cv(s): return 1.0 if s in ("","+") else (-1.0 if s=="-" else float(s))
        a = cv(m.group(1)); b = float(m.group(2)); c = float(m.group(3))
        return {"type":"quadratic","a":a,"b":b,"c":c}
    m = re.match(r"^([+-]?\d*\.?\d*)x([+-]\d*\.?\d*)=0$", t)
    if m:
        def cv(s): return 1.0 if s in ("","+") else (-1.0 if s=="-" else float(s))
        a = cv(m.group(1)); b = float(m.group(2))
        return {"type":"linear","a":a,"b":b}
    return None

# ---------------- image handling ----------------
async def fetch_line_image(message_id: str) -> bytes:
    # api-data 固定（404回避）
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    async with httpx.AsyncClient(timeout=Timeout(10, read=45, write=20, pool=10)) as cli:
        r = await cli.get(url, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"content get failed: {r.status_code}")
        return r.content

def preprocess(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    # **必ず200%拡大**（上限2048px）
    w, h = img.size
    scale_min = 2.0
    scale_cap = 2048 / max(w, h)
    scale = max(scale_min, scale_cap) if scale_cap > scale_min else scale_min
    nw, nh = int(w*scale), int(h*scale)
    if max(nw, nh) > 2048:
        k = 2048 / max(nw, nh)
        nw, nh = int(nw*k), int(nh*k)
    img = img.resize((nw, nh), Image.LANCZOS)

    # コントラスト&シャープ
    img = ImageOps.autocontrast(img, cutoff=2)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

async def vision_to_equation_lines(img_bytes: bytes) -> List[str]:
    if not OPENAI_API_KEY:
        print("WARN: OPENAI_API_KEY missing"); return []
    b64 = base64.b64encode(img_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}
    prompt = (
      "画像から一次/二次方程式を最大2つ、各1行だけで厳密に抽出し、"
      "例: 'x^2-3x+2=0' や '2x+5=0' のように出力。説明や余計な文字は書かない。"
      "xは小文字、二次は x^2、必ず '=0' を右端に付ける。"
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role":"system","content":"Output only equations, one per line."},
            {"role":"user","content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":data_url}}
            ]},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    async with httpx.AsyncClient(timeout=Timeout(10, read=60, write=20, pool=10)) as cli:
        r = await cli.post("https://api.openai.com/v1/chat/completions",
                           headers=headers, json=payload)
        if r.status_code != 200:
            print("OpenAI status:", r.status_code, r.text[:300])
            return []
        content = r.json()["choices"][0]["message"]["content"]
        lines = [ln.strip() for ln in content.splitlines() if "=0" in ln]
        return lines[:2]

# ---------------- FastAPI ----------------
@app.get("/")
def root(): return {"ok": True}

@app.post("/webhook")
async def webhook(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}
    print("WEBHOOK:", body)

    for ev in body.get("events", []):
        rtoken = ev.get("replyToken")
        if ev.get("type") != "message": continue
        m = ev.get("message", {})
        mtype = m.get("type")

        if mtype == "image":
            try:
                await line_reply(rtoken, ["解析中…（最大2問）"])
                raw = await fetch_line_image(m["id"])
                proc = preprocess(raw)
                lines = await vision_to_equation_lines(proc)
                if not lines:
                    await line_reply(rtoken, ["式を特定できませんでした。紙面をできるだけ大きく正面から撮って再送してね。"])
                    continue

                probs = []
                for ln in lines:
                    p = parse_equation_line(ln)
                    if p: probs.append(p)

                if not probs:
                    await line_reply(rtoken, ["式の解析に失敗しました。もう一度お試しください。"])
                    continue

                out = []
                for i, p in enumerate(probs, 1):
                    if p["type"] == "quadratic":
                        sol = solve_quadratic(p["a"], p["b"], p["c"])
                        out += [
                            f"【問題{i}/{len(probs)}】{sol['equation']}",
                            f"種別: {sol['kind']}\n解: {', '.join(sol['roots'])}",
                            steps_quadratic(p["a"], p["b"], p["c"])
                        ]
                    else:
                        sol = solve_linear(p["a"], p["b"])
                        out += [
                            f"【問題{i}/{len(probs)}】{sol['equation']}",
                            f"種別: {sol['kind']}\n解: {', '.join(sol['roots']) if sol['roots'] else 'なし'}",
                            steps_linear(p["a"], p["b"])
                        ]
                await line_reply(rtoken, out)

            except Exception as e:
                print("image flow error:", repr(e))
                await line_reply(rtoken, ["画像解析で内部エラーが発生しました。撮り直すか、係数を『二次 1.-3.2』や『二次 1,-3,2』で送ってください。"])

        elif mtype == "text":
            text = (m.get("text") or "").strip()
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
