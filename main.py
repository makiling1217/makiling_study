# main.py — fx-CG50 × LINE Bot（画像→Vision 安定版 v2）
from fastapi import FastAPI, Request
import os, re, io, json, cmath, asyncio, time, base64
from typing import List, Dict, Any
import httpx
from httpx import Timeout
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

app = FastAPI()

LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")  # 変更可: gpt-4o 等

# ---------------- LINE送信 ----------------
async def line_reply(reply_token: str, texts: List[str]):
    if not LINE_TOKEN or not reply_token: return
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    payload = {"replyToken": reply_token,
               "messages": [{"type":"text","text": t[:4900]} for t in texts[:5]]}
    async with httpx.AsyncClient(timeout=Timeout(10, read=45)) as cli:
        r = await cli.post(url, headers=headers, json=payload)
        print("LINE reply status:", r.status_code, r.text[:160])

async def line_push(user_id: str, texts: List[str]):
    if not LINE_TOKEN or not user_id: return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    async with httpx.AsyncClient(timeout=Timeout(10, read=45)) as cli:
        for i in range(0, len(texts), 5):
            payload = {"to": user_id,
                       "messages": [{"type":"text","text": t[:4900]} for t in texts[i:i+5]]}
            r = await cli.post(url, headers=headers, json=payload)
            print("LINE push status:", r.status_code, r.text[:160])

HELP_TEXT = (
    "使い方：\n"
    "1) 問題の写真を送る → 画像解析して『式＋答え＋番号付き手順』（最大2問）\n"
    "2) 係数指定：例) 二次 1 -3 2 / 二次 1,-3,2 / 二次 1.-3.2 / 二次 a=1 b=-3 c=2\n"
    "3) キー操作の一覧：『操作方法』\n"
)

GENERAL_OPS = (
    "【fx-CG50 キー操作の総合ガイド】\n"
    "1. 負の数：［(−)］（×10^x 左）/ 引き算[−]とは別\n"
    "2. 分数↔小数：[SHIFT]→[S⇔D]\n"
    "3. 角度：[SHIFT]→[SETUP]→Angle\n"
    "4. 複素数：[SHIFT]→[SETUP]→Complex: a+bi\n"
    "5. EQUATION：F1戻る / F2 POLY / F3 SIML / F4 次数 / F5 COEF / F6 SOLV\n"
    "6. クリア：[AC/ON]\n"
)

def steps_quadratic(a: float, b: float, c: float) -> str:
    return (
        "【fx-CG50 二次方程式（EQUATION）】\n"
        "1. [MENU] → 『EQUA (Equation)』 → [EXE]\n"
        "2. [F2] POLY → 次数『2』\n"
        f"3. a={a}→[EXE] → b={b}→[EXE] → c={c}→[EXE]（負の数は［(−)］）\n"
        "4. [▼/▲]で x₁, x₂ を確認（[SHIFT]→[S⇔D]で表記切替）\n"
        "5. 複素解時： [SHIFT]→[SETUP]→Complex: a+bi をON\n"
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

# -------- solvers --------
def solve_linear(a: float, b: float) -> Dict[str, Any]:
    if abs(a) < 1e-12:
        return {"equation": f"{a}x + {b} = 0", "kind": "解なし（a=0）", "roots": []}
    return {"equation": f"{a}x + {b} = 0", "kind": "一次方程式", "roots": [f"{(-b/a):.10g}"]}

def solve_quadratic(a: float, b: float, c: float) -> Dict[str, Any]:
    if abs(a) < 1e-12:
        return solve_linear(b, c) | {"as_linear": True}
    D = b*b - 4*a*c
    kind = "D>0（2実数解）" if D > 1e-12 else ("D=0（重解）" if abs(D) <= 1e-12 else "D<0（虚数解）")
    r1 = (-b + cmath.sqrt(D)) / (2*a)
    r2 = (-b - cmath.sqrt(D)) / (2*a)
    def fmt(z): return f"{z.real:.10g}" if abs(z.imag) < 1e-12 else f"{z.real:.10g} + {z.imag:.10g}i"
    return {"equation": f"{a}x^2 + {b}x + {c} = 0", "kind": kind, "roots": [fmt(r1), fmt(r2)]}

# -------- parsing --------
NUM = r"[-+]?\d+(?:\.\d+)?"

def parse_quadratic_command(text: str):
    s = text.replace("，", ",").replace("　", " ").strip()
    if not s.startswith("二次"): return None
    tail = s[2:].strip()

    m = re.search(r"a\s*=\s*("+NUM+").*?b\s*=\s*("+NUM+").*?c\s*=\s*("+NUM+")", tail, re.I)
    if m: return tuple(float(x) for x in m.groups())

    m = re.match(r"^\s*([+-]?\d+)\.\s*([+-]?\d+)\.\s*([+-]?\d+)\s*$", tail)  # 1.-3.2
    if m: return tuple(float(x) for x in m.groups())

    norm = tail
    for ch in [",","、","/","／","・","･",";","；"]:
        norm = norm.replace(ch, " ")
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

# -------- 画像取得と前処理 --------
async def fetch_line_image_content(message_id: str) -> bytes:
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"  # ←重要
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    async with httpx.AsyncClient(timeout=Timeout(10, read=45)) as cli:
        for i in range(3):
            r = await cli.get(url, headers=headers)
            print("get image:", r.status_code)
            if r.status_code == 200:
                return r.content
            await asyncio.sleep(0.8)
        raise RuntimeError(f"content get failed: {r.status_code}")

def preprocess(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    # 200%拡大（上限2048px）
    w,h = img.size
    scale = 2.0
    nw, nh = int(w*scale), int(h*scale)
    if max(nw, nh) > 2048:
        k = 2048 / max(nw, nh); nw, nh = int(nw*k), int(nh*k)
    img = img.resize((nw, nh), Image.LANCZOS)
    # コントラスト強調＋シャープ
    img = ImageOps.autocontrast(img, cutoff=2)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

# -------- Vision 呼び出し（ResponsesAPI → ChatCompletions フェイルバック） --------
async def vision_to_equations(img_bytes: bytes) -> List[str]:
    if not OPENAI_API_KEY:
        print("NO OPENAI_API_KEY"); return []
    b64 = base64.b64encode(img_bytes).decode("ascii")

    # 1) Responses API
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}
    payload_resp = {
        "model": VISION_MODEL,
        "input": [
            {"role":"user","content":[
                {"type":"input_text","text":(
                    "画像から一次/二次方程式を最大2つ、各1行で抽出。"
                    "例: 'x^2-3x+2=0' や '2x+5=0'。説明禁止、必ず '=0' を付ける。"
                )},
                {"type":"input_image","image_url": f"data:image/jpeg;base64,{b64}"}
            ]}
        ],
        "temperature": 0.0,
        "max_output_tokens": 200
    }
    try:
        async with httpx.AsyncClient(timeout=Timeout(15, read=45)) as cli:
            r = await cli.post("https://api.openai.com/v1/responses",
                               headers=headers, json=payload_resp)
            print("OpenAI(responses) status:", r.status_code)
            if r.status_code == 200:
                out = r.json()["output"][0]["content"][0]["text"]
                lines = [ln.strip() for ln in out.splitlines() if "=0" in ln]
                if lines: return lines[:2]
    except Exception as e:
        print("responses api error:", repr(e))

    # 2) Chat Completions fallback
    payload_cc = {
        "model": VISION_MODEL,
        "messages": [
            {"role":"system","content":"Output only equations, one per line."},
            {"role":"user","content":[
                {"type":"text","text":(
                    "画像から一次/二次方程式を最大2つ、各1行のみ。必ず '=0' を付ける。"
                )},
                {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ],
        "temperature": 0.0,
        "max_tokens": 200
    }
    try:
        async with httpx.AsyncClient(timeout=Timeout(15, read=45)) as cli:
            r = await cli.post("https://api.openai.com/v1/chat/completions",
                               headers=headers, json=payload_cc)
            print("OpenAI(chat) status:", r.status_code)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                lines = [ln.strip() for ln in content.splitlines() if "=0" in ln]
                return lines[:2]
    except Exception as e:
        print("chat api error:", repr(e))
    return []

# -------- 画像フロー（バックグラウンド） --------
async def handle_image(message_id: str, user_id: str):
    t0 = time.time()
    try:
        print("image flow: start")
        raw = await fetch_line_image_content(message_id)
        proc = preprocess(raw)

        # 45秒ガード：長引いたら中断してメッセージ
        async def _run():
            return await vision_to_equations(proc)
        try:
            eq_lines = await asyncio.wait_for(_run(), timeout=45.0)
        except asyncio.TimeoutError:
            await line_push(user_id, ["解析がタイムアウトしました。A4を画面いっぱい・正面で撮り直して再送してください。"])
            return

        if not eq_lines:
            await line_push(user_id, ["式を特定できませんでした。コントラストを強めて再送してください。"])
            return

        probs = []
        for ln in eq_lines:
            p = parse_equation_line(ln)
            if p: probs.append(p)
        if not probs:
            await line_push(user_id, ["式の解析に失敗しました。もう一度お試しください。"])
            return

        out: List[str] = []
        for i, p in enumerate(probs, 1):
            if p["type"] == "quadratic":
                s = solve_quadratic(p["a"], p["b"], p["c"])
                out += [
                    f"【問題{i}/{len(probs)}】{s['equation']}",
                    f"種別: {s['kind']}\n解: {', '.join(s['roots'])}",
                    steps_quadratic(p["a"], p["b"], p["c"])
                ]
            else:
                s = solve_linear(p["a"], p["b"])
                out += [
                    f"【問題{i}/{len(probs)}】{s['equation']}",
                    f"種別: {s['kind']}\n解: {', '.join(s['roots']) if s['roots'] else 'なし'}",
                    steps_linear(p["a"], p["b"])
                ]
        await line_push(user_id, out)
        print("image flow: done in", round(time.time()-t0,2), "sec")

    except Exception as e:
        print("image flow error:", repr(e))
        await line_push(user_id, ["画像解析で内部エラーが発生しました。撮り直すか、係数を『二次 1.-3.2』形式で送ってください。"])

# ---------------- FastAPI ----------------
@app.get("/healthz")
def health(): return {"ok": True}

@app.get("/")
def root(): return {"ok": True}

@app.post("/webhook")
async def webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    print("WEBHOOK:", body)

    for ev in body.get("events", []):
        if ev.get("type") != "message": continue
        msg = ev.get("message", {})
        mtype = msg.get("type")
        reply_token = ev.get("replyToken")
        user_id = ev.get("source", {}).get("userId")

        if mtype == "image":
            await line_reply(reply_token, ["解析中…（最大2問）"])
            asyncio.create_task(handle_image(msg.get("id",""), user_id))
            continue

        if mtype == "text":
            text = (msg.get("text") or "").strip()
            if text in ("操作方法","ヘルプ","help"):
                await line_reply(reply_token, [GENERAL_OPS]); continue

            abc = parse_quadratic_command(text)
            if abc:
                a,b,c = abc
                s = solve_quadratic(a,b,c)
                await line_reply(reply_token, [
                    f"【式】{s['equation']}\n【種別】{s['kind']}\n【解】"+(", ".join(s["roots"]) or "なし"),
                    steps_quadratic(a,b,c)
                ])
                continue

            await line_reply(reply_token, [HELP_TEXT])

    return {"ok": True}
