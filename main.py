import os
import hmac
import hashlib
import base64
import io
import json
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image, ImageOps, ImageFilter
import httpx
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

# ====== 環境変数 ======
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ====== OpenAI client ======
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# ---------- ユーティリティ ----------
def verify_line_signature(body: bytes, signature: str) -> None:
    if not LINE_SECRET:
        return
    mac = hmac.new(LINE_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expect = base64.b64encode(mac).decode("utf-8")
    if not hmac.compare_digest(expect, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

def line_headers(json_type: bool = True) -> Dict[str, str]:
    h = {"Authorization": f"Bearer {LINE_TOKEN}"}
    if json_type:
        h["Content-Type"] = "application/json"
    return h

async def line_reply(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    url = "https://api.line.me/v2/bot/message/reply"
    body = {"replyToken": reply_token, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.post(url, headers=line_headers(True), json=body)
        r.raise_for_status()

async def line_push(user_id: str, messages: List[Dict[str, Any]]) -> None:
    url = "https://api.line.me/v2/bot/message/push"
    body = {"to": user_id, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.post(url, headers=line_headers(True), json=body)
        r.raise_for_status()

def pil_enhance_for_ocr(pil: Image.Image) -> Image.Image:
    """
    A4想定：200%拡大 → 自動コントラスト → 軽いシャープ → 疑似二値化（しきい値=自動中央値）
    ※ Pillowのみ。numpy/opencvは未使用。
    """
    # 200%拡大
    w, h = pil.size
    pil = pil.resize((int(w * 2), int(h * 2)), Image.LANCZOS)
    # グレースケール＋自動コントラスト
    g = ImageOps.autocontrast(pil.convert("L"))
    # シャープ
    g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))
    # 簡易二値化（ヒストグラムの重心付近で閾値）
    hist = g.histogram()
    total = sum(hist)
    csum = 0
    thresh = 128
    half = total // 2
    for i, cnt in enumerate(hist):
        csum += cnt
        if csum >= half:
            thresh = i
            break
    bw = g.point(lambda x: 255 if x >= thresh else 0, mode="1").convert("L")
    return bw

def img_to_data_url(pil: Image.Image) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

# ---------- 数学：二次式 ----------
def parse_quadratic(text: str) -> Optional[Tuple[float, float, float]]:
    s = text.strip().replace("　", " ")
    if not s.startswith("二次"):
        return None
    s2 = s[2:].strip()

    # a=1 b=-3 c=2 形式
    if "a=" in s2.lower() or "b=" in s2.lower() or "c=" in s2.lower():
        s2 = s2.replace("，", ",").replace("、", ",")
        parts = s2.replace("A=", "a=").replace("B=", "b=").replace("C=", "c=")
        vals = {}
        for token in parts.split(","):
            token = token.strip()
            if "=" in token:
                k, v = token.split("=", 1)
                k = k.strip().lower()
                try:
                    vals[k] = float(v.strip())
                except:
                    return None
        if {"a", "b", "c"} <= set(vals.keys()):
            return float(vals["a"]), float(vals["b"]), float(vals["c"])
        return None

    # 区切りはカンマ or スペース（ピリオド区切りは不採用）
    tmp = s2.replace("，", ",").replace("、", ",")
    tokens = [t for t in (tmp.split(",") if "," in tmp else tmp.split()) if t]
    nums: List[float] = []
    for t in tokens:
        try:
            nums.append(float(t))
        except:
            pass
    return tuple(nums) if len(nums) == 3 else None

def solve_quadratic(a: float, b: float, c: float) -> Dict[str, Any]:
    if a == 0:
        if b == 0:
            return {"type": "degenerate"}
        return {"type": "linear", "x": -c / b}
    D = b * b - 4 * a * c
    if D > 0:
        r = D ** 0.5
        return {"type": "real2", "x1": (-b + r) / (2 * a), "x2": (-b - r) / (2 * a), "D": D}
    if D == 0:
        return {"type": "double", "x": -b / (2 * a), "D": D}
    r = (-D) ** 0.5
    return {"type": "complex", "real": -b / (2 * a), "imag": r / (2 * a), "D": D}

def fxcg50_steps_for_quadratic(a: float, b: float, c: float) -> str:
    equa = (
        "【EQUA（多項式）で解く】\n"
        "1) [MENU] → EQUA → [EXE]\n"
        "2) [F2](Polynomial) → [EXE]\n"
        "3) degree=2 を選択 → [EXE]\n"
        f"4) a={a} → [EXE],  b={b} → [EXE],  c={c} → [EXE]\n"
        "5) [EXE] または [F6](SOLVE) で解を表示"
    )
    graph = (
        "【GRAPH で根を読む】\n"
        "1) [MENU] → GRAPH → [EXE]\n"
        "2) Y1 行 → [F1](SELECT)（左『＝』を濃く）\n"
        "3) [DEL] で空に → 次を入力 → [EXE]\n"
        "   [-] [X,θ,T] [x²] [+] |b/a| を避け、今回は数値そのまま：\n"
        "   y = a×x^2 + b×x + c（x=[X,θ,T], 二乗=[x²]）\n"
        "4) [F6](DRAW) → [SHIFT][F5](G-Solv) → [F1](ROOT) → 矢印 → [EXE]"
    )
    notes = (
        "《fx-CG50 入門メモ》 A=[ALPHA][log], B=[ALPHA][10^x], C=[ALPHA][ln] / "
        "代入は白い [→] 単押し（SHIFT不要） / 必ず式末で [EXE]"
    )
    return equa + "\n\n" + graph + "\n\n" + notes

VISION_SYSTEM_PROMPT = """\
あなたは日本語の学習支援ボットです。A4用紙の写真（最大2問）から問題文を読み取り、
各問ごとに ①答え ②必要なら途中式 ③fx-CG50のキー列（EXEを含め初心者向け）を日本語で簡潔に出力してください。
各問は『Q1』『Q2』で区切り、最後に一行サマリを付けてください。
"""

def vision_solve_from_images(images: List[Image.Image]) -> str:
    contents = [{"type": "text", "text": VISION_SYSTEM_PROMPT}]
    for im in images:
        enhanced = pil_enhance_for_ocr(im)
        contents.append({"type": "image_url", "image_url": {"url": img_to_data_url(im)}})
        contents.append({"type": "image_url", "image_url": {"url": img_to_data_url(enhanced)}})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": contents}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ---------- FastAPI ----------
@app.get("/")
async def root():
    return PlainTextResponse("ok")

@app.post("/webhook")
async def webhook(request: Request, x_line_signature: str = Header(None)):
    body = await request.body()
    try:
        verify_line_signature(body, x_line_signature or "")
    except HTTPException:
        return JSONResponse({"status": "ng-signature"}, status_code=200)

    data = await request.json()
    for ev in data.get("events", []):
        if ev.get("type") != "message":
            continue

        msg = ev.get("message", {})
        user_id = ev.get("source", {}).get("userId")
        reply_token = ev.get("replyToken")
        mtype = msg.get("type")

        # 即レス
        if reply_token:
            try:
                await line_reply(reply_token, [{"type": "text", "text": "解析中… 少し待ってね。"}])
            except:
                pass

        if mtype == "image":
            try:
                mid = msg.get("id")
                url = f"https://api-data.line.me/v2/bot/message/{mid}/content"
                async with httpx.AsyncClient(timeout=30) as ac:
                    r = await ac.get(url, headers=line_headers(False))
                    r.raise_for_status()
                pil = Image.open(io.BytesIO(r.content)).convert("RGB")
                out = vision_solve_from_images([pil])
                if user_id:
                    await line_push(user_id, [{"type": "text", "text": out}])
            except Exception:
                if user_id:
                    await line_push(user_id, [{
                        "type": "text",
                        "text": "画像の取得/解析に失敗しました。A4を正面・ピント・影少なめで取り直して送ってください。"
                    }])

        elif mtype == "text":
            text = msg.get("text", "").strip()
            abc = parse_quadratic(text)
            if abc is not None:
                a, b, c = abc
                sol = solve_quadratic(a, b, c)
                if sol["type"] == "real2":
                    ans = f"[答] x = {sol['x1']} , {sol['x2']}（D={sol['D']}）"
                elif sol["type"] == "double":
                    ans = f"[答] 重解 x = {sol['x']}（D=0）"
                elif sol["type"] == "linear":
                    ans = f"[答] 一次方程式：x = {sol['x']}"
                elif sol["type"] == "complex":
                    ans = f"[答] x = {sol['real']} ± {abs(sol['imag'])}i（D={sol['D']}）"
                else:
                    ans = "[答] 特殊な係数のため一意に定まりません。"
                steps = fxcg50_steps_for_quadratic(a, b, c)
                text_out = f"a={a}, b={b}, c={c}\n{ans}\n\n{steps}"
                if user_id:
                    await line_push(user_id, [{"type": "text", "text": text_out}])
            else:
                help_txt = (
                    "テキスト例：\n"
                    "・二次 1,-3,2\n"
                    "・二次 1 -3 2\n"
                    "・二次 a=1 b=-3 c=2\n"
                    "（小数OK／ピリオド区切り『1.-3.2』は不可）\n\n"
                    "画像はA4を正面から送ってください（最大2問対応）。"
                )
                if user_id:
                    await line_push(user_id, [{"type": "text", "text": help_txt}])

    return JSONResponse({"status": "ok"})
